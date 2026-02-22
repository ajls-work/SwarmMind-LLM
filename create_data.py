import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


# ===== 配置 =====
API_KEY_ENV = "LLM_API_KEY"
BASE_URL_ENV = "LLM_BASE_URL"
MODEL_ENV = "LLM_MODEL"

API_KEY = os.environ.get(API_KEY_ENV, "").strip()
# Keep defaults as placeholders to avoid leaking local/service details in the repo.
BASE_URL = os.environ.get(BASE_URL_ENV, "").strip() or "..."
# Optional. If unset, the script will pick the first available model from `/v1/models`.
MODEL = os.environ.get(MODEL_ENV, "").strip()

TARGET_COUNT = 2000
BATCH_SIZE = 50
SAMPLE_SIZE = 3
MAX_RETRIES = 5
SUCCESS_SLEEP_SECONDS = 0
RETRY_BACKOFF_BASE_SECONDS = 5
CHECKPOINT_EVERY = 100
PRINT_EVERY = 1
MAX_CONSECUTIVE_FAILURES = 20
MIN_REQUEST_INTERVAL_SECONDS = 1.0
RESUME_FROM_TEMP = True
PAUSE_ON_QUOTA = False
# 如果需要等待太久则提前退出，避免空耗时间；设为 None 表示始终等待
MAX_PAUSE_SECONDS = None

SEED_FILE = "seed_50.json"
TEMP_FILE = "marl_llm_dataset_temp.json"
OUTPUT_FILE = "marl_llm_dataset_2000.json"


def normalize_base_url(url: str) -> str:
    clean = url.rstrip("/")
    if clean.endswith("/v1"):
        return clean
    return f"{clean}/v1"


def http_json(
    method: str,
    path: str,
    *,
    api_key: str,
    base_url: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    url = f"{base_url}{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = None
    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url=url, method=method, headers=headers, data=data)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error: {exc}") from exc


def list_models(api_key: str, base_url: str) -> list[str]:
    data = http_json("GET", "/models", api_key=api_key, base_url=base_url, timeout=30)
    return [m.get("id") for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]


def pick_model(available: list[str], requested: str) -> str:
    requested = (requested or "").strip()
    if requested:
        if requested in available:
            return requested
        raise RuntimeError(f"指定模型不可用：{requested}。可用模型: {available}")
    if not available:
        raise RuntimeError("未返回任何可用模型（/v1/models 为空）。")
    return available[0]


def build_prompt(samples: list[dict[str, Any]], n_items: int) -> str:
    prompt = (
        "你是一个强化学习与具身智能领域的数据生成专家。"
        f"请模仿下面示例的格式，生成 {n_items} 条全新的无人机集群调度任务数据。\n\n"
        "要求：\n"
        "1. input 必须是自然语言战术意图。\n"
        "2. output 必须是严格 JSON 字符串，且与 input 对应。\n"
        "3. 不要与示例重复，场景可创新（如反潜、电磁静默、诱捕、搜救、星际探索等）。\n"
        "4. 输出必须是严格 JSON 数组，每个元素是 JSON 对象。\n\n"
        "### 示例开始 ###\n"
    )
    for idx, seed in enumerate(samples, start=1):
        prompt += f"【例子 {idx}】\nInput: {seed['input']}\nOutput: {seed['output']}\n\n"
    prompt += "### 示例结束 ###\n\n"
    prompt += (
        "请只输出 JSON 数组，数组元素仅包含 instruction、input、output 三个字段。"
        "不要输出任何额外解释。"
    )
    return prompt


def strip_markdown_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()

def extract_json_text(raw_text: str) -> str:
    cleaned = strip_markdown_fence(raw_text)
    try:
        json.loads(cleaned)
        return cleaned
    except json.JSONDecodeError:
        pass

    array_match = re.search(r"\[[\s\S]*\]", cleaned)
    if array_match:
        return array_match.group(0)

    obj_match = re.search(r"\{[\s\S]*\}", cleaned)
    if obj_match:
        return obj_match.group(0)

    return cleaned


def parse_generated_items(raw_text: str) -> tuple[list[dict[str, Any]], int]:
    cleaned = extract_json_text(raw_text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        snippet = cleaned.replace("\n", " ")[:200]
        raise ValueError(f"JSON 解析失败: {exc}; 片段: {snippet}") from exc

    if isinstance(parsed, dict):
        items = [parsed]
    elif isinstance(parsed, list):
        items = parsed
    else:
        raise ValueError("模型返回不是 JSON 对象或数组")

    valid: list[dict[str, Any]] = []
    dropped = 0
    for item in items:
        if not isinstance(item, dict):
            dropped += 1
            continue
        missing = [key for key in ("instruction", "input", "output") if key not in item]
        if missing:
            dropped += 1
            continue
        if not isinstance(item["output"], str):
            item["output"] = json.dumps(item["output"], ensure_ascii=False)
        valid.append(item)

    if not valid:
        raise ValueError("模型返回无有效数据项")

    return valid, dropped


def parse_duration_seconds(duration_text: str) -> int | None:
    text = duration_text.strip()
    if not text:
        return None

    match = re.fullmatch(r"(\d+(?:\.\d+)?)s", text)
    if match:
        return math.ceil(float(match.group(1)))

    match = re.fullmatch(
        r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?",
        text,
    )
    if not match:
        return None

    hours, minutes, seconds = match.groups()
    total_seconds = 0.0
    if hours:
        total_seconds += float(hours) * 3600
    if minutes:
        total_seconds += float(minutes) * 60
    if seconds:
        total_seconds += float(seconds)

    if total_seconds <= 0:
        return None
    return math.ceil(total_seconds)


def parse_retry_delay_seconds(error_message: str) -> int | None:
    for key in ("retryDelay", "quotaResetDelay", "reset_time", "resetTime"):
        match = re.search(rf'"{key}"\s*:\s*"([^"]+)"', error_message)
        if match:
            parsed = parse_duration_seconds(match.group(1))
            if parsed is not None:
                return parsed

    match = re.search(r'"reset_seconds"\s*:\s*(\d+)', error_message)
    if match:
        return int(match.group(1))

    return None


def is_retriable_error(error_message: str) -> bool:
    msg = error_message.lower()
    retriable_markers = [
        "http 408",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "rate limit",
        "resource_exhausted",
        "quota",
        "model_capacity_exhausted",
        "unavailable",
        "auth_unavailable",
        "timeout",
        "timed out",
        "read timed out",
        "connection timed out",
        "timeouterror",
        "temporarily",
    ]
    return any(marker in msg for marker in retriable_markers)


def is_quota_or_cooldown_error(error_message: str) -> bool:
    msg = error_message.lower()
    markers = [
        "http 429",
        "rate limit",
        "quota",
        "resource_exhausted",
        "quota_exhausted",
        "model_cooldown",
        "cooling down",
        "model_capacity_exhausted",
    ]
    return any(marker in msg for marker in markers)


def is_json_parse_error(error_message: str) -> bool:
    msg = error_message.lower()
    markers = [
        "expecting value",
        "jsondecodeerror",
        "invalid json",
        "not json",
        "json 解析失败",
        "解析失败",
        "不是 json",
        "不是json",
    ]
    return any(marker in msg for marker in markers)


def is_timeout_error(error_message: str) -> bool:
    msg = error_message.lower()
    markers = [
        "timed out",
        "timeout",
        "timeouterror",
        "read timed out",
        "connection timed out",
    ]
    return any(marker in msg for marker in markers)


def is_model_unavailable_error(error_message: str) -> bool:
    msg = error_message.lower()
    markers = [
        "no longer available",
        "model is no longer available",
        "please switch to",
        "switch to",
    ]
    return any(marker in msg for marker in markers)


def save_checkpoint(
    temp_path: Path,
    seed_data: list[dict[str, Any]],
    generated_dataset: list[dict[str, Any]],
) -> None:
    checkpoint = seed_data + generated_dataset
    with temp_path.open("w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    print(
        f"已保存中间文件: {temp_path}，当前总数 {len(checkpoint)}",
        flush=True,
    )


def create_one_item(
    *,
    api_key: str,
    base_url: str,
    model: str,
    prompt: str,
    temperature: float = 0.8,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是严格遵循 JSON 输出格式的数据生成器。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    data = http_json(
        "POST",
        "/chat/completions",
        api_key=api_key,
        base_url=base_url,
        payload=payload,
        timeout=300,
    )

    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"返回中缺少 choices: {data}")

    content = (choices[0].get("message") or {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"返回中缺少 message.content: {data}")
    return content


def main() -> None:
    base_url = normalize_base_url(BASE_URL)
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    seed_path = data_dir / SEED_FILE
    temp_path = data_dir / TEMP_FILE
    output_path = data_dir / OUTPUT_FILE

    if not API_KEY:
        print(f"[WARN] 环境变量 {API_KEY_ENV} 未设置，将使用空 API key 请求。", flush=True)
    if BASE_URL.strip() in {"", "..."}:
        raise ValueError(f"请设置环境变量 {BASE_URL_ENV} 为你的服务地址（例如 ...）。")

    if not seed_path.exists():
        raise FileNotFoundError(f"未找到种子文件: {seed_path}")

    with seed_path.open("r", encoding="utf-8") as f:
        seed_data = json.load(f)

    if not isinstance(seed_data, list) or len(seed_data) < SAMPLE_SIZE:
        raise ValueError(f"seed 数据不足，至少需要 {SAMPLE_SIZE} 条。")

    models = list_models(API_KEY, base_url)
    model = pick_model(models, MODEL)

    generated_dataset: list[dict[str, Any]] = []
    if RESUME_FROM_TEMP and temp_path.exists():
        try:
            with temp_path.open("r", encoding="utf-8") as f:
                temp_data = json.load(f)
            if isinstance(temp_data, list) and len(temp_data) >= len(seed_data):
                generated_dataset = temp_data[len(seed_data) :]
                print(
                    f"检测到中间文件 {temp_path}，已恢复 {len(generated_dataset)} 条生成数据。",
                    flush=True,
                )
        except Exception as exc:
            print(f"[WARN] 读取中间文件失败，已忽略：{exc}", flush=True)

    if len(generated_dataset) >= TARGET_COUNT:
        final_dataset = seed_data + generated_dataset[:TARGET_COUNT]
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(final_dataset, f, ensure_ascii=False, indent=2)
        print(
            f"已有足够数据，当前总数据量 {len(final_dataset)}，输出文件: {output_path}",
            flush=True,
        )
        return

    consecutive_failures = 0
    last_request_start_ts = 0.0
    print(
        f"开始 Self-Instruct 数据生成，目标新增 {TARGET_COUNT} 条，模型 {model}",
        flush=True,
    )
    if generated_dataset:
        print(
            f"将继续生成 {TARGET_COUNT - len(generated_dataset)} 条数据。",
            flush=True,
        )

    current_batch_size = BATCH_SIZE
    while len(generated_dataset) < TARGET_COUNT:
        remaining = TARGET_COUNT - len(generated_dataset)
        batch_size = min(current_batch_size, remaining)

        success = False
        attempt = 0
        item_index = len(generated_dataset) + 1
        while attempt < MAX_RETRIES:
            sampled_seeds = random.sample(seed_data, SAMPLE_SIZE)
            prompt = build_prompt(sampled_seeds, batch_size)
            try:
                # Global request-rate guard: at most one request start per second.
                now = time.monotonic()
                elapsed = now - last_request_start_ts
                if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
                    time.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
                last_request_start_ts = time.monotonic()

                raw_content = create_one_item(
                    api_key=API_KEY,
                    base_url=base_url,
                    model=model,
                    prompt=prompt,
                    temperature=0.8,
                )
                items, dropped = parse_generated_items(raw_content)
                if dropped:
                    print(
                        f"[WARN] 本批次丢弃 {dropped} 条无效数据。",
                        flush=True,
                    )
                if len(items) > remaining:
                    items = items[:remaining]
                generated_dataset.extend(items)
                success = True
                consecutive_failures = 0
                current_batch_size = min(BATCH_SIZE, batch_size + 5) if batch_size < BATCH_SIZE else batch_size

                if len(generated_dataset) % PRINT_EVERY == 0:
                    print(
                        f"进度: {len(generated_dataset)}/{TARGET_COUNT}",
                        flush=True,
                    )

                time.sleep(SUCCESS_SLEEP_SECONDS)
                break
            except Exception as exc:
                raw_msg = str(exc)
                if is_model_unavailable_error(raw_msg):
                    print(
                        f"[ERROR] 第 {item_index} 条提示模型不可用，已保存进度并退出。错误: {exc}",
                        flush=True,
                    )
                    save_checkpoint(temp_path, seed_data, generated_dataset)
                    return
                if is_json_parse_error(raw_msg) and batch_size > 1:
                    new_batch_size = max(5, batch_size // 2)
                    if new_batch_size == batch_size:
                        new_batch_size = max(1, batch_size - 1)
                    if new_batch_size < batch_size:
                        attempt += 1
                        batch_size = new_batch_size
                        current_batch_size = batch_size
                        print(
                            f"[WARN] 第 {item_index} 条解析失败，已将批大小降为 {batch_size} 后重试 "
                            f"({attempt}/{MAX_RETRIES})，错误: {exc}",
                            flush=True,
                        )
                        continue
                if is_timeout_error(raw_msg) and batch_size > 1:
                    new_batch_size = max(5, batch_size // 2)
                    if new_batch_size == batch_size:
                        new_batch_size = max(1, batch_size - 1)
                    if new_batch_size < batch_size:
                        attempt += 1
                        batch_size = new_batch_size
                        current_batch_size = batch_size
                        print(
                            f"[WARN] 第 {item_index} 条请求超时，已将批大小降为 {batch_size} 后重试 "
                            f"({attempt}/{MAX_RETRIES})，模型: {model}，错误: {exc}",
                            flush=True,
                        )
                        continue

                if is_quota_or_cooldown_error(raw_msg):
                    retry_after = parse_retry_delay_seconds(raw_msg)
                    wait_seconds = retry_after if retry_after is not None else RETRY_BACKOFF_BASE_SECONDS
                    if PAUSE_ON_QUOTA:
                        if MAX_PAUSE_SECONDS is not None and wait_seconds > MAX_PAUSE_SECONDS:
                            print(
                                f"[ERROR] 第 {item_index} 条触发配额/冷却，需要等待 {wait_seconds}s，"
                                f"超过阈值 {MAX_PAUSE_SECONDS}s，已保存进度并退出。",
                                flush=True,
                            )
                            save_checkpoint(temp_path, seed_data, generated_dataset)
                            return
                        print(
                            f"[WARN] 第 {item_index} 条触发配额/冷却，等待 {wait_seconds}s 后重试，错误: {exc}",
                            flush=True,
                        )
                        time.sleep(wait_seconds)
                        continue

                    print(
                        f"[ERROR] 第 {item_index} 条触发配额/冷却，已保存进度并退出。错误: {exc}",
                        flush=True,
                    )
                    save_checkpoint(temp_path, seed_data, generated_dataset)
                    return

                if is_retriable_error(raw_msg) and attempt < MAX_RETRIES - 1:
                    retry_after = parse_retry_delay_seconds(raw_msg)
                    wait_seconds = (
                        retry_after
                        if retry_after is not None
                        else ((attempt + 1) * RETRY_BACKOFF_BASE_SECONDS)
                    )
                    attempt += 1
                    print(
                        f"[WARN] 第 {item_index} 条临时失败，等待 {wait_seconds}s 后重试 "
                        f"({attempt}/{MAX_RETRIES})，模型: {model}，批大小: {batch_size}，错误: {exc}",
                        flush=True,
                    )
                    time.sleep(wait_seconds)
                    continue

                print(
                    f"[ERROR] 第 {item_index} 条失败并跳过，错误: {exc}",
                    flush=True,
                )
                break

        if success and len(generated_dataset) % CHECKPOINT_EVERY == 0:
            save_checkpoint(temp_path, seed_data, generated_dataset)

        if not success:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(
                    f"[ERROR] 连续失败达到 {MAX_CONSECUTIVE_FAILURES} 条，停止本轮生成以避免继续丢数据。",
                    flush=True,
                )
                break
            continue

    final_dataset = seed_data + generated_dataset
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(final_dataset, f, ensure_ascii=False, indent=2)

    print(
        f"生成完成，总数据量 {len(final_dataset)}，输出文件: {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
