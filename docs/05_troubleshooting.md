# 故障排查

## 数据生成（`create_data.py`）

### 429 / quota exhausted / model cooldown

现象：

- 日志里出现 `HTTP 429`、`RESOURCE_EXHAUSTED`、`QUOTA_EXHAUSTED`、`model_cooldown` 等。

当前策略（脚本默认）：

- 默认不等待配额恢复：触发 429/冷却后会把进度写入 `LLM/data/marl_llm_dataset_temp.json` 然后退出（避免长时间空等）。

可选策略：

- 如果你确认“等一会就能恢复”，可以把 `LLM/create_data.py` 顶部的 `PAUSE_ON_QUOTA = True` 打开，让脚本按返回的 `retryDelay/reset_time` 等待后继续跑。

### timed out / read timed out

含义：

- 请求超时。通常是网关/代理/上游模型响应慢，或者单次请求太大。

优先排查顺序：

- 降低 `BATCH_SIZE`（例如 50 -> 30 -> 20），减少一次响应体积。
- 检查环境变量 `LLM_BASE_URL` 指向的服务是否稳定、是否有代理/限速。
- 确认本机网络到该服务可达（`/v1/models` 是否能正常返回）。

脚本行为：

- 如果是超时且 `batch_size > 1`，脚本会自动把本轮 batch 砍半后重试，尽量把“失败成本”降下来。

### JSON 解析失败

现象：

- 日志出现 `JSON 解析失败`，通常是模型输出混入了说明文字、代码块标记，或者数组/对象不闭合。

建议：

- 降低 `BATCH_SIZE`（输出越长越容易跑偏）。
- 让上游模型更“死板”：降低 `temperature`（脚本里当前固定为 `0.8`，你可以改小一点）。
- 必要时增加 system 提示的约束（例如强调 “Only output a JSON array.”）。

## 训练（`train/lora_sft_trainer.py`）

### bitsandbytes 相关报错

最直接的解法：

```powershell
.\.venv\Scripts\python.exe LLM\train\lora_sft_trainer.py --no-4bit
```

说明：

- 4-bit 不是必须项，它主要是为了省显存。
- 如果你在 Windows 原生环境里遇到安装/加载困难，优先用 `--no-4bit` 跑通，再考虑 WSL2/Linux。

### CUDA OOM / 显存不够

常用手段：

- 降低 `--max-length`
- 降低 `--batch-size`
- 提高 `--grad-accum`（保持有效 batch 不变）
- 使用 4-bit（如果环境支持）

## 推理（`cli/chat_cli.py`）

### 输出不是 JSON

建议：

- 用 `/reset` 清上下文；
- `--temperature 0.0`；
- system prompt 明确要求 “Output JSON only (no extra text).”

### 输出语言不符合预期（中文/英文混用）

建议：

- 输入语言会显著影响输出语言；想要英文就用英文提问；
- 通过 `--system` 显式约束输出语言（见 `LLM/docs/04_infer.md`）。

---

# Troubleshooting

## Data generation (`create_data.py`)

### 429 / quota exhausted / model cooldown

Symptoms:

- Logs contain `HTTP 429`, `RESOURCE_EXHAUSTED`, `QUOTA_EXHAUSTED`, `model_cooldown`, etc.

Default behavior:

- By default the script does not wait for quota resets: on 429/cooldown it saves progress to `LLM/data/marl_llm_dataset_temp.json` and exits.

Optional behavior:

- If you're fine waiting until the quota resets, set `PAUSE_ON_QUOTA = True` near the top of `LLM/create_data.py` so the script sleeps based on `retryDelay/reset_time` and continues.

### timed out / read timed out

Meaning:

- Request timeout. Usually the gateway/upstream is slow, or the request/response is too large.

What to try first:

- Reduce `BATCH_SIZE` (e.g., 50 -> 30 -> 20).
- Verify the service behind `LLM_BASE_URL` is stable and reachable.
- Confirm `/v1/models` works.

Script behavior:

- On timeout with `batch_size > 1`, it halves the batch and retries to reduce failure blast radius.

### JSON parse failures

Symptoms:

- `JSON parse failed` errors, often caused by extra explanation text, code fences, or malformed JSON.

Suggestions:

- Reduce `BATCH_SIZE`.
- Lower `temperature` (the script currently uses `0.8`; you can dial it down).
- Tighten the system prompt to enforce "JSON array only".

## Training (`train/lora_sft_trainer.py`)

### bitsandbytes errors

Fast workaround:

```powershell
.\.venv\Scripts\python.exe LLM\train\lora_sft_trainer.py --no-4bit
```

Notes:

- 4-bit is optional; it's mainly for saving VRAM.
- On native Windows, it's often easier to run without 4-bit first, then switch to WSL2/Linux if needed.

### CUDA OOM / not enough VRAM

Common knobs:

- Lower `--max-length`
- Lower `--batch-size`
- Increase `--grad-accum`
- Use 4-bit if supported

## Inference (`cli/chat_cli.py`)

### Output is not JSON

Try:

- `/reset` to clear history
- `--temperature 0.0`
- A strict system prompt: "Output JSON only (no extra text)."

### Language mismatch (Chinese vs English)

Try:

- The input language heavily influences the output language.
- Use `--system` to force the language (see `LLM/docs/04_infer.md`).
