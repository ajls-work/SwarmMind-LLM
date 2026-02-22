# 数据集（格式、生成与清洗）

## 数据格式

训练数据是一个 JSON 数组，每条数据结构如下：

```json
{
  "instruction": "system prompt",
  "input": "user text",
  "output": "{\"task_type\":\"...\",\"agents_config\":[],\"rl_parameters\":{}}"
}
```

要点：

- `instruction` 用于描述任务约束/输出格式；在本项目中通常与 `input` 合并作为 user 内容，具体由训练脚本控制。
- `output` 在数据文件里是一个字符串，但字符串内容必须是可解析的 JSON 对象。
- `LLM/data_cleaner.py` 会校验 `output` 必须至少包含 `task_type`、`agents_config`、`rl_parameters` 三个字段。

## 文件位置

- 种子数据：`LLM/data/seed_50.json`
- 生成中间文件（断点续跑）：`LLM/data/marl_llm_dataset_temp.json`
- 目标数据集：`LLM/data/marl_llm_dataset_2000.json`
- 清洗后数据集：`LLM/data/marl_llm_dataset_cleaned.json`

## 生成数据（API Self-Instruct）

入口：`LLM/create_data.py`

特点：

- 每次请求会让模型返回一个 JSON 数组，也就是“一次生成多条”，由 `BATCH_SIZE` 控制。
- 程序会定期把 `seed + 已生成数据` 写入 `marl_llm_dataset_temp.json`，再次运行会自动恢复进度。

### 一次 50 条还是 100 条

经验上：

- 50 条通常更稳：响应更小，JSON 更不容易跑偏，超时/429 发生时损失更少。
- 100 条更省请求次数：但更容易遇到超时、限流、以及“输出夹杂解释文字导致 JSON 解析失败”的情况。

建议做法：

- 先用 20-50 跑通并观察错误类型（`timed out`、`HTTP 429`、`JSON 解析失败`）。
- 接口稳定后再尝试提高到 80-100；一旦解析失败明显上升，就退回 50。

## 生成数据（本地 Procedural）

入口：`LLM/create_data_procedural.py`

用途：

- 不依赖 API，快速生成可用的 baseline 数据集用于流程验证或冒烟测试。

## 追加数据

如果你把数据按批次保存成多个 JSON 文件（每个文件都是 list），可以用：

```powershell
.\.venv\Scripts\python.exe LLM\append_data.py LLM\data\batches\batch_1.json
```

默认会追加到 `LLM/data/marl_llm_dataset_2000.json`，也可以用 `--dataset` 指定目标文件。

## 清洗数据

入口：`LLM/data_cleaner.py`

建议开启 `--minify-output`，把 `output` 统一重写为稳定的最小化 JSON（更省 token，也更利于训练一致性）：

```powershell
.\.venv\Scripts\python.exe LLM\data_cleaner.py --minify-output
```

---

# Dataset (Format, Generation, Cleanup)

## Format

The dataset is a JSON array. Each item looks like:

```json
{
  "instruction": "system prompt",
  "input": "user text",
  "output": "{\"task_type\":\"...\",\"agents_config\":[],\"rl_parameters\":{}}"
}
```

Notes:

- `instruction` describes task constraints / output format. In this project it's typically merged into the user message; see the training script for the exact behavior.
- `output` is stored as a string, but the string itself must be a valid JSON object.
- `LLM/data_cleaner.py` validates that `output` contains at least `task_type`, `agents_config`, and `rl_parameters`.

## Paths

- Seed set: `LLM/data/seed_50.json`
- Temp checkpoint (resume): `LLM/data/marl_llm_dataset_temp.json`
- Target dataset: `LLM/data/marl_llm_dataset_2000.json`
- Cleaned dataset: `LLM/data/marl_llm_dataset_cleaned.json`

## API Self-Instruct generation

Entry: `LLM/create_data.py`

Behavior:

- Each API call asks the model to return a JSON array (multiple items per request). `BATCH_SIZE` controls the target count per response.
- The script periodically writes `seed + generated` into `marl_llm_dataset_temp.json` so you can resume by rerunning.

### 50 items vs 100 items per call

In practice:

- 50 is usually safer: smaller responses, fewer JSON drift issues, lower timeout/429 blast radius.
- 100 reduces request count, but increases the chance of timeouts, rate limits, and JSON parsing failures.

Suggested workflow:

- Start with 20-50 and watch your failure modes (`timed out`, `HTTP 429`, `JSON parse failed`).
- If the endpoint is stable, try 80-100; if JSON failures spike, drop back to 50.

## Local procedural generation

Entry: `LLM/create_data_procedural.py`

Use case:

- No API required. Good for a baseline dataset to validate the pipeline quickly.

## Appending batches

If you store batches as multiple JSON files (each file is a list), append them with:

```powershell
.\.venv\Scripts\python.exe LLM\append_data.py LLM\data\batches\batch_1.json
```

By default it appends into `LLM/data/marl_llm_dataset_2000.json`. Use `--dataset` to override.

## Cleaning

Entry: `LLM/data_cleaner.py`

Enable `--minify-output` to rewrite outputs into stable minified JSON (fewer tokens, more consistent training):

```powershell
.\.venv\Scripts\python.exe LLM\data_cleaner.py --minify-output
```
