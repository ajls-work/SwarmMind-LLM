# LLM：数据生成与 LoRA 微调（Qwen3-4B）

这个目录是一套最小可用的本地微调流水线：

- 数据集生成/追加/清洗：`LLM/create_data.py`、`LLM/create_data_procedural.py`、`LLM/append_data.py`、`LLM/data_cleaner.py`
- 模型下载（ModelScope）：`LLM/download.py`
- LoRA SFT 训练：`LLM/train/lora_sft_trainer.py`
- 本地命令行推理（base + LoRA）：`LLM/cli/chat_cli.py`（macOS/MPS 用 `LLM/cli/chat_cli_mps.py`）

## 目录结构

- `LLM/data/` 数据与中间文件
- `LLM/models/` 本地模型权重（ModelScope 下载，默认不进 git）
- `LLM/outputs/` 训练输出（LoRA、日志，默认不进 git）
- `LLM/train/` 训练代码
- `LLM/cli/` 命令行推理
- `LLM/docs/` 文档（中英双语，中文在上）

## 快速开始（Windows / PowerShell）

依赖见 `LLM/requirements.txt`。`torch` 版本请按你的 CUDA/驱动自行选择。

```powershell
.\.venv\Scripts\python.exe -m pip install -r LLM\requirements.txt
```

1) 下载模型（默认下载 Qwen3 4B 的 instruct 变体）

```powershell
.\.venv\Scripts\python.exe LLM\download.py
```

2) 生成数据（API Self-Instruct）

`LLM/create_data.py` 通过api接口使用大模型完成数据生成。

```powershell
$env:LLM_API_KEY="..."
$env:LLM_BASE_URL="..."
# 可选：指定使用的模型 ID（不设置则从 /v1/models 里取第一个可用的）
# $env:LLM_MODEL="..."
.\.venv\Scripts\python.exe LLM\create_data.py
```

3) 清洗数据（可选，但推荐）

```powershell
.\.venv\Scripts\python.exe LLM\data_cleaner.py --minify-output
```

4) LoRA 微调训练

```powershell
.\.venv\Scripts\python.exe LLM\train\lora_sft_trainer.py
```

5) 命令行问答（加载 base + LoRA）

```powershell
.\.venv\Scripts\python.exe LLM\cli\chat_cli.py
```

默认 `--adapter-mode always`：全程启用 LoRA。需要无人机任务严格输出 JSON 时，用 `--system` 传入约束，或把约束写进训练数据的 `instruction`。

macOS（MPS、全量加载、不做 4-bit）：

```powershell
python3 LLM/cli/chat_cli_mps.py
```

单轮测试：

```powershell
.\.venv\Scripts\python.exe LLM\cli\chat_cli.py --query "Write a JSON dispatch plan for a 4v1 tag scenario." --temperature 0.0
```

## 常见问题（先看文档）

- 429 / 冷却 / 超时：见 `LLM/docs/05_troubleshooting.md`
- 数据格式与 batch 建议：见 `LLM/docs/02_data.md`
- bitsandbytes 与 4-bit：见 `LLM/docs/01_setup.md`
- 推理输出语言（中文/英文）控制：见 `LLM/docs/04_infer.md`

---

# LLM: Dataset Generation & LoRA SFT (Qwen3-4B)

This folder is a minimal local fine-tuning pipeline:

- Dataset generation/append/cleanup: `LLM/create_data.py`, `LLM/create_data_procedural.py`, `LLM/append_data.py`, `LLM/data_cleaner.py`
- Model download (ModelScope): `LLM/download.py`
- LoRA SFT training: `LLM/train/lora_sft_trainer.py`
- Local CLI inference (base + LoRA): `LLM/cli/chat_cli.py` (macOS/MPS: `LLM/cli/chat_cli_mps.py`)

## Layout

- `LLM/data/` datasets and intermediates
- `LLM/models/` local weights (ignored by default)
- `LLM/outputs/` training outputs (ignored by default)
- `LLM/train/` training code
- `LLM/cli/` inference CLI
- `LLM/docs/` docs (bilingual; Chinese first)

## Quickstart (Windows / PowerShell)

Install dependencies (pick a `torch` build that matches your CUDA/driver):

```powershell
.\.venv\Scripts\python.exe -m pip install -r LLM\requirements.txt
```

1) Download models (defaults to the Qwen3 4B instruct variant)

```powershell
.\.venv\Scripts\python.exe LLM\download.py
```

2) Generate data (API Self-Instruct)

`LLM/create_data.py` uses a model API to generate the dataset (endpoint URL: `...`).

```powershell
$env:LLM_API_KEY="..."
$env:LLM_BASE_URL="..."
# Optional: model id (if unset, the script picks the first one from /v1/models)
# $env:LLM_MODEL="..."
.\.venv\Scripts\python.exe LLM\create_data.py
```

3) Clean the dataset (optional but recommended)

```powershell
.\.venv\Scripts\python.exe LLM\data_cleaner.py --minify-output
```

4) Train LoRA

```powershell
.\.venv\Scripts\python.exe LLM\train\lora_sft_trainer.py
```

5) CLI chat (base + LoRA)

```powershell
.\.venv\Scripts\python.exe LLM\cli\chat_cli.py
```

Default `--adapter-mode always`: LoRA is enabled for all queries. For strict JSON in UAV-dispatch tasks, pass your constraint via `--system` or put it into the SFT `instruction`.

macOS (MPS, full model load, no 4-bit):

```powershell
python3 LLM/cli/chat_cli_mps.py
```

Single-turn:

```powershell
.\.venv\Scripts\python.exe LLM\cli\chat_cli.py --query "Write a JSON dispatch plan for a 4v1 tag scenario." --temperature 0.0
```

## Docs

- Rate limits / cooldown / timeouts: `LLM/docs/05_troubleshooting.md`
- Dataset format and batching: `LLM/docs/02_data.md`
- bitsandbytes / 4-bit notes: `LLM/docs/01_setup.md`
- Language control during inference: `LLM/docs/04_infer.md`
