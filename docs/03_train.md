# 训练（LoRA SFT）

入口脚本：`LLM/train/lora_sft_trainer.py`

默认行为：

- Base 模型：`LLM/models/Qwen3-4B-Instruct-2507`
- 数据集（按优先级自动选择）：
  - `LLM/data/marl_llm_dataset_mix.json`
  - `LLM/data/marl_llm_dataset_cleaned.json`
  - `LLM/data/marl_llm_dataset_2000.json`
- 输出：
  - LoRA：`LLM/outputs/lora_latest/`
  - 训练日志与曲线：`LLM/outputs/logs/`

## 最小命令

```powershell
.\.venv\Scripts\python.exe LLM\train\lora_sft_trainer.py
```

如果你的环境没有可用的 4-bit（`bitsandbytes`/CUDA 问题），加上 `--no-4bit`：

```powershell
.\.venv\Scripts\python.exe LLM\train\lora_sft_trainer.py --no-4bit
```

## 常用参数

- 使用清洗后的数据集：

```powershell
.\.venv\Scripts\python.exe LLM\train\lora_sft_trainer.py --dataset LLM\data\marl_llm_dataset_cleaned.json
```

- 调整训练轮数、学习率、上下文长度：

```powershell
.\.venv\Scripts\python.exe LLM\train\lora_sft_trainer.py --epochs 2 --lr 1e-4 --max-length 768
```

- 调整有效 batch（通过 `--batch-size` 与 `--grad-accum`）：

```powershell
.\.venv\Scripts\python.exe LLM\train\lora_sft_trainer.py --batch-size 1 --grad-accum 16
```

## 输出检查

训练完成后，至少应看到：

- `LLM/outputs/lora_latest/adapter_model.safetensors`
- `LLM/outputs/lora_latest/adapter_config.json`

推理时 `LLM/cli/chat_cli.py` 默认会优先加载 `LLM/outputs/lora_latest/`。

---

# Training (LoRA SFT)

Entry script: `LLM/train/lora_sft_trainer.py`

Defaults:

- Base model: `LLM/models/Qwen3-4B-Instruct-2507`
- Dataset (auto-picked by priority):
  - `LLM/data/marl_llm_dataset_mix.json`
  - `LLM/data/marl_llm_dataset_cleaned.json`
  - `LLM/data/marl_llm_dataset_2000.json`
- Outputs:
  - LoRA: `LLM/outputs/lora_latest/`
  - Logs/plots: `LLM/outputs/logs/`

## Minimal command

```powershell
.\.venv\Scripts\python.exe LLM\train\lora_sft_trainer.py
```

If 4-bit isn't available (CUDA/`bitsandbytes` issues), disable it:

```powershell
.\.venv\Scripts\python.exe LLM\train\lora_sft_trainer.py --no-4bit
```

## Common flags

- Use the cleaned dataset:

```powershell
.\.venv\Scripts\python.exe LLM\train\lora_sft_trainer.py --dataset LLM\data\marl_llm_dataset_cleaned.json
```

- Tune epochs, LR, and context length:

```powershell
.\.venv\Scripts\python.exe LLM\train\lora_sft_trainer.py --epochs 2 --lr 1e-4 --max-length 768
```

- Effective batch via `--batch-size` and `--grad-accum`:

```powershell
.\.venv\Scripts\python.exe LLM\train\lora_sft_trainer.py --batch-size 1 --grad-accum 16
```

## Output sanity check

After training, you should at least have:

- `LLM/outputs/lora_latest/adapter_model.safetensors`
- `LLM/outputs/lora_latest/adapter_config.json`

`LLM/cli/chat_cli.py` loads `LLM/outputs/lora_latest/` by default.
