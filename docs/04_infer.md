# 推理（命令行问答）

入口脚本：`LLM/cli/chat_cli.py`（通用，Windows/Linux/macOS）

默认加载：

- Base：`LLM/models/Qwen3-4B-Instruct-2507`
- LoRA：`LLM/outputs/lora_latest`

如果你把 LoRA 输出到了其他目录，可以用 `--lora` 指定。

## LoRA 启用方式

CLI 默认 `--adapter-mode always`：所有问题都启用 LoRA。

对比基线（不加载 LoRA）：

```powershell
.\.venv\Scripts\python.exe LLM\cli\chat_cli.py --adapter-mode never
```

## 交互模式

```powershell
.\.venv\Scripts\python.exe LLM\cli\chat_cli.py
```

可用命令：

- `/reset`：清空上下文（在需要恢复“无人机任务仅输出 JSON”约束时很有用）
- `/exit`：退出

## 单轮模式

```powershell
.\.venv\Scripts\python.exe LLM\cli\chat_cli.py --query "Write a JSON dispatch plan for a 6v2 scenario." --temperature 0.0
```

## 输出语言（为什么会输出中文）

这类模型通常会“跟随用户语言”。你用中文提问时，即使训练数据主要是英文输出，也可能得到中文回答。

想强制英文输出，做两件事：

1) 用英文提问。
2) 把 system prompt 写死为英文并明确约束：

```powershell
.\.venv\Scripts\python.exe LLM\cli\chat_cli.py --temperature 0.0 --system "You are a UAV swarm tactical planner. Respond in English only. Output JSON only (no extra text)."
```

如果你需要更严格的 JSON（比如下游要直接 `json.loads()`），建议 `--temperature 0.0` 并在 system prompt 里强调 “JSON only”。

不发送 system message：

```powershell
.\.venv\Scripts\python.exe LLM\cli\chat_cli.py --no-system
```

## macOS / MPS（全量加载）

如果你在 macOS 上希望用 MPS 跑，并且不做 4-bit 量化加载，用：

```powershell
python3 LLM/cli/chat_cli_mps.py
```

单轮：

```powershell
python3 LLM/cli/chat_cli_mps.py --query "Give me a short overview of PPO in reinforcement learning." --temperature 0.2
```

## 4-bit 与兼容性

如果你遇到 `bitsandbytes` 相关报错，直接加 `--no-4bit`：

```powershell
.\.venv\Scripts\python.exe LLM\cli\chat_cli.py --no-4bit
```

---

# Inference (CLI Chat)

Entry script: `LLM/cli/chat_cli.py` (cross-platform)

Default paths:

- Base: `LLM/models/Qwen3-4B-Instruct-2507`
- LoRA: `LLM/outputs/lora_latest`

Use `--lora` if your adapter is stored elsewhere.

## LoRA enable/disable

By default the CLI uses `--adapter-mode always` (LoRA enabled for all queries).

Baseline (no LoRA):

```powershell
.\.venv\Scripts\python.exe LLM\cli\chat_cli.py --adapter-mode never
```

## Interactive mode

```powershell
.\.venv\Scripts\python.exe LLM\cli\chat_cli.py
```

Commands:

- `/reset`: clear the conversation state (useful when the UAV-dispatch responses drift from the expected JSON-only format)
- `/exit`: quit

## Single-turn mode

```powershell
.\.venv\Scripts\python.exe LLM\cli\chat_cli.py --query "Write a JSON dispatch plan for a 6v2 scenario." --temperature 0.0
```

## Output language (why Chinese happens)

Models often follow the user's language. If you ask in Chinese, you may get Chinese output even if most training outputs are English.

To force English:

1) Ask in English.
2) Use an explicit English system prompt:

```powershell
.\.venv\Scripts\python.exe LLM\cli\chat_cli.py --temperature 0.0 --system "You are a UAV swarm tactical planner. Respond in English only. Output JSON only (no extra text)."
```

If you need strict JSON for downstream parsing, keep `--temperature 0.0` and keep the system prompt strict ("JSON only").

No system message:

```powershell
.\.venv\Scripts\python.exe LLM\cli\chat_cli.py --no-system
```

## macOS / MPS (full load)

On macOS, if you want to run on MPS and avoid 4-bit quantization, use:

```powershell
python3 LLM/cli/chat_cli_mps.py
```

Single-turn:

```powershell
python3 LLM/cli/chat_cli_mps.py --query "Give me a short overview of PPO in reinforcement learning." --temperature 0.2
```

## 4-bit compatibility

If you hit `bitsandbytes` errors, disable 4-bit:

```powershell
.\.venv\Scripts\python.exe LLM\cli\chat_cli.py --no-4bit
```
