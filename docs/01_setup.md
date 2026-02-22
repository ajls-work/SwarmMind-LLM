# 环境与依赖

## Python 与虚拟环境

- 建议 `Python 3.10+`（本项目脚本默认使用类型标注与较新的标准库写法）。
- 推荐使用项目根目录的虚拟环境运行（Windows 下常见为 `.\.venv\Scripts\python.exe`）。

安装依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r LLM\requirements.txt
```

`torch` 请按你的 CUDA/驱动选择对应版本（CPU 也能跑推理，但会很慢）。

## GPU / 4-bit（bitsandbytes）

训练脚本 `LLM/train/lora_sft_trainer.py` 默认尝试 4-bit 量化加载；如果你的环境没有可用 CUDA 或 `bitsandbytes` 无法正常工作，可以直接禁用：

```powershell
.\.venv\Scripts\python.exe LLM\train\lora_sft_trainer.py --no-4bit
```

推理脚本 `LLM/cli/chat_cli.py` 也支持 `--no-4bit`。

说明：

- Windows 原生环境下，`bitsandbytes` 兼容性经常是问题（与 CUDA、PyTorch、驱动、编译工具链有关）。
- 如果你只想先跑通流程，优先用 `--no-4bit` 或者在 WSL2/Linux 环境训练。

## ModelScope 下载目录

`LLM/download.py` 默认会把权重下载到 `LLM/models/`。你也可以用环境变量覆盖默认路径：

- `MODELSCOPE_LOCAL_DIR`：自定义权重根目录
- `MODELSCOPE_MODEL_REVISION`：可选，指定模型 revision/tag

示例：

```powershell
$env:MODELSCOPE_LOCAL_DIR="D:\\weights\\qwen"
.\.venv\Scripts\python.exe LLM\download.py
```

---

# Environment & Dependencies

## Python & venv

- Recommended: `Python 3.10+` (the scripts use modern typing and stdlib features).
- Use your project venv (on Windows typically `.\.venv\Scripts\python.exe`).

Install dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r LLM\requirements.txt
```

Pick a `torch` build that matches your CUDA/driver (CPU inference works but is slow).

## GPU / 4-bit (bitsandbytes)

`LLM/train/lora_sft_trainer.py` loads the model in 4-bit by default. If CUDA is not available or `bitsandbytes` is not usable, disable it:

```powershell
.\.venv\Scripts\python.exe LLM\train\lora_sft_trainer.py --no-4bit
```

The inference CLI `LLM/cli/chat_cli.py` also supports `--no-4bit`.

Notes:

- On native Windows, `bitsandbytes` can be fragile (CUDA/PyTorch/driver/toolchain mismatch).
- If you just want a working baseline, use `--no-4bit` or train under WSL2/Linux.

## ModelScope storage location

`LLM/download.py` downloads weights into `LLM/models/` by default. You can override it via:

- `MODELSCOPE_LOCAL_DIR`: custom weights root directory
- `MODELSCOPE_MODEL_REVISION`: optional model revision/tag

Example:

```powershell
$env:MODELSCOPE_LOCAL_DIR="D:\\weights\\qwen"
.\.venv\Scripts\python.exe LLM\download.py
```
