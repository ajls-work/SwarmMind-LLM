import argparse
import os
import sys
import threading
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _default_paths() -> tuple[str, str]:
    llm_dir = Path(__file__).resolve().parents[1]
    base = llm_dir / "models" / "Qwen3-4B-Instruct-2507"

    candidates = [
        llm_dir / "outputs" / "lora_latest",
        llm_dir / "train" / "checkpoints" / "lora_latest",
        llm_dir.parents[0] / "checkpoints" / "lora_latest",
    ]
    for c in candidates:
        if c.exists():
            return str(base), str(c)
    return str(base), str(candidates[0])


def _pick_device(model) -> torch.device:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for v in device_map.values():
            if isinstance(v, str) and v.startswith("cuda"):
                return torch.device(v)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_tokenizer(base_path: str):
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(base_path: str, *, use_4bit: bool, tokenizer):
    use_4bit = bool(use_4bit and torch.cuda.is_available())
    if use_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except Exception:
            use_4bit = False
            print("[WARN] bitsandbytes not available; falling back to non-4bit.", flush=True)

    compute_dtype = torch.bfloat16
    if torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        compute_dtype = torch.float16

    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        if use_4bit
        else None
    )

    model_kwargs = {"device_map": "auto", "trust_remote_code": True}
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    else:
        model_kwargs["torch_dtype"] = compute_dtype if torch.cuda.is_available() else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(base_path, **model_kwargs)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.eval()
    return base_model


def load_lora_adapter(base_model, lora_path: str):
    model = PeftModel.from_pretrained(base_model, lora_path, is_trainable=False)
    model.eval()
    return model


def stream_generate(
    tokenizer,
    model,
    messages,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
):
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt")

    device = _pick_device(model)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    do_sample = temperature > 0
    eos_ids = [tokenizer.eos_token_id]
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id not in eos_ids:
        eos_ids.append(tokenizer.pad_token_id)
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generate_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        eos_token_id=eos_ids,
        pad_token_id=tokenizer.pad_token_id,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
    )

    def _run_generate() -> None:
        with torch.inference_mode():
            model.generate(**generate_kwargs)

    thread = threading.Thread(target=_run_generate, daemon=True)
    thread.start()
    try:
        for text in streamer:
            yield text
    finally:
        thread.join()


def main() -> int:
    default_base, default_lora = _default_paths()

    parser = argparse.ArgumentParser(description="CLI chat for base model + LoRA adapter.")
    parser.add_argument("--base", default=default_base, help="Base model folder path")
    parser.add_argument("--lora", default=default_lora, help="LoRA adapter folder path")
    parser.add_argument(
        "--adapter-mode",
        choices=["always", "never"],
        default="always",
        help="always: use LoRA for all queries; never: base-only",
    )
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--system", default=DEFAULT_SYSTEM_PROMPT, help="System prompt (optional)")
    parser.add_argument("--no-system", action="store_true", help="Do not send a system message")
    parser.add_argument("--query", default=None, help="Single-turn query (non-interactive)")
    args = parser.parse_args()

    base_path = os.path.abspath(args.base)
    lora_path = os.path.abspath(args.lora)
    if not os.path.exists(base_path):
        print(f"[ERROR] Base model path not found: {base_path}", file=sys.stderr)
        return 2
    if args.adapter_mode == "always" and not os.path.exists(lora_path):
        print(f"[ERROR] LoRA path not found: {lora_path}", file=sys.stderr)
        return 2

    tokenizer = load_tokenizer(base_path)
    base_model = load_base_model(base_path, use_4bit=not args.no_4bit, tokenizer=tokenizer)

    model = base_model
    if args.adapter_mode == "always":
        model = load_lora_adapter(base_model, lora_path)

    messages: list[dict] = []
    if not args.no_system and args.system.strip():
        messages.append({"role": "system", "content": args.system})

    def run_turn(user_text: str, *, stream: bool, prefix: str = "Assistant> ") -> str:
        messages.append({"role": "user", "content": user_text})
        out_parts: list[str] = []
        if stream:
            if prefix:
                print(prefix, end="", flush=True)
            for chunk in stream_generate(
                tokenizer,
                model,
                messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            ):
                out_parts.append(chunk)
                print(chunk, end="", flush=True)
            print(flush=True)
        else:
            for chunk in stream_generate(
                tokenizer,
                model,
                messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            ):
                out_parts.append(chunk)

        out = "".join(out_parts).strip()
        messages.append({"role": "assistant", "content": out})
        return out

    if args.query:
        run_turn(args.query, stream=True, prefix="")
        return 0

    print("Type your instruction. Commands: /reset, /exit", flush=True)
    while True:
        try:
            user_text = input("User> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text in {"/exit", "exit", "quit"}:
            break
        if user_text == "/reset":
            messages = []
            if not args.no_system and args.system.strip():
                messages.append({"role": "system", "content": args.system})
            print("[INFO] History cleared.", flush=True)
            continue

        run_turn(user_text, stream=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
