import argparse
import os
import time
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training


DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


class JsonSFTDataset(Dataset):
    def __init__(
        self,
        json_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 1024,
        *,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        instruction_as_system: bool = False,
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        samples: list[dict[str, list[int]]] = []
        dropped = 0

        for item in raw:
            if not isinstance(item, dict):
                dropped += 1
                continue

            instruction = item.get("instruction")
            user_input = item.get("input")
            output = item.get("output")
            if output is None:
                dropped += 1
                continue

            instruction = str(instruction or "").strip()
            user_input = str(user_input or "").strip()
            output = str(output).strip()

            if not output:
                dropped += 1
                continue

            messages: list[dict[str, str]] = []
            if instruction_as_system:
                if not instruction or not user_input:
                    dropped += 1
                    continue
                messages.append({"role": "system", "content": instruction})
                messages.append({"role": "user", "content": user_input})
            else:
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})

                # Most of this project's items are: instruction=task spec, input=free-form intent.
                # Keep the format predictable by merging them into a single user message.
                merged = user_input
                if instruction and (not system_prompt or instruction != system_prompt.strip()):
                    merged = instruction if not merged else f"{instruction}\n\n{merged}"

                if not merged:
                    dropped += 1
                    continue
                messages.append({"role": "user", "content": merged})
            prompt_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            answer_text = output + (tokenizer.eos_token or "")

            prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
            answer_ids = tokenizer(answer_text, add_special_tokens=False).input_ids

            input_ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + answer_ids

            if len(input_ids) > max_length:
                input_ids = input_ids[-max_length:]
                labels = labels[-max_length:]

            if not any(x != -100 for x in labels):
                dropped += 1
                continue

            samples.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": [1] * len(input_ids),
                    "labels": labels,
                }
            )

        self.samples = samples
        if dropped:
            print(f"[INFO] Dropped {dropped} invalid items when building dataset.", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.samples[idx]


@dataclass
class CausalLMCollator:
    tokenizer: AutoTokenizer

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []

        for x in features:
            n = len(x["input_ids"])
            pad = max_len - n
            input_ids.append(x["input_ids"] + [pad_id] * pad)
            attention_mask.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"] + [-100] * pad)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class SimpleTagTrainer:
    def __init__(self, model_name_or_path: str, is_4bit: bool = True):
        self.model_path = model_name_or_path

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        use_4bit = bool(is_4bit)
        if use_4bit and not torch.cuda.is_available():
            use_4bit = False
            print("[WARN] CUDA not available; disabling 4-bit quantization.", flush=True)
        if use_4bit:
            try:
                import bitsandbytes  # noqa: F401
            except Exception:
                use_4bit = False
                print("[WARN] bitsandbytes not installed; disabling 4-bit quantization.", flush=True)

        self.use_4bit = use_4bit

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
            model_kwargs["torch_dtype"] = compute_dtype

        self.base_model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        self.base_model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = None

        self.metrics = {
            "loss": [],
            "eval_loss": [],
        }

    def setup_lora(self, r: int = 8, alpha: int = 32, target_modules: list[str] | None = None):
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        if self.use_4bit:
            self.base_model = prepare_model_for_kbit_training(self.base_model)

        self.base_model.gradient_checkpointing_enable()
        self.base_model.config.use_cache = False

        config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            target_modules=target_modules,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(self.base_model, config)
        self.model.print_trainable_parameters()

    def train_loop(
        self,
        dataset_path: str,
        *,
        max_length: int = 1024,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        instruction_as_system: bool = False,
        num_train_epochs: int = 1,
        per_device_train_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 2e-4,
        eval_ratio: float = 0.02,
        logging_steps: int = 10,
        eval_steps: int = 100,
        max_steps: int | None = None,
        out_dir: str | None = None,
        seed: int = 42,
    ):
        if self.model is None:
            raise ValueError("Call setup_lora() before train_loop().")

        if out_dir is None:
            llm_dir = Path(__file__).resolve().parents[1]
            out_dir = str(llm_dir / "outputs" / "sft_run")

        dataset = JsonSFTDataset(
            dataset_path,
            self.tokenizer,
            max_length=max_length,
            system_prompt=system_prompt,
            instruction_as_system=instruction_as_system,
        )
        if len(dataset) == 0:
            raise ValueError(f"Empty dataset: {dataset_path}")

        indices = list(range(len(dataset)))
        rng = random.Random(seed)
        rng.shuffle(indices)

        eval_size = int(len(indices) * eval_ratio)
        eval_samples = [dataset[i] for i in indices[:eval_size]] if eval_size else None
        train_samples = [dataset[i] for i in indices[eval_size:]]

        use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())

        args = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=learning_rate,
            bf16=use_bf16,
            fp16=not use_bf16,
            seed=seed,
            logging_strategy="steps",
            logging_steps=logging_steps,
            eval_strategy="steps" if eval_samples else "no",
            eval_steps=eval_steps,
            save_strategy="no",
            max_steps=max_steps if max_steps is not None else -1,
            report_to=[],
            remove_unused_columns=False,
            dataloader_num_workers=0,
        )

        print(f"[INFO] Train examples: {len(train_samples)}", flush=True)
        print(f"[INFO] Eval examples:  {len(eval_samples) if eval_samples else 0}", flush=True)

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=train_samples,
            eval_dataset=eval_samples,
            data_collator=CausalLMCollator(self.tokenizer),
        )
        trainer.train()

        self.metrics["loss"].clear()
        self.metrics["eval_loss"].clear()
        for row in trainer.state.log_history:
            if "loss" in row:
                self.metrics["loss"].append(row["loss"])
            if "eval_loss" in row:
                self.metrics["eval_loss"].append(row["eval_loss"])

    def save_parameters(self, ckpt_dir: str | None = None):
        if not self.model:
            raise ValueError("Model not initialized.")

        if ckpt_dir is None:
            llm_dir = Path(__file__).resolve().parents[1]
            ckpt_dir = str(llm_dir / "outputs" / "lora_latest")
        os.makedirs(ckpt_dir, exist_ok=True)
        self.model.save_pretrained(ckpt_dir)
        self.tokenizer.save_pretrained(ckpt_dir)
        print(f"Weights saved to {ckpt_dir}", flush=True)

    def load_parameters(self, ckpt_dir: str | None = None):
        if ckpt_dir is None:
            llm_dir = Path(__file__).resolve().parents[1]
            ckpt_dir = str(llm_dir / "outputs" / "lora_latest")
        if not os.path.exists(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint {ckpt_dir} not found.")
        self.model = PeftModel.from_pretrained(self.base_model, ckpt_dir)
        print(f"Restored weights from {ckpt_dir}", flush=True)

    def log_and_plot(self, out_dir: str | None = None):
        if out_dir is None:
            llm_dir = Path(__file__).resolve().parents[1]
            out_dir = str(llm_dir / "outputs" / "logs")
        os.makedirs(out_dir, exist_ok=True)
        npz_path = os.path.join(out_dir, "training_data.npz")
        fig_path = os.path.join(out_dir, "metrics_plot.png")

        np.savez(npz_path, loss=self.metrics["loss"], eval_loss=self.metrics["eval_loss"])

        def smooth_curve(points: list[float], factor: float = 0.85) -> list[float]:
            smoothed: list[float] = []
            for i, p in enumerate(points):
                if i == 0:
                    smoothed.append(p)
                else:
                    smoothed.append(smoothed[-1] * factor + p * (1 - factor))
            return smoothed

        plt.style.use("bmh")
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        if self.metrics["loss"]:
            ax.plot(self.metrics["loss"], alpha=0.3, label="Train (raw)")
            ax.plot(smooth_curve(self.metrics["loss"]), lw=2, label="Train (smoothed)")
        if self.metrics["eval_loss"]:
            ax.plot(self.metrics["eval_loss"], lw=2, label="Eval")
        ax.set_title("Loss")
        ax.legend()

        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Logging complete. Saved NPZ and Plot to {out_dir}", flush=True)


def _default_paths() -> tuple[Path, Path, Path]:
    llm_dir = Path(__file__).resolve().parents[1]
    model_path = llm_dir / "models" / "Qwen3-4B-Instruct-2507"

    dataset_candidates = [
        llm_dir / "data" / "marl_llm_dataset_mix.json",
        llm_dir / "data" / "marl_llm_dataset_cleaned.json",
        llm_dir / "data" / "marl_llm_dataset_2000.json",
    ]
    dataset_path = next((p for p in dataset_candidates if p.exists()), dataset_candidates[-1])
    outputs_dir = llm_dir / "outputs"
    return model_path, dataset_path, outputs_dir


def main() -> int:
    model_path, dataset_path, outputs_dir = _default_paths()

    parser = argparse.ArgumentParser(description="LoRA SFT trainer for Qwen3-4B.")
    parser.add_argument("--model", default=str(model_path), help="Base model folder path")
    parser.add_argument("--dataset", default=str(dataset_path), help="Dataset JSON path")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eval-ratio", type=float, default=0.02)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means no limit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit loading")
    parser.add_argument("--no-system-prompt", action="store_true", help="Do not add a system message during training")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--instruction-as-system",
        action="store_true",
        help="Legacy format: treat item.instruction as the system message (and item.input as the user message).",
    )
    parser.add_argument("--trainer-out", default=str(outputs_dir / "sft_run"))
    parser.add_argument("--lora-out", default=str(outputs_dir / "lora_latest"))
    parser.add_argument("--log-dir", default=str(outputs_dir / "logs"))
    args = parser.parse_args()

    trainer = SimpleTagTrainer(args.model, is_4bit=not args.no_4bit)
    trainer.setup_lora()

    start_time = time.time()
    trainer.train_loop(
        dataset_path=args.dataset,
        max_length=args.max_length,
        system_prompt=None if args.no_system_prompt else args.system_prompt,
        instruction_as_system=args.instruction_as_system,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        eval_ratio=args.eval_ratio,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        max_steps=args.max_steps or None,
        out_dir=args.trainer_out,
        seed=args.seed,
    )
    print(f"Training finished in {time.time() - start_time:.2f}s", flush=True)

    trainer.log_and_plot(out_dir=args.log_dir)
    trainer.save_parameters(ckpt_dir=args.lora_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
