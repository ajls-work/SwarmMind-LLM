import argparse
import os
from pathlib import Path

from modelscope import snapshot_download


DEFAULT_MODEL_IDS = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-4B-Base",
    "Qwen/Qwen3-4B-Thinking-2507",
]


def _default_root_dir() -> Path:
    llm_dir = Path(__file__).resolve().parent
    return llm_dir / "models"


def _split_models(models: list[str]) -> list[str]:
    out: list[str] = []
    for part in models:
        for token in part.split(","):
            token = token.strip()
            if token:
                out.append(token)
    return out


def download_models(model_ids: list[str], root_dir: Path, *, revision: str | None = None) -> None:
    root_dir.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[str, str]] = []

    for model_id in model_ids:
        model_name = model_id.split("/")[-1]
        local_dir = root_dir / model_name
        local_dir.mkdir(parents=True, exist_ok=True)

        print(f"==> Downloading: {model_id}", flush=True)
        print(f"    Local dir:   {local_dir}", flush=True)
        try:
            kwargs = {"local_dir": str(local_dir)}
            if revision:
                kwargs["revision"] = revision
            snapshot_download(model_id, **kwargs)
        except Exception as exc:
            failures.append((model_id, str(exc)))
            print(f"    FAILED: {exc}", flush=True)
        else:
            print("    Done.", flush=True)

    if failures:
        print("\nSome downloads failed:", flush=True)
        for model_id, error in failures:
            print(f"- {model_id}: {error}", flush=True)
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Qwen models from ModelScope.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Model IDs (space-separated or comma-separated). Default: Qwen3 4B variants.",
    )
    parser.add_argument(
        "--root-dir",
        default=os.environ.get("MODELSCOPE_LOCAL_DIR", str(_default_root_dir())),
        help="Local folder to store weights (default: LLM/models).",
    )
    parser.add_argument(
        "--revision",
        default=os.environ.get("MODELSCOPE_MODEL_REVISION", None),
        help="Optional model revision/tag.",
    )
    args = parser.parse_args()

    model_ids = _split_models(args.models) if args.models else DEFAULT_MODEL_IDS
    root_dir = Path(args.root_dir).resolve()
    download_models(model_ids, root_dir, revision=args.revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

