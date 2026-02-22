import argparse
import json
from pathlib import Path


REQUIRED_OUTPUT_KEYS = {"task_type", "agents_config", "rl_parameters"}


def _default_input_path() -> Path:
    llm_dir = Path(__file__).resolve().parent
    return llm_dir / "data" / "marl_llm_dataset_2000.json"


def _default_output_path() -> Path:
    llm_dir = Path(__file__).resolve().parent
    return llm_dir / "data" / "marl_llm_dataset_cleaned.json"


def _parse_output_json(output_str: str) -> dict:
    parsed = json.loads(output_str)
    if not isinstance(parsed, dict):
        raise ValueError("output is not a JSON object")
    return parsed


def validate_and_clean_dataset(input_path: Path, output_path: Path, *, minify_output: bool) -> None:
    with input_path.open("r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not isinstance(dataset, list):
        raise ValueError(f"Dataset file must be a JSON list: {input_path}")

    cleaned: list[dict] = []
    errors = {
        "invalid_item": 0,
        "missing_keys": 0,
        "output_json_error": 0,
        "output_missing_keys": 0,
        "output_invalid_types": 0,
    }

    for item in dataset:
        if not isinstance(item, dict):
            errors["invalid_item"] += 1
            continue

        if not all(k in item for k in ("instruction", "input", "output")):
            errors["missing_keys"] += 1
            continue

        output = item.get("output", "")
        if not isinstance(output, str) or not output.strip():
            errors["output_json_error"] += 1
            continue

        try:
            parsed_output = _parse_output_json(output)
        except Exception:
            errors["output_json_error"] += 1
            continue

        if not REQUIRED_OUTPUT_KEYS.issubset(parsed_output.keys()):
            errors["output_missing_keys"] += 1
            continue

        if (
            not isinstance(parsed_output.get("task_type"), str)
            or not isinstance(parsed_output.get("agents_config"), list)
            or not isinstance(parsed_output.get("rl_parameters"), dict)
        ):
            errors["output_invalid_types"] += 1
            continue

        if minify_output:
            item = dict(item)
            item["output"] = json.dumps(parsed_output, ensure_ascii=False, separators=(",", ":"))

        cleaned.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    total = len(dataset)
    valid = len(cleaned)
    ratio = (valid / total * 100.0) if total else 0.0

    print("=" * 40)
    print("Dataset Validation Report")
    print("=" * 40)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Total:  {total}")
    print(f"Valid:  {valid} ({ratio:.2f}%)")
    print("-" * 40)
    for k, v in errors.items():
        print(f"{k}: {v}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and clean the generated dataset.")
    parser.add_argument("--input", default=str(_default_input_path()), help="Input dataset JSON path")
    parser.add_argument("--output", default=str(_default_output_path()), help="Output cleaned JSON path")
    parser.add_argument(
        "--minify-output",
        action="store_true",
        help="Rewrite item['output'] as minified JSON (stable formatting, fewer tokens).",
    )
    args = parser.parse_args()

    validate_and_clean_dataset(
        Path(args.input).resolve(),
        Path(args.output).resolve(),
        minify_output=bool(args.minify_output),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

