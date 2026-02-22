import argparse
import json
from pathlib import Path


def _default_dataset_path() -> Path:
    llm_dir = Path(__file__).resolve().parent
    return llm_dir / "data" / "marl_llm_dataset_2000.json"


def append_data(batch_items: list[dict], dataset_path: Path) -> None:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_path.exists():
        try:
            with dataset_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    else:
        data = []

    if not isinstance(data, list):
        raise ValueError(f"Dataset file is not a JSON list: {dataset_path}")

    data.extend(batch_items)
    with dataset_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Appended {len(batch_items)} items. Total: {len(data)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Append a JSON batch into the dataset file.")
    parser.add_argument("batch_file", help="Path to a JSON file (list of items) to append")
    parser.add_argument("--dataset", default=str(_default_dataset_path()), help="Dataset JSON path")
    args = parser.parse_args()

    batch_path = Path(args.batch_file).resolve()
    dataset_path = Path(args.dataset).resolve()

    with batch_path.open("r", encoding="utf-8") as f:
        batch_items = json.load(f)
    if not isinstance(batch_items, list):
        raise ValueError(f"Batch file must be a JSON list: {batch_path}")

    append_data(batch_items, dataset_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

