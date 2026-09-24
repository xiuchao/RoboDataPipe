from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vlm.task_dashboard import records_from_summary, write_task_outcome_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a self-contained task-outcome dashboard from a summary file."
    )
    parser.add_argument("--input", required=True, help="Task evaluation summary text path.")
    parser.add_argument("--dataset", required=True, help="Dataset name shown in the dashboard.")
    parser.add_argument("--output", default=None, help="Output HTML path. Defaults beside input.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.input)
    records = records_from_summary(source.read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"No episode outcomes found in {source}")
    output = Path(args.output) if args.output else source.with_suffix(".html")
    write_task_outcome_dashboard(
        records,
        args.dataset,
        output,
        subtitle=f"Historical task evaluation | {source.name}",
    )
    print(f"Dashboard: {output}")


if __name__ == "__main__":
    main()