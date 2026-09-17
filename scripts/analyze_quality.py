from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import DATASET_REGISTRY
from project_paths import QUALITY_OUTPUT_DIR
from quality_analysis import analyze_dataset_quality


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze LeRobot demonstration quality with DEM-owned adapters and metrics."
    )
    parser.add_argument("--dataset", required=True, help="Dataset nickname in datasets.yaml.")
    parser.add_argument("--registry", default=DATASET_REGISTRY, help="Path to datasets.yaml.")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Analyze only the first N episodes. Default: all episodes.",
    )
    parser.add_argument("--output", default=None, help="Output JSON path.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Exit with status 1 when the overall score is below this value.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else (
        QUALITY_OUTPUT_DIR / f"{args.dataset}_quality_{timestamp}.json"
    )
    report = analyze_dataset_quality(
        args.dataset,
        registry_path=args.registry,
        output_path=output_path,
        sample=args.sample,
    )

    print(f"Dataset: {args.dataset}")
    print(f"Episodes: {report.num_episodes}")
    print(f"Overall quality: {report.overall_score:.2f}/10")
    if report.flagged_episodes:
        print("Flagged episodes:")
        for flag, episode_ids in sorted(report.flagged_episodes.items()):
            print(f"  {flag}: {len(episode_ids)}")
    else:
        print("Flagged episodes: none")
    print(f"Report: {output_path}")

    return int(args.min_score is not None and report.overall_score < args.min_score)


if __name__ == "__main__":
    raise SystemExit(main())