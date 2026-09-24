from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_io import DATASET_REGISTRY
from project_paths import QUALITY_OUTPUT_DIR
from quality.calibration import (
    QualityCalibration,
    apply_quality_calibration,
    calibrate_quality_thresholds,
)
from quality.dashboard import write_quality_dashboard
from quality.pipeline import analyze_dataset_quality
from quality.review import quality_report_to_markdown


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
    parser.add_argument(
        "--write-calibration",
        default=None,
        help="Write P5/P95 thresholds learned from --reference-episodes.",
    )
    parser.add_argument(
        "--reference-episodes",
        default=None,
        help="JSON list or newline-delimited IDs for known-good calibration episodes.",
    )
    parser.add_argument(
        "--calibration",
        default=None,
        help="Apply dataset-relative flags from an existing calibration JSON.",
    )
    parser.add_argument("--lower-quantile", type=float, default=0.05)
    parser.add_argument("--upper-quantile", type=float, default=0.95)
    parser.add_argument("--min-reference-count", type=int, default=20)
    return parser.parse_args()


def load_reference_episode_ids(path: str | Path) -> set[str]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        values = json.loads(text)
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            raise ValueError("reference episode JSON must be a list of episode ID strings")
    else:
        values = [line.strip() for line in text.splitlines() if line.strip()]
    if not values:
        raise ValueError("reference episode list is empty")
    return set(values)


def main() -> int:
    args = parse_args()
    if args.write_calibration is not None and args.reference_episodes is None:
        raise ValueError("--write-calibration requires --reference-episodes")
    if args.write_calibration is not None and args.calibration is not None:
        raise ValueError("generate and apply calibration in separate runs")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else (
        QUALITY_OUTPUT_DIR / f"{args.dataset}_quality_{timestamp}.json"
    )
    report = analyze_dataset_quality(
        args.dataset,
        registry_path=args.registry,
        sample=args.sample,
    )
    if args.write_calibration is not None:
        calibration = calibrate_quality_thresholds(
            report,
            args.dataset,
            load_reference_episode_ids(args.reference_episodes),
            lower_quantile=args.lower_quantile,
            upper_quantile=args.upper_quantile,
            min_reference_count=args.min_reference_count,
        )
        calibration.to_json(args.write_calibration)
        print(f"Calibration: {args.write_calibration}")
    if args.calibration is not None:
        calibration = QualityCalibration.from_json(args.calibration)
        if calibration.dataset_name != args.dataset:
            raise ValueError(
                f"calibration is for {calibration.dataset_name!r}, not {args.dataset!r}"
            )
        apply_quality_calibration(report, calibration)
        print(f"Calibration applied: {args.calibration}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_json(output_path)
    summary_path = output_path.with_suffix(".md")
    summary_path.write_text(
        quality_report_to_markdown(report, args.dataset, sample=args.sample),
        encoding="utf-8",
    )
    dashboard_path = output_path.with_suffix(".html")
    write_quality_dashboard(
        report,
        args.dataset,
        dashboard_path,
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
    print(f"Summary: {summary_path}")
    print(f"Dashboard: {dashboard_path}")

    return int(args.min_score is not None and report.overall_score < args.min_score)


if __name__ == "__main__":
    raise SystemExit(main())