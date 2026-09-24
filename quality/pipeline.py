from __future__ import annotations

from pathlib import Path
from typing import Any

from dataset_io import (
    DATASET_REGISTRY,
    build_episode_index,
    load_episode_items,
    load_lerobot_dataset,
    load_registry,
)
from quality.contextual_rules import behavior_expectations_from_config
from quality.evaluator import DatasetQualityAnalyzer
from quality.results import QualityReport
from trajectory.adapters import build_robot_adapter


def analyze_loaded_dataset_quality(
    dataset: Any,
    dataset_config: dict[str, Any],
    *,
    sample: int = 0,
    dataset_path: str | Path | None = None,
) -> QualityReport:
    adapter = build_robot_adapter(dataset_config)
    analyzer = DatasetQualityAnalyzer(
        behavior_expectations_from_config(dataset_config)
    )
    episodes = []
    for index, episode_slice in enumerate(build_episode_index(dataset)):
        if sample > 0 and index >= sample:
            break
        items = load_episode_items(dataset, episode_slice.episode_index)
        trajectory = adapter.build_trajectory(items)
        episodes.append(
            analyzer.analyze_episode(
                f"episode_{episode_slice.episode_index:03d}",
                trajectory,
            )
        )
    return analyzer.build_report(str(dataset_path or "<loaded_dataset>"), episodes)


def analyze_dataset_quality(
    dataset_name: str,
    *,
    registry_path: str | Path = DATASET_REGISTRY,
    output_path: str | Path | None = None,
    sample: int = 0,
) -> QualityReport:
    registry = load_registry(registry_path)
    if dataset_name not in registry:
        available = ", ".join(registry)
        raise KeyError(f"Unknown dataset '{dataset_name}'. Available: {available}")

    dataset_config = registry[dataset_name]
    dataset_root = Path(dataset_config["root"]).expanduser()
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset '{dataset_name}' is not available at {dataset_root}."
        )

    dataset, _ = load_lerobot_dataset(
        dataset_name,
        registry_path=registry_path,
        require_videos=False,
    )
    report = analyze_loaded_dataset_quality(
        dataset,
        dataset_config,
        sample=sample,
        dataset_path=dataset_root,
    )
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        report.to_json(destination)
    return report