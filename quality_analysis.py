from __future__ import annotations

from pathlib import Path

from dataloader import DATASET_REGISTRY, load_lerobot_dataset, load_registry
from quality.analyzer import DatasetQualityAnalyzer
from robot_events.keyframes import build_episode_index
from robot_events.registry import build_robot_adapter, load_episode_items


def analyze_loaded_dataset_quality(
    ds,
    dataset_config: dict,
    *,
    sample: int = 0,
    dataset_path: str | Path | None = None,
):
    """Analyze an already loaded dataset using DEM-owned adapters and metrics."""
    adapter = build_robot_adapter(dataset_config)
    analyzer = DatasetQualityAnalyzer()
    episodes = []
    for index, episode_slice in enumerate(build_episode_index(ds)):
        if sample > 0 and index >= sample:
            break
        items = load_episode_items(ds, episode_slice.episode_index)
        trajectory = adapter.build_quality_trajectory(items)
        episodes.append(
            analyzer.analyze_episode(
                f"episode_{episode_slice.episode_index:06d}",
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
):
    """Analyze a registered local dataset using DEM-owned robot adapters and metrics."""
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

    ds, _cfg = load_lerobot_dataset(dataset_name, registry_path=registry_path, require_videos=False)
    report = analyze_loaded_dataset_quality(
        ds,
        dataset_config,
        sample=sample,
        dataset_path=dataset_root,
    )

    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        report.to_json(destination)

    return report