from __future__ import annotations

from pathlib import Path
from typing import Any

from dataset_io.registry import DATASET_REGISTRY, load_registry


def is_lerobot_dataset_downloaded(root: str | Path, require_videos: bool = True) -> bool:
    root = Path(root)
    if not root.exists():
        return False

    required = ["data", "meta"]
    if require_videos:
        required.append("videos")
    if any(not (root / name).exists() for name in required):
        return False

    has_parquet = any((root / "data").glob("chunk-*/*.parquet"))
    has_info = (root / "meta" / "info.json").exists()
    has_video = not require_videos or any((root / "videos").glob("**/*.mp4"))
    return has_parquet and has_info and has_video


def ensure_lerobot_dataset_local(
    name: str,
    registry_path: str | Path = DATASET_REGISTRY,
    *,
    force_download: bool = False,
    require_videos: bool = True,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    if name not in registry:
        available = ", ".join(registry)
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")

    config = registry[name]
    root = Path(config["root"])
    downloaded = is_lerobot_dataset_downloaded(root, require_videos=require_videos)
    if force_download or not downloaded:
        from huggingface_hub import snapshot_download

        root.mkdir(parents=True, exist_ok=True)
        print(f"[download] {name}")
        print(f"  repo_id: {config['repo_id']}")
        print(f"  local_dir: {root}")
        snapshot_download(
            repo_id=config["repo_id"],
            repo_type="dataset",
            local_dir=str(root),
        )
    else:
        print(f"[local] {name}: {root}")
    return config


def load_lerobot_dataset(
    name: str,
    registry_path: str | Path = DATASET_REGISTRY,
    *,
    force_download: bool = False,
    require_videos: bool = True,
) -> tuple[Any, dict[str, Any]]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    config = ensure_lerobot_dataset_local(
        name,
        registry_path,
        force_download=force_download,
        require_videos=require_videos,
    )
    dataset = LeRobotDataset(repo_id=config["repo_id"], root=config["root"])
    return dataset, config