from __future__ import annotations

from typing import Any

from robot_events.adapters.agibot2 import Agibot2Adapter
from robot_events.adapters.ur5 import UR5Adapter
from robot_events.keyframes import build_episode_index


def build_robot_adapter(cfg: dict[str, Any]):
    robot_type = cfg.get("robot_type", "ur5")
    if robot_type == "ur5":
        return UR5Adapter(cfg)
    if robot_type in {"aibot2", "agibot2"}:
        return Agibot2Adapter(cfg)
    raise KeyError(f"Unknown robot_type '{robot_type}'")


def load_episode_items(ds: Any, episode_index: int) -> list[dict[str, Any]]:
    episode_slice = next(
        (episode_slice for episode_slice in build_episode_index(ds) if episode_slice.episode_index == episode_index),
        None,
    )
    if episode_slice is None:
        raise ValueError(f"episode {episode_index} not found")

    hf_dataset = getattr(ds, "hf_dataset", None)
    if hf_dataset is None:
        return [ds[i] for i in episode_slice.global_indices]

    batch = hf_dataset.select(episode_slice.global_indices)
    return [
        {column: batch[column][i] for column in batch.column_names}
        for i in range(len(episode_slice.global_indices))
    ]


def resolve_episode_cameras(
    ds: Any,
    cfg: dict[str, Any],
    episode_index: int,
    cameras: list[str] | None,
    *,
    keyframe_type: str | None = None,
) -> tuple[list[str], dict[str, Any] | None]:
    if cameras is not None:
        return cameras, None

    cameras_cfg = cfg.get("cameras")
    if isinstance(cameras_cfg, list):
        return list(cameras_cfg), None

    if isinstance(cameras_cfg, dict):
        adapter = build_robot_adapter(cfg)
        items = load_episode_items(ds, episode_index)
        motion = adapter.detect_moving_arm(items)
        selected_cameras = adapter.select_cameras(motion["moving_arm"], keyframe_type=keyframe_type)
        return selected_cameras, motion

    default_camera = cfg.get("default_camera")
    if default_camera is not None:
        return [default_camera], None

    raise ValueError("No camera specified. Use --camera or set cameras/default_camera in datasets.yaml")