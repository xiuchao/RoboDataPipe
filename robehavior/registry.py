from __future__ import annotations

from typing import Any

from dataset_io.episodes import load_episode_items
from robehavior.activity import analyze_trajectory_activity, infer_moving_arm
from trajectory.adapters import build_robot_adapter


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
        trajectory = adapter.build_trajectory(items)
        motion = infer_moving_arm(analyze_trajectory_activity(trajectory))
        selected_cameras = adapter.select_cameras(motion["moving_arm"], keyframe_type=keyframe_type)
        return selected_cameras, motion

    default_camera = cfg.get("default_camera")
    if default_camera is not None:
        return [default_camera], None

    raise ValueError("No camera specified. Use --camera or set cameras/default_camera in datasets.yaml")