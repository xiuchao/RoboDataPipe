from __future__ import annotations

from typing import Any

from trajectory.adapters.base import BaseRobotAdapter
from trajectory.canonicalize import canonicalize_trajectory
from trajectory.contracts import load_trajectory_contract
from trajectory.models import CanonicalTrajectory


class UR5Adapter(BaseRobotAdapter):
    def build_trajectory(self, items: list[dict[str, Any]]) -> CanonicalTrajectory:
        return canonicalize_trajectory(items, load_trajectory_contract(self.config))

    def select_cameras(self, moving_arm: str, keyframe_type: str | None = None) -> list[str]:
        cameras_config = self.config.get("cameras", {})
        if keyframe_type in {"post_place", "episode_end", "placement_success"}:
            cameras = cameras_config.get("global")
            if cameras:
                return cameras if isinstance(cameras, list) else [cameras]
        default = cameras_config.get("default")
        if default:
            return default if isinstance(default, list) else [default]
        return super().select_cameras(moving_arm, keyframe_type)