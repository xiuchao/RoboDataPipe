from __future__ import annotations

from typing import Any

from trajectory.adapters.base import BaseRobotAdapter
from trajectory.canonicalize import canonicalize_trajectory
from trajectory.contracts import load_trajectory_contract
from trajectory.models import CanonicalTrajectory


class Agibot2Adapter(BaseRobotAdapter):
    def build_trajectory(self, items: list[dict[str, Any]]) -> CanonicalTrajectory:
        return canonicalize_trajectory(items, load_trajectory_contract(self.config))

    def select_cameras(self, moving_arm: str, keyframe_type: str | None = None) -> list[str]:
        cameras_config = self.config.get("cameras", {})
        if keyframe_type in {
            "pre_grasp", "gripper_close", "post_grasp", "gripper_fully_open",
            "release_keyframe", "pre_retreat", "retreat_start", "post_retreat",
        }:
            cameras = (
                cameras_config.get("top")
                or cameras_config.get("global")
                or cameras_config.get("both")
                or cameras_config.get("default")
            )
            return cameras if isinstance(cameras, list) else [cameras]
        if keyframe_type in {"post_place", "episode_end", "placement_success", "upright_check"}:
            cameras = (
                cameras_config.get("global")
                or cameras_config.get("both")
                or cameras_config.get("default")
            )
            return cameras if isinstance(cameras, list) else [cameras]
        cameras = cameras_config.get(moving_arm)
        if cameras:
            return cameras if isinstance(cameras, list) else [cameras]
        return super().select_cameras(moving_arm, keyframe_type)