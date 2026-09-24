from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from trajectory.models import CanonicalTrajectory


class BaseRobotAdapter(ABC):
    def __init__(self, config: dict[str, Any]):
        self.config = config

    @abstractmethod
    def build_trajectory(self, items: list[dict[str, Any]]) -> CanonicalTrajectory:
        raise NotImplementedError

    def select_cameras(self, moving_arm: str, keyframe_type: str | None = None) -> list[str]:
        del keyframe_type
        cameras_config = self.config.get("cameras", {})
        if moving_arm in cameras_config:
            cameras = cameras_config[moving_arm]
            return cameras if isinstance(cameras, list) else [cameras]

        default = cameras_config.get("default")
        if default is not None:
            return default if isinstance(default, list) else [default]

        global_cameras = cameras_config.get("global")
        if global_cameras is not None:
            return global_cameras if isinstance(global_cameras, list) else [global_cameras]
        raise KeyError("No default/global camera configured.")