from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseRobotAdapter(ABC):
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg

    @abstractmethod
    def get_gripper_signal(self, items: list[dict[str, Any]]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def build_quality_trajectory(self, items: list[dict[str, Any]]):
        raise NotImplementedError

    def detect_moving_arm(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "moving_arm": "single",
            "left_score": None,
            "right_score": None,
        }

    def select_cameras(self, moving_arm: str, keyframe_type: str | None = None) -> list[str]:
        del keyframe_type

        cameras_cfg = self.cfg.get("cameras", {})

        if moving_arm in cameras_cfg:
            cameras = cameras_cfg[moving_arm]
            return cameras if isinstance(cameras, list) else [cameras]

        default = cameras_cfg.get("default")
        if default is not None:
            return default if isinstance(default, list) else [default]

        global_cameras = cameras_cfg.get("global")
        if global_cameras is not None:
            return global_cameras if isinstance(global_cameras, list) else [global_cameras]

        raise KeyError("No default/global camera configured.")