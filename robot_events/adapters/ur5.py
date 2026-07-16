# robot_events/adapters/ur5.py
from __future__ import annotations

from typing import Any
import numpy as np
import torch

from .base import BaseRobotAdapter


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class UR5Adapter(BaseRobotAdapter):
    def get_gripper_signal(self, items: list[dict[str, Any]]) -> np.ndarray:
        grip_cfg = self.cfg.get("gripper", {})
        source = grip_cfg.get("signal_source", "observation.state")
        dim = int(grip_cfg.get("dim", -1))

        vals = []
        for item in items:
            x = to_numpy(item[source])
            vals.append(float(x[dim]))

        return np.asarray(vals, dtype=np.float32)

    def detect_moving_arm(self, items: list[dict[str, Any]]) -> dict:
        # UR5 is single-arm.
        return {
            "moving_arm": "single",
            "left_score": None,
            "right_score": None,
        }

    def select_cameras(self, moving_arm: str, keyframe_type: str | None = None) -> list[str]:
        cameras_cfg = self.cfg.get("cameras", {})

        # For final placement / upright judgment, prefer global camera.
        if keyframe_type in {"post_place", "episode_end", "placement_success"}:
            cams = cameras_cfg.get("global")
            if cams:
                return cams if isinstance(cams, list) else [cams]

        default = cameras_cfg.get("default")
        if default:
            return default if isinstance(default, list) else [default]

        return super().select_cameras(moving_arm, keyframe_type)