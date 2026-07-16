# robot_events/adapters/agibot2.py
from __future__ import annotations

from typing import Any
import numpy as np
import torch

from .base import BaseRobotAdapter


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def motion_score(
    seq: np.ndarray,
    pos_slice=(0, 3),
    quat_slice=(3, 7),
    gripper_index=7,
    pos_weight=1.0,
    quat_weight=0.2,
    grip_weight=0.5,
) -> float:
    if len(seq) < 2:
        return 0.0

    pos = seq[:, pos_slice[0]:pos_slice[1]]
    quat = seq[:, quat_slice[0]:quat_slice[1]]
    grip = seq[:, gripper_index:gripper_index + 1]

    dpos = np.linalg.norm(np.diff(pos, axis=0), axis=1).mean()
    dquat = np.linalg.norm(np.diff(quat, axis=0), axis=1).mean()
    dgrip = np.abs(np.diff(grip, axis=0)).mean()

    return float(pos_weight * dpos + quat_weight * dquat + grip_weight * dgrip)


class Agibot2Adapter(BaseRobotAdapter):
    def _get_action_seq(self, items: list[dict[str, Any]], action_key="action") -> np.ndarray:
        actions = [to_numpy(item[action_key]).astype(np.float32) for item in items]
        return np.stack(actions, axis=0)

    def _split_action(self, action_seq: np.ndarray):
        layout = self.cfg.get("action_layout", {})
        left_slice = layout.get("left_slice", [0, 8])
        right_slice = layout.get("right_slice", [8, 16])

        left = action_seq[:, left_slice[0]:left_slice[1]]
        right = action_seq[:, right_slice[0]:right_slice[1]]
        return left, right

    def detect_moving_arm(
        self,
        items: list[dict[str, Any]],
        ratio_threshold: float = 1.5,
        min_motion: float = 1e-4,
    ) -> dict:
        action_seq = self._get_action_seq(items)
        left, right = self._split_action(action_seq)

        layout = self.cfg.get("action_layout", {})
        pos_slice = tuple(layout.get("pos_slice", [0, 3]))
        quat_slice = tuple(layout.get("quat_slice", [3, 7]))
        gripper_index = int(layout.get("gripper_index", 7))

        left_score = motion_score(left, pos_slice, quat_slice, gripper_index)
        right_score = motion_score(right, pos_slice, quat_slice, gripper_index)

        if left_score < min_motion and right_score < min_motion:
            arm = "none"
        elif left_score > ratio_threshold * right_score:
            arm = "left"
        elif right_score > ratio_threshold * left_score:
            arm = "right"
        else:
            arm = "both"

        return {
            "moving_arm": arm,
            "left_score": left_score,
            "right_score": right_score,
        }

    def get_gripper_signal(self, items: list[dict[str, Any]]) -> np.ndarray:
        """
        For bimanual robot, return the max-moving arm's gripper signal by default.
        For event-specific logic, you can later expose left/right separately.
        """
        action_seq = self._get_action_seq(items)
        left, right = self._split_action(action_seq)

        moving = self.detect_moving_arm(items)["moving_arm"]

        layout = self.cfg.get("action_layout", {})
        gripper_index = int(layout.get("gripper_index", 7))

        if moving == "right":
            return right[:, gripper_index].astype(np.float32)
        if moving == "left":
            return left[:, gripper_index].astype(np.float32)

        # both/none fallback: use stronger gripper change
        left_sig = left[:, gripper_index]
        right_sig = right[:, gripper_index]

        left_change = np.abs(np.diff(left_sig)).mean() if len(left_sig) > 1 else 0
        right_change = np.abs(np.diff(right_sig)).mean() if len(right_sig) > 1 else 0

        return (right_sig if right_change > left_change else left_sig).astype(np.float32)

    def select_cameras(self, moving_arm: str, keyframe_type: str | None = None) -> list[str]:
        cameras_cfg = self.cfg.get("cameras", {})

        # For grasp-related moments, prefer the top/global view to see object contact clearly.
        if keyframe_type in {"pre_grasp", "gripper_close", "post_grasp"}:
            cams = cameras_cfg.get("top") or cameras_cfg.get("global") or cameras_cfg.get("both") or cameras_cfg.get("default")
            return cams if isinstance(cams, list) else [cams]

        # For final-state judgement, include global camera even if one wrist is moving.
        if keyframe_type in {"post_place", "episode_end", "placement_success", "upright_check"}:
            cams = cameras_cfg.get("global") or cameras_cfg.get("both") or cameras_cfg.get("default")
            return cams if isinstance(cams, list) else [cams]

        cams = cameras_cfg.get(moving_arm)
        if cams:
            return cams if isinstance(cams, list) else [cams]

        return super().select_cameras(moving_arm, keyframe_type)