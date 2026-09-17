# robot_events/adapters/agibot2.py
from __future__ import annotations

from typing import Any
import numpy as np
import torch

from quality.models import CanonicalArmTrajectory, CanonicalTrajectory

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

    def _action_layout(self) -> dict[str, tuple[int, int] | int]:
        layout = self.cfg.get("action_layout", {})
        left_slice = tuple(layout.get("left_slice", [0, 8]))
        right_slice = tuple(layout.get("right_slice", [8, 16]))
        pos_slice = tuple(layout.get("pos_slice", [0, 3]))
        quat_slice = tuple(layout.get("quat_slice", [3, 7]))
        gripper_index = int(layout.get("gripper_index", right_slice[1] - right_slice[0] - 1))
        return {
            "left_slice": left_slice,
            "right_slice": right_slice,
            "pos_slice": pos_slice,
            "quat_slice": quat_slice,
            "gripper_index": gripper_index,
        }

    def _action_gripper_indices(self) -> dict[str, int]:
        layout = self._action_layout()
        gripper_cfg = self.cfg.get("gripper", {}).get("action", {})
        default_index = int(layout["gripper_index"])
        return {
            "left": int(gripper_cfg.get("left_local_index", default_index)),
            "right": int(gripper_cfg.get("right_local_index", default_index)),
        }

    def _split_action(self, action_seq: np.ndarray):
        layout = self._action_layout()
        left_slice = layout["left_slice"]
        right_slice = layout["right_slice"]

        left = action_seq[:, left_slice[0]:left_slice[1]]
        right = action_seq[:, right_slice[0]:right_slice[1]]
        return left, right

    def _split_state(self, state_seq: np.ndarray):
        gripper_cfg = self.cfg.get("gripper", {}).get("state", {})
        left_gripper_index = gripper_cfg.get("left_index")
        right_gripper_index = gripper_cfg.get("right_index")

        if left_gripper_index is not None and right_gripper_index is not None:
            joint_dims = max(0, int(left_gripper_index))
            joint_dims_per_arm = joint_dims // 2
        else:
            joint_dims_per_arm = min(7, state_seq.shape[1] // 2)

        left_joint_end = joint_dims_per_arm
        right_joint_start = joint_dims_per_arm
        right_joint_end = min(joint_dims_per_arm * 2, state_seq.shape[1])

        left_joints = state_seq[:, :left_joint_end] if left_joint_end > 0 else None
        right_joints = (
            state_seq[:, right_joint_start:right_joint_end]
            if right_joint_end > right_joint_start
            else None
        )
        left_gripper = (
            state_seq[:, int(left_gripper_index):int(left_gripper_index) + 1]
            if left_gripper_index is not None and int(left_gripper_index) < state_seq.shape[1]
            else None
        )
        right_gripper = (
            state_seq[:, int(right_gripper_index):int(right_gripper_index) + 1]
            if right_gripper_index is not None and int(right_gripper_index) < state_seq.shape[1]
            else None
        )
        return left_joints, right_joints, left_gripper, right_gripper

    def build_quality_trajectory(self, items: list[dict[str, Any]]) -> CanonicalTrajectory:
        action_seq = self._get_action_seq(items)
        state_seq = np.stack([
            to_numpy(item["observation.state"]).astype(np.float32)
            for item in items
        ], axis=0)
        left_action, right_action = self._split_action(action_seq)
        left_joint, right_joint, left_gripper_state, right_gripper_state = self._split_state(state_seq)

        layout = self._action_layout()
        pos_slice = layout["pos_slice"]
        quat_slice = layout["quat_slice"]
        gripper_indices = self._action_gripper_indices()

        timestamps = self._timestamps(items)

        arms = {
            "left": self._build_arm_trajectory(
                left_action,
                left_joint,
                left_gripper_state,
                pos_slice,
                quat_slice,
                gripper_indices["left"],
            ),
            "right": self._build_arm_trajectory(
                right_action,
                right_joint,
                right_gripper_state,
                pos_slice,
                quat_slice,
                gripper_indices["right"],
            ),
        }
        return CanonicalTrajectory(arms=arms, timestamps=timestamps)

    def _build_arm_trajectory(
        self,
        action_slice: np.ndarray,
        joint_position: np.ndarray | None,
        state_gripper: np.ndarray | None,
        pos_slice: tuple[int, int],
        quat_slice: tuple[int, int],
        gripper_index: int,
    ) -> CanonicalArmTrajectory:
        return CanonicalArmTrajectory(
            position=action_slice[:, pos_slice[0]:pos_slice[1]],
            orientation=action_slice[:, quat_slice[0]:quat_slice[1]],
            gripper=action_slice[:, gripper_index:gripper_index + 1],
            joint_position=joint_position,
            state_gripper=state_gripper,
        )

    def _timestamps(self, items: list[dict[str, Any]]) -> np.ndarray:
        timestamps = []
        fps = float(self.cfg.get("gripper", {}).get("fps", 10.0))
        for index, item in enumerate(items):
            if "timestamp" in item and item["timestamp"] is not None:
                timestamps.append(float(item["timestamp"]))
            else:
                timestamps.append(index / fps)
        return np.asarray(timestamps, dtype=np.float64)

    def detect_moving_arm(
        self,
        items: list[dict[str, Any]],
        ratio_threshold: float = 1.5,
        min_motion: float = 1e-4,
    ) -> dict:
        action_seq = self._get_action_seq(items)
        return self._detect_moving_arm_from_action_seq(
            action_seq,
            ratio_threshold=ratio_threshold,
            min_motion=min_motion,
        )

    def _detect_moving_arm_from_action_seq(
        self,
        action_seq: np.ndarray,
        *,
        ratio_threshold: float,
        min_motion: float,
    ) -> dict:
        left, right = self._split_action(action_seq)
        layout = self._action_layout()
        pos_slice = layout["pos_slice"]
        quat_slice = layout["quat_slice"]
        gripper_index = int(layout["gripper_index"])

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
        moving = self._detect_moving_arm_from_action_seq(
            action_seq,
            ratio_threshold=1.5,
            min_motion=1e-4,
        )["moving_arm"]
        gripper_index = int(self._action_layout()["gripper_index"])

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
        if keyframe_type in {"pre_grasp", "gripper_close", "post_grasp", "gripper_fully_open", "release_keyframe", "pre_retreat", "retreat_start", "post_retreat"}:
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