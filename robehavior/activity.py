from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from trajectory.contracts import ActivityThresholds, TrajectoryContract
from trajectory.models import CanonicalArmTrajectory, CanonicalTrajectory
from trajectory.signals import quaternion_angular_steps, scalar_step_changes, vector_step_norms


MAX_INACTIVE_GAP_SECONDS = 0.2
ACTIVITY_SCORE_WINDOW_SECONDS = 0.5
MAX_NORMALIZED_COMPONENT = 10.0
QUIET_SCORE_MAX = 0.5
ACTIVE_SCORE_MIN = 1.0
ACTIVE_RELATIVE_MIN = 0.5

ActivityIntensity = Literal["quiet", "passive", "active"]


@dataclass(frozen=True)
class ActivityInterval:
    start_transition: int
    stop_transition: int


@dataclass(frozen=True)
class ArmActivity:
    arm_name: str
    score: np.ndarray
    relative_score: np.ndarray
    intensity: np.ndarray
    component_scores: dict[str, np.ndarray]
    active_mask: np.ndarray
    intervals: tuple[ActivityInterval, ...]

    @property
    def active_ratio(self) -> float:
        return float(np.mean(self.active_mask)) if len(self.active_mask) else 0.0

    def intensity_ratio(self, intensity: ActivityIntensity) -> float:
        return float(np.mean(self.intensity == intensity)) if len(self.intensity) else 0.0


@dataclass(frozen=True)
class TrajectoryActivity:
    per_arm: dict[str, ArmActivity]
    any_active_ratio: float
    bimanual_ratio: float

    def to_summary(self) -> dict[str, dict[str, float]]:
        return {
            arm_name: {
                "activity_score_mean": (
                    float(np.mean(activity.score)) if len(activity.score) else 0.0
                ),
                "relative_activity_mean": (
                    float(np.mean(activity.relative_score))
                    if len(activity.relative_score)
                    else 0.0
                ),
                "passive_ratio": activity.intensity_ratio("passive"),
                "behaviorally_active_ratio": activity.intensity_ratio("active"),
            }
            for arm_name, activity in self.per_arm.items()
        }


def infer_moving_arm(
    activity: TrajectoryActivity,
    *,
    ratio_threshold: float = 1.5,
    min_activity: float = QUIET_SCORE_MAX,
) -> dict[str, float | str | None]:
    if len(activity.per_arm) == 1:
        return {
            "moving_arm": "single",
            "left_score": None,
            "right_score": None,
        }

    left = activity.per_arm.get("left")
    right = activity.per_arm.get("right")
    if left is None or right is None:
        raise ValueError("moving-arm inference requires left and right arm activity")
    left_score = float(np.mean(left.score)) if len(left.score) else 0.0
    right_score = float(np.mean(right.score)) if len(right.score) else 0.0

    if left_score < min_activity and right_score < min_activity:
        moving_arm = "none"
    elif left_score > ratio_threshold * right_score:
        moving_arm = "left"
    elif right_score > ratio_threshold * left_score:
        moving_arm = "right"
    else:
        moving_arm = "both"
    return {
        "moving_arm": moving_arm,
        "left_score": left_score,
        "right_score": right_score,
    }


def analyze_trajectory_activity(trajectory: CanonicalTrajectory) -> TrajectoryActivity:
    contract = trajectory.contract
    thresholds = (
        contract.activity_thresholds
        if contract is not None
        else ActivityThresholds(0.005, 0.05, 0.05, 0.01)
    )
    dt = _mean_timestep(trajectory.timestamps)
    per_arm = {
        arm_name: detect_arm_activity(arm_name, arm, contract, thresholds, dt)
        for arm_name, arm in trajectory.arms.items()
    }
    if per_arm:
        peak_score = np.maximum.reduce(
            [activity.score for activity in per_arm.values()]
        )
        per_arm = {
            arm_name: _with_relative_intensity(activity, peak_score)
            for arm_name, activity in per_arm.items()
        }
    masks = [activity.active_mask for activity in per_arm.values()]
    if not masks or not len(masks[0]):
        return TrajectoryActivity(per_arm, any_active_ratio=0.0, bimanual_ratio=0.0)
    transition_count = len(masks[0])
    if any(len(mask) != transition_count for mask in masks):
        raise ValueError("all arms must have the same number of transitions")
    active_counts = np.sum(np.stack(masks), axis=0)
    return TrajectoryActivity(
        per_arm=per_arm,
        any_active_ratio=float(np.mean(active_counts > 0)),
        bimanual_ratio=float(np.mean(active_counts >= 2)),
    )


def detect_arm_activity(
    arm_name: str,
    arm: CanonicalArmTrajectory,
    contract: TrajectoryContract | None,
    thresholds: ActivityThresholds,
    dt: float,
) -> ArmActivity:
    frame_count = max(
        (
            len(signal)
            for signal in (
                arm.action_position,
                arm.action_orientation,
                arm.action_gripper,
                arm.joint_position,
                arm.state_gripper,
            )
            if signal is not None
        ),
        default=0,
    )
    active = np.zeros(max(frame_count - 1, 0), dtype=bool)
    if frame_count < 2 or dt <= 0:
        empty_float = np.zeros_like(active, dtype=float)
        empty_labels = np.full(len(active), "quiet", dtype=object)
        return ArmActivity(
            arm_name,
            empty_float,
            empty_float,
            empty_labels,
            {},
            active,
            (),
        )

    action_mode = contract.action.mode if contract is not None else "absolute"
    component_scores: dict[str, np.ndarray] = {}
    if arm.action_position is not None:
        _validate_length(arm.action_position, frame_count, arm_name)
        translation = (
            vector_step_norms(arm.action_position)
            if action_mode == "absolute"
            else np.linalg.norm(arm.action_position[1:], axis=1)
        )
        translation_speed = translation / dt
        component_scores["translation"] = _normalized_score(
            translation_speed,
            thresholds.translation_speed,
        )
        active |= translation_speed >= thresholds.translation_speed

    if arm.action_orientation is not None:
        _validate_length(arm.action_orientation, frame_count, arm_name)
        rotation = _rotation_steps(arm.action_orientation, contract, action_mode)
        rotation_speed = rotation / dt
        component_scores["rotation"] = _normalized_score(
            rotation_speed,
            thresholds.rotation_speed,
        )
        active |= rotation_speed >= thresholds.rotation_speed

    if arm.joint_position is not None:
        _validate_length(arm.joint_position, frame_count, arm_name)
        joint_speed = vector_step_norms(arm.joint_position) / dt
        component_scores["joint"] = _normalized_score(
            joint_speed,
            thresholds.joint_speed,
        )
        active |= joint_speed >= thresholds.joint_speed

    active = _close_short_inactive_gaps(
        active,
        max_gap=max(1, int(round(MAX_INACTIVE_GAP_SECONDS / dt))),
    )
    gripper = arm.action_gripper if arm.action_gripper is not None else arm.state_gripper
    if gripper is not None:
        _validate_length(gripper, frame_count, arm_name)
        gripper_change = scalar_step_changes(gripper)
        component_scores["gripper"] = _normalized_score(
            gripper_change,
            thresholds.gripper_change,
        )
        active |= gripper_change >= thresholds.gripper_change

    score = _smooth_score(
        np.mean(np.stack(list(component_scores.values())), axis=0),
        max(1, int(round(ACTIVITY_SCORE_WINDOW_SECONDS / dt))),
    )
    return ArmActivity(
        arm_name,
        score,
        np.zeros_like(score),
        np.full(len(score), "quiet", dtype=object),
        component_scores,
        active,
        _active_intervals(active),
    )


def _with_relative_intensity(
    activity: ArmActivity,
    peak_score: np.ndarray,
) -> ArmActivity:
    relative_score = np.divide(
        activity.score,
        peak_score,
        out=np.zeros_like(activity.score),
        where=peak_score > 0,
    )
    intensity = np.full(len(activity.score), "passive", dtype=object)
    intensity[activity.score < QUIET_SCORE_MAX] = "quiet"
    intensity[
        (activity.score >= ACTIVE_SCORE_MIN)
        & (relative_score >= ACTIVE_RELATIVE_MIN)
    ] = "active"
    return replace(
        activity,
        relative_score=relative_score,
        intensity=intensity,
    )


def _normalized_score(values: np.ndarray, threshold: float) -> np.ndarray:
    scale = max(float(threshold), np.finfo(float).eps)
    return np.minimum(np.asarray(values, dtype=float) / scale, MAX_NORMALIZED_COMPONENT)


def _smooth_score(score: np.ndarray, window: int) -> np.ndarray:
    window = min(window, len(score))
    if window <= 1:
        return score
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(score, kernel, mode="same")


def _close_short_inactive_gaps(mask: np.ndarray, *, max_gap: int) -> np.ndarray:
    if len(mask) < 3 or max_gap < 1:
        return mask
    smoothed = mask.copy()
    for interval in _active_intervals(~mask):
        if (
            interval.start_transition > 0
            and interval.stop_transition < len(mask)
            and interval.stop_transition - interval.start_transition <= max_gap
        ):
            smoothed[interval.start_transition:interval.stop_transition] = True
    return smoothed


def _rotation_steps(
    orientation: np.ndarray,
    contract: TrajectoryContract | None,
    action_mode: str,
) -> np.ndarray:
    if action_mode == "delta":
        return np.linalg.norm(orientation[1:], axis=1)
    rotation = contract.action.rotation if contract is not None else None
    if rotation is not None and rotation.representation == "quaternion":
        return quaternion_angular_steps(orientation)
    return vector_step_norms(orientation)


def _active_intervals(mask: np.ndarray) -> tuple[ActivityInterval, ...]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return tuple(
        ActivityInterval(int(start), int(stop))
        for start, stop in zip(starts, stops)
    )


def _mean_timestep(timestamps: np.ndarray | None) -> float:
    if timestamps is not None and len(timestamps) > 1:
        return float(np.mean(np.diff(timestamps)))
    return 0.1


def _validate_length(signal: np.ndarray, frame_count: int, arm_name: str) -> None:
    if len(signal) != frame_count:
        raise ValueError(f"arm '{arm_name}' signals must have equal frame counts")