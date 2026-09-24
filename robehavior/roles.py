from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from robehavior.activity import TrajectoryActivity
from robehavior.phases import BehaviorPhases


ArmRole = Literal["actor", "support", "passive", "idle"]


@dataclass(frozen=True)
class RoleSegment:
    role: ArmRole
    start_frame: int
    stop_frame: int


@dataclass(frozen=True)
class ArmRoleTimeline:
    arm_name: str
    labels: np.ndarray
    segments: tuple[RoleSegment, ...]


@dataclass(frozen=True)
class BehaviorRoles:
    per_arm: dict[str, ArmRoleTimeline]


def infer_arm_roles(
    activity: TrajectoryActivity,
    phases: BehaviorPhases,
) -> BehaviorRoles:
    arm_names = list(activity.per_arm)
    if not arm_names:
        return BehaviorRoles({})
    transition_count = len(activity.per_arm[arm_names[0]].active_mask)
    labels = {
        arm_name: np.full(transition_count, "idle", dtype=object)
        for arm_name in arm_names
    }
    task_phases = {"grasp", "transport", "release", "retreat"}
    task_arms = {
        arm_name
        for arm_name in arm_names
        if any(label in task_phases for label in phases.per_arm[arm_name].labels)
    }
    for transition_index in range(transition_count):
        for arm_name in arm_names:
            phase = phases.per_arm[arm_name].labels[transition_index]
            if phase in task_phases:
                labels[arm_name][transition_index] = "actor"
            elif phase == "unclassified_motion" and arm_name in task_arms:
                labels[arm_name][transition_index] = "actor"
            elif phase == "unclassified_motion":
                intensity = activity.per_arm[arm_name].intensity[transition_index]
                labels[arm_name][transition_index] = (
                    "actor"
                    if not task_arms and intensity == "active"
                    else "passive"
                )
            elif phase == "manipulate":
                intensity = activity.per_arm[arm_name].intensity[transition_index]
                labels[arm_name][transition_index] = (
                    "actor" if intensity == "active" else "passive"
                )
    return BehaviorRoles(
        {
            arm_name: ArmRoleTimeline(
                arm_name,
                arm_labels,
                _merge_segments(arm_labels),
            )
            for arm_name, arm_labels in labels.items()
        }
    )


def _merge_segments(labels: np.ndarray) -> tuple[RoleSegment, ...]:
    if not len(labels):
        return ()
    segments: list[RoleSegment] = []
    start = 0
    current = str(labels[0])
    for index, role in enumerate(labels[1:], start=1):
        if role == current:
            continue
        boundary_frame = index + 1
        segments.append(RoleSegment(current, start, boundary_frame))
        start = boundary_frame
        current = str(role)
    segments.append(RoleSegment(current, start, len(labels) + 1))
    return tuple(segments)