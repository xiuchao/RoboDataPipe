from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from robehavior.activity import TrajectoryActivity
from robehavior.events import Event, EventTimeline
from trajectory.contracts import GripperSignalContract
from trajectory.models import CanonicalArmTrajectory, CanonicalTrajectory


PhaseName = Literal[
    "idle",
    "unclassified_motion",
    "grasp",
    "transport",
    "release",
    "retreat",
    "manipulate",
]


@dataclass(frozen=True)
class BehaviorPhase:
    phase: PhaseName
    start_frame: int
    stop_frame: int


@dataclass(frozen=True)
class ArmPhaseTimeline:
    arm_name: str
    labels: np.ndarray
    segments: tuple[BehaviorPhase, ...]


@dataclass(frozen=True)
class BehaviorPhases:
    per_arm: dict[str, ArmPhaseTimeline]


def build_phase_event_timeline(
    phases: BehaviorPhases,
    arm_name: str,
) -> EventTimeline:
    phase_timeline = phases.per_arm[arm_name]
    event_names = {
        "grasp": "grasp_start",
        "release": "release_start",
        "retreat": "retreat_start",
    }
    events: dict[str, Event] = {}
    for segment in phase_timeline.segments:
        event_name = event_names.get(segment.phase)
        if event_name is None or event_name in events:
            continue
        events[event_name] = Event(
            name=event_name,
            local_index=max(segment.start_frame, 1),
            score=1.0,
            method=f"phase_boundary:{segment.phase}",
        )
    return EventTimeline(events=events)


def segment_behavior_phases(
    trajectory: CanonicalTrajectory,
    activity: TrajectoryActivity,
) -> BehaviorPhases:
    return BehaviorPhases(
        per_arm={
            arm_name: _segment_arm_phases(
                arm_name,
                arm,
                activity.per_arm[arm_name].active_mask,
                trajectory,
            )
            for arm_name, arm in trajectory.arms.items()
        }
    )


def _segment_arm_phases(
    arm_name: str,
    arm: CanonicalArmTrajectory,
    active_mask: np.ndarray,
    trajectory: CanonicalTrajectory,
) -> ArmPhaseTimeline:
    signal, semantics = _gripper_signal_and_semantics(arm, trajectory)
    labels = np.full(len(active_mask), "idle", dtype=object)
    if signal is None or semantics is None:
        labels[active_mask] = "manipulate"
        return ArmPhaseTimeline(arm_name, labels, _merge_segments(labels))

    values = signal[:, 0] if signal.ndim == 2 else signal
    if len(values) != len(active_mask) + 1:
        raise ValueError(f"arm '{arm_name}' gripper signal must align with activity transitions")
    threshold = (
        trajectory.contract.activity_thresholds.gripper_change
        if trajectory.contract is not None
        else 0.01
    )
    released = False
    for transition_index, is_active in enumerate(active_mask):
        before = float(values[transition_index])
        after = float(values[transition_index + 1])
        change = abs(after - before)
        if change >= threshold and _moves_toward(after, before, semantics.closed_value):
            labels[transition_index] = "grasp"
            released = False
        elif (
            change >= threshold
            and not _near_target(before, semantics.open_value, semantics)
            and _moves_toward(after, before, semantics.open_value)
        ):
            labels[transition_index] = "release"
            released = True
        elif released and not _near_target(after, semantics.open_value, semantics):
            labels[transition_index] = "release"
        elif is_active:
            if released:
                labels[transition_index] = "retreat"
            else:
                labels[transition_index] = (
                    "transport"
                    if _nearest_state(after, semantics) == "closed"
                    else "unclassified_motion"
                )
    return ArmPhaseTimeline(arm_name, labels, _merge_segments(labels))


def _gripper_signal_and_semantics(
    arm: CanonicalArmTrajectory,
    trajectory: CanonicalTrajectory,
) -> tuple[np.ndarray | None, GripperSignalContract | None]:
    contract = trajectory.contract
    if arm.state_gripper is not None:
        return arm.state_gripper, None if contract is None else contract.gripper_state
    if arm.action_gripper is not None:
        return arm.action_gripper, None if contract is None else contract.gripper_action
    return None, None


def _moves_toward(after: float, before: float, target: float) -> bool:
    return abs(after - target) < abs(before - target)


def _nearest_state(value: float, semantics: GripperSignalContract) -> Literal["open", "closed"]:
    return (
        "closed"
        if abs(value - semantics.closed_value) <= abs(value - semantics.open_value)
        else "open"
    )


def _near_target(
    value: float,
    target: float,
    semantics: GripperSignalContract,
) -> bool:
    tolerance = max(abs(semantics.open_value - semantics.closed_value) * 0.05, 1e-6)
    return abs(value - target) <= tolerance


def _merge_segments(labels: np.ndarray) -> tuple[BehaviorPhase, ...]:
    if not len(labels):
        return ()
    segments: list[BehaviorPhase] = []
    start = 0
    current = str(labels[0])
    for index, label in enumerate(labels[1:], start=1):
        if label == current:
            continue
        boundary_frame = index + 1
        segments.append(BehaviorPhase(current, start, boundary_frame))
        start = boundary_frame
        current = str(label)
    segments.append(BehaviorPhase(current, start, len(labels) + 1))
    return tuple(segments)