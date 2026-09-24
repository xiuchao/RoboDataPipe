from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from quality.results import QualityFinding
from robehavior.phases import BehaviorPhases, PhaseName
from robehavior.roles import ArmRole, BehaviorRoles


PHASE_NAMES = {
    "idle",
    "unclassified_motion",
    "grasp",
    "transport",
    "release",
    "retreat",
    "manipulate",
}
ARM_ROLES = {"actor", "support", "passive", "idle"}
SEVERITIES = {"info", "warning", "error"}


@dataclass(frozen=True)
class BehaviorExpectation:
    trigger_arm: str
    trigger_phase: PhaseName
    target_arm: str
    expected_role: ArmRole
    severity: Literal["info", "warning", "error"] = "warning"

    def __post_init__(self) -> None:
        if not self.trigger_arm or not self.target_arm:
            raise ValueError("behavior expectation arm names must be non-empty")
        if self.trigger_phase not in PHASE_NAMES:
            raise ValueError(f"unknown behavior phase {self.trigger_phase!r}")
        if self.expected_role not in ARM_ROLES:
            raise ValueError(f"unknown arm role {self.expected_role!r}")
        if self.severity not in SEVERITIES:
            raise ValueError(f"unknown finding severity {self.severity!r}")


def behavior_expectations_from_config(
    dataset_config: dict[str, Any],
) -> tuple[BehaviorExpectation, ...]:
    quality_config = dataset_config.get("quality", {})
    if not isinstance(quality_config, dict):
        raise ValueError("dataset quality config must be a mapping")
    raw_expectations = quality_config.get("behavior_expectations", [])
    if not isinstance(raw_expectations, list):
        raise ValueError("quality.behavior_expectations must be a list")
    expectations: list[BehaviorExpectation] = []
    for index, raw in enumerate(raw_expectations):
        if not isinstance(raw, dict):
            raise ValueError(f"behavior expectation {index} must be a mapping")
        when = raw.get("when")
        expected = raw.get("expect")
        if not isinstance(when, dict) or not isinstance(expected, dict):
            raise ValueError(
                f"behavior expectation {index} must define when and expect mappings"
            )
        try:
            expectations.append(
                BehaviorExpectation(
                    trigger_arm=str(when["arm"]),
                    trigger_phase=when["phase"],
                    target_arm=str(expected["arm"]),
                    expected_role=expected["role"],
                    severity=raw.get("severity", "warning"),
                )
            )
        except KeyError as error:
            raise ValueError(
                f"behavior expectation {index} is missing {error.args[0]!r}"
            ) from error
    return tuple(expectations)


def evaluate_behavior_expectations(
    phases: BehaviorPhases,
    roles: BehaviorRoles,
    expectations: tuple[BehaviorExpectation, ...],
) -> list[QualityFinding]:
    findings: list[QualityFinding] = []
    for expectation in expectations:
        if expectation.trigger_arm not in phases.per_arm:
            raise ValueError(f"unknown trigger arm {expectation.trigger_arm!r}")
        if expectation.target_arm not in roles.per_arm:
            raise ValueError(f"unknown target arm {expectation.target_arm!r}")
        trigger_labels = phases.per_arm[expectation.trigger_arm].labels
        observed_roles = roles.per_arm[expectation.target_arm].labels
        if len(trigger_labels) != len(observed_roles):
            raise ValueError("phase and role timelines must have equal lengths")
        mismatch = (trigger_labels == expectation.trigger_phase) & (
            observed_roles != expectation.expected_role
        )
        for start, stop in _true_runs(mismatch):
            observed = sorted(set(str(value) for value in observed_roles[start:stop]))
            code = (
                "unexpected_static"
                if expectation.expected_role != "idle" and observed == ["idle"]
                else "unexpected_role"
            )
            findings.append(
                QualityFinding(
                    code=code,
                    severity=expectation.severity,
                    arm=expectation.target_arm,
                    start_frame=start,
                    stop_frame=stop + 1,
                    expected=expectation.expected_role,
                    observed=",".join(observed),
                    context=(
                        f"{expectation.trigger_arm}:{expectation.trigger_phase}"
                    ),
                    message=(
                        f"Expected {expectation.target_arm} to be {expectation.expected_role} "
                        f"while {expectation.trigger_arm} was {expectation.trigger_phase}; "
                        f"observed {','.join(observed)}."
                    ),
                )
            )
    return findings


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist()))