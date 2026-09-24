from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trajectory.contracts import TrajectoryContract


@dataclass(frozen=True)
class CanonicalArmTrajectory:
    action_position: np.ndarray | None = None
    action_orientation: np.ndarray | None = None
    action_gripper: np.ndarray | None = None
    joint_position: np.ndarray | None = None
    state_gripper: np.ndarray | None = None


@dataclass(frozen=True)
class CanonicalTrajectory:
    arms: dict[str, CanonicalArmTrajectory] = field(default_factory=dict)
    timestamps: np.ndarray | None = None
    contract: TrajectoryContract | None = None