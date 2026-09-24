from __future__ import annotations

from dataclasses import dataclass

from robehavior.activity import TrajectoryActivity, analyze_trajectory_activity
from robehavior.phases import BehaviorPhases, segment_behavior_phases
from robehavior.roles import BehaviorRoles, infer_arm_roles
from trajectory.models import CanonicalTrajectory


@dataclass(frozen=True)
class BehaviorProfile:
    activity: TrajectoryActivity
    phases: BehaviorPhases
    roles: BehaviorRoles


def analyze_behavior(trajectory: CanonicalTrajectory) -> BehaviorProfile:
    activity = analyze_trajectory_activity(trajectory)
    phases = segment_behavior_phases(trajectory, activity)
    roles = infer_arm_roles(activity, phases)
    return BehaviorProfile(activity=activity, phases=phases, roles=roles)