from __future__ import annotations

import unittest

import numpy as np
import yaml

from robehavior.activity import analyze_trajectory_activity
from robehavior.phases import segment_behavior_phases
from robehavior.roles import infer_arm_roles
from trajectory.contracts import load_trajectory_contract
from trajectory.models import CanonicalArmTrajectory, CanonicalTrajectory


class BehaviorPhaseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open("datasets.yaml") as handle:
            registry = yaml.safe_load(handle)
        cls.contract = load_trajectory_contract(registry["DEM_pickplace"])

    def test_rule_based_phase_sequence(self) -> None:
        trajectory = _trajectory(self.contract, right_moves=False)
        activity = analyze_trajectory_activity(trajectory)
        phases = segment_behavior_phases(trajectory, activity)

        self.assertEqual(
            phases.per_arm["left"].labels.tolist(),
            ["unclassified_motion", "grasp", "transport", "transport", "release"],
        )
        self.assertEqual(
            [segment.phase for segment in phases.per_arm["left"].segments],
            ["unclassified_motion", "grasp", "transport", "release"],
        )
        self.assertEqual(
            [
                (segment.start_frame, segment.stop_frame)
                for segment in phases.per_arm["left"].segments
            ],
            [(0, 2), (2, 3), (3, 5), (5, 6)],
        )

    def test_roles_keep_inactive_arm_idle(self) -> None:
        trajectory = _trajectory(self.contract, right_moves=False)
        activity = analyze_trajectory_activity(trajectory)
        phases = segment_behavior_phases(trajectory, activity)
        roles = infer_arm_roles(activity, phases)

        self.assertEqual(roles.per_arm["right"].labels.tolist(), ["idle"] * 5)
        self.assertEqual(roles.per_arm["left"].labels.tolist(), ["actor"] * 5)

    def test_motion_after_release_is_retreat(self) -> None:
        trajectory = CanonicalTrajectory(
            arms={
                "left": _arm(
                    np.asarray([0.0, 0.01, 0.02, 0.03, 0.04]),
                    [0.0, 1.0, 1.0, 0.0, 0.0],
                )
            },
            timestamps=np.arange(5) * 0.1,
            contract=self.contract,
        )

        activity = analyze_trajectory_activity(trajectory)
        phases = segment_behavior_phases(trajectory, activity)

        self.assertEqual(
            phases.per_arm["left"].labels.tolist(),
            ["grasp", "transport", "release", "retreat"],
        )

    def test_open_tolerance_adjustment_does_not_restart_release(self) -> None:
        trajectory = CanonicalTrajectory(
            arms={
                "left": CanonicalArmTrajectory(
                    action_position=np.column_stack(
                        (np.arange(5) * 0.01, np.zeros(5), np.zeros(5))
                    ),
                    state_gripper=np.asarray([0.0, 0.78, 0.78, 0.795, 0.795])[:, None],
                )
            },
            timestamps=np.arange(5) * 0.1,
            contract=self.contract,
        )

        activity = analyze_trajectory_activity(trajectory)
        phases = segment_behavior_phases(trajectory, activity)

        self.assertEqual(
            phases.per_arm["left"].labels.tolist(),
            ["release", "retreat", "retreat", "retreat"],
        )

    def test_roles_classify_smaller_accompanying_motion_as_passive(self) -> None:
        trajectory = CanonicalTrajectory(
            arms={
                "left": _arm(
                    np.asarray([0.0, 0.001, 0.002, 0.003]),
                    [0.0] * 4,
                ),
                "right": _arm(
                    np.asarray([0.0, 0.01, 0.02, 0.03]),
                    [0.0] * 4,
                ),
            },
            timestamps=np.arange(4) * 0.1,
            contract=self.contract,
        )

        activity = analyze_trajectory_activity(trajectory)
        phases = segment_behavior_phases(trajectory, activity)
        roles = infer_arm_roles(activity, phases)

        np.testing.assert_array_equal(
            roles.per_arm["left"].labels,
            ["passive", "passive", "passive"],
        )
        np.testing.assert_array_equal(
            roles.per_arm["right"].labels,
            ["actor", "actor", "actor"],
        )

    def test_non_task_arm_stays_passive_during_pick_place(self) -> None:
        trajectory = CanonicalTrajectory(
            arms={
                "left": _arm(
                    np.asarray([0.0, 0.001, 0.002, 0.003, 0.004, 0.005]),
                    [0.0] * 6,
                ),
                "right": _arm(
                    np.asarray([0.0, 0.01, 0.02, 0.03, 0.04, 0.05]),
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                ),
            },
            timestamps=np.arange(6) * 0.1,
            contract=self.contract,
        )

        activity = analyze_trajectory_activity(trajectory)
        phases = segment_behavior_phases(trajectory, activity)
        roles = infer_arm_roles(activity, phases)

        self.assertNotIn("actor", roles.per_arm["left"].labels)
        self.assertIn("actor", roles.per_arm["right"].labels)
        self.assertEqual(
            [
                (segment.role, segment.start_frame, segment.stop_frame)
                for segment in roles.per_arm["left"].segments
            ],
            [("passive", 0, 6)],
        )


def _trajectory(contract, *, right_moves: bool) -> CanonicalTrajectory:
    left_x = np.asarray([0.0, 0.01, 0.02, 0.03, 0.04, 0.04])
    right_x = np.asarray([0.0, 0.0, 0.0, 0.01, 0.02, 0.02]) if right_moves else np.zeros(6)
    return CanonicalTrajectory(
        arms={
            "left": _arm(left_x, [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]),
            "right": _arm(right_x, [0.0] * 6),
        },
        timestamps=np.arange(6) * 0.1,
        contract=contract,
    )


def _arm(x: np.ndarray, gripper: list[float]) -> CanonicalArmTrajectory:
    return CanonicalArmTrajectory(
        action_position=np.column_stack((x, np.zeros(len(x)), np.zeros(len(x)))),
        action_gripper=np.asarray(gripper)[:, None],
    )


if __name__ == "__main__":
    unittest.main()