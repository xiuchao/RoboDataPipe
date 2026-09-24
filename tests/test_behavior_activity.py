from __future__ import annotations

import unittest

import numpy as np
import yaml

from quality.evaluator import DatasetQualityAnalyzer
from robehavior.activity import analyze_trajectory_activity
from trajectory.contracts import load_trajectory_contract
from trajectory.models import CanonicalArmTrajectory, CanonicalTrajectory


class BehaviorActivityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open("datasets.yaml") as handle:
            registry = yaml.safe_load(handle)
        cls.contract = load_trajectory_contract(registry["DEM_pickplace"])

    def test_activity_is_independent_per_arm(self) -> None:
        trajectory = CanonicalTrajectory(
            arms={
                "left": _arm([0.0, 0.01, 0.02, 0.03]),
                "right": _arm([0.0, 0.0, 0.02, 0.02]),
            },
            timestamps=np.arange(4) * 0.1,
            contract=self.contract,
        )

        activity = analyze_trajectory_activity(trajectory)

        np.testing.assert_array_equal(activity.per_arm["left"].active_mask, [True, True, True])
        np.testing.assert_array_equal(activity.per_arm["right"].active_mask, [False, True, False])
        self.assertAlmostEqual(activity.bimanual_ratio, 1.0 / 3.0)
        self.assertEqual(activity.any_active_ratio, 1.0)

    def test_gripper_change_counts_as_activity(self) -> None:
        arm = _arm([0.0, 0.0, 0.0, 0.0], gripper=[0.0, 0.0, 1.0, 1.0])
        trajectory = CanonicalTrajectory(
            arms={"left": arm},
            timestamps=np.arange(4) * 0.1,
            contract=self.contract,
        )

        activity = analyze_trajectory_activity(trajectory)

        np.testing.assert_array_equal(activity.per_arm["left"].active_mask, [False, True, False])

    def test_single_transition_gap_is_smoothed(self) -> None:
        trajectory = CanonicalTrajectory(
            arms={"left": _arm([0.0, 0.01, 0.01, 0.02])},
            timestamps=np.arange(4) * 0.1,
            contract=self.contract,
        )

        activity = analyze_trajectory_activity(trajectory)

        np.testing.assert_array_equal(activity.per_arm["left"].active_mask, [True, True, True])

    def test_two_transition_gap_is_smoothed_at_ten_hertz(self) -> None:
        trajectory = CanonicalTrajectory(
            arms={"left": _arm([0.0, 0.01, 0.01, 0.01, 0.02])},
            timestamps=np.arange(5) * 0.1,
            contract=self.contract,
        )

        activity = analyze_trajectory_activity(trajectory)

        np.testing.assert_array_equal(
            activity.per_arm["left"].active_mask,
            [True, True, True, True],
        )

    def test_smaller_motion_is_passive_relative_to_dominant_arm(self) -> None:
        trajectory = CanonicalTrajectory(
            arms={
                "left": _arm([0.0, 0.001, 0.002, 0.003]),
                "right": _arm([0.0, 0.01, 0.02, 0.03]),
            },
            timestamps=np.arange(4) * 0.1,
            contract=self.contract,
        )

        activity = analyze_trajectory_activity(trajectory)

        np.testing.assert_array_equal(
            activity.per_arm["left"].intensity,
            ["passive", "passive", "passive"],
        )
        np.testing.assert_array_equal(
            activity.per_arm["right"].intensity,
            ["active", "active", "active"],
        )
        self.assertTrue(
            np.all(
                activity.per_arm["left"].relative_score
                < activity.per_arm["right"].relative_score
            )
        )

    def test_idle_arm_is_described_without_being_flagged(self) -> None:
        trajectory = CanonicalTrajectory(
            arms={
                "left": _arm([0.0, 0.01, 0.02, 0.03]),
                "right": _arm([0.0, 0.0, 0.0, 0.0]),
            },
            timestamps=np.arange(4) * 0.1,
            contract=self.contract,
        )

        assessment = DatasetQualityAnalyzer().analyze_episode("episode", trajectory)

        self.assertEqual(
            assessment.behavior_summary["right"]["behaviorally_active_ratio"],
            0.0,
        )
        self.assertIsNone(assessment.per_arm["right"].overall_score)
        self.assertEqual(assessment.per_arm["right"].flags, [])


def _arm(
    x_positions: list[float],
    *,
    gripper: list[float] | None = None,
) -> CanonicalArmTrajectory:
    frame_count = len(x_positions)
    return CanonicalArmTrajectory(
        action_position=np.column_stack(
            (x_positions, np.zeros(frame_count), np.zeros(frame_count))
        ),
        action_gripper=np.asarray(
            gripper if gripper is not None else np.zeros(frame_count),
            dtype=np.float64,
        )[:, None],
    )


if __name__ == "__main__":
    unittest.main()