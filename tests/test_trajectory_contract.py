from __future__ import annotations

import copy
import dataclasses
import unittest

import numpy as np
import yaml

from quality.evaluator import DatasetQualityAnalyzer
from quality.metrics import quaternion_angular_steps
from robehavior.activity import analyze_trajectory_activity, infer_moving_arm
from robehavior.keyframes import (
    gripper_signal_spec_from_config,
    gripper_value_spec_from_config,
    position_signal_spec_from_config,
    release_behavior_spec_from_config,
)
from trajectory.adapters import build_robot_adapter
from trajectory.contracts import TrajectoryContract, load_trajectory_contract
from trajectory.models import CanonicalArmTrajectory, CanonicalTrajectory


def _registry() -> dict:
    with open("datasets.yaml") as handle:
        return yaml.safe_load(handle)


class TrajectoryContractTest(unittest.TestCase):
    def test_rejects_out_of_bounds_index(self) -> None:
        with open("contracts/dem_pickplace.yaml") as handle:
            data = yaml.safe_load(handle)
        invalid = copy.deepcopy(data)
        invalid["action"]["arms"]["left"]["position"] = [0, 1, 16]

        with self.assertRaisesRegex(ValueError, "index 16 exceeds dim 16"):
            TrajectoryContract.from_dict(invalid)

    def test_dem_contract_maps_left_and_right_signals(self) -> None:
        config = _registry()["DEM_pickplace"]
        items = [
            {
                "action": np.arange(16, dtype=np.float32) + offset,
                "observation.state": np.arange(16, dtype=np.float32) + 1000 + offset,
                "timestamp": offset / 1000,
            }
            for offset in (0, 100)
        ]

        trajectory = build_robot_adapter(config).build_trajectory(items)

        left = trajectory.arms["left"]
        right = trajectory.arms["right"]
        assert left.action_position is not None and right.action_position is not None
        assert left.joint_position is not None and right.joint_position is not None
        assert left.action_gripper is not None and right.action_gripper is not None
        np.testing.assert_array_equal(left.action_position[0], [0, 1, 2])
        np.testing.assert_array_equal(right.action_position[0], [7, 8, 9])
        np.testing.assert_array_equal(left.joint_position[0], np.arange(1000, 1007))
        np.testing.assert_array_equal(right.joint_position[0], np.arange(1007, 1014))
        self.assertEqual(left.action_gripper[0, 0], 14)
        self.assertEqual(right.action_gripper[0, 0], 15)

    def test_ur5_contract_keeps_rotation_vector_separate_from_gripper(self) -> None:
        config = _registry()["DSRFM_easy"]
        item = {
            "action": np.arange(7, dtype=np.float32),
            "observation.state": np.arange(14, dtype=np.float32),
            "timestamp": 0.0,
        }

        arm = build_robot_adapter(config).build_trajectory([item]).arms["single"]

        assert arm.action_orientation is not None
        assert arm.action_gripper is not None
        assert arm.joint_position is not None
        np.testing.assert_array_equal(arm.action_orientation[0], [3, 4, 5])
        self.assertEqual(arm.action_gripper[0, 0], 6)
        np.testing.assert_array_equal(arm.joint_position[0], np.arange(6))

    def test_behavior_detection_and_position_specs_use_contract_arms(self) -> None:
        config = _registry()["DEM_pickplace"]
        adapter = build_robot_adapter(config)
        base_action = np.asarray(
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            dtype=np.float32,
        )
        state = np.zeros(16, dtype=np.float32)
        items = [
            {"action": base_action.copy(), "observation.state": state, "timestamp": 0.0},
            {"action": base_action.copy(), "observation.state": state, "timestamp": 0.1},
        ]
        items[1]["action"][15] = 1.0

        activity = analyze_trajectory_activity(adapter.build_trajectory(items))
        self.assertEqual(infer_moving_arm(activity)["moving_arm"], "right")
        left = position_signal_spec_from_config(config, side="left")
        right = position_signal_spec_from_config(config, side="right")
        self.assertEqual((left.slice_start, left.slice_stop), (0, 3))
        self.assertEqual((right.slice_start, right.slice_stop), (7, 10))

    def test_event_gripper_semantics_are_derived_from_contract(self) -> None:
        registry = _registry()
        config = registry["DEM_pickplace"]
        action = gripper_signal_spec_from_config(config, source="action", side="left")
        state = gripper_signal_spec_from_config(
            config,
            source="observation.state",
            side="right",
        )

        self.assertEqual((action.dim, action.close_direction, action.open_direction), (14, "increase", "decrease"))
        self.assertEqual((state.dim, state.close_direction, state.open_direction), (15, "decrease", "increase"))
        action_value = gripper_value_spec_from_config(config, source="action", side="left")
        state_value = gripper_value_spec_from_config(
            config,
            source="observation.state",
            side="right",
        )
        self.assertEqual(
            (action_value.open_value, action_value.closed_value, action_value.larger_means),
            (0.0, 1.0, "more_closed"),
        )
        self.assertEqual(
            (state_value.open_value, state_value.closed_value, state_value.larger_means),
            (0.8, 0.0, "more_open"),
        )
        release = release_behavior_spec_from_config(
            config,
            gripper_signal_spec=action,
            side="left",
        )
        self.assertEqual(release.fully_open_signal.source, "observation.state")

        dsrfm = gripper_signal_spec_from_config(registry["DSRFM_easy"])
        self.assertEqual(
            (dsrfm.source, dsrfm.dim, dsrfm.close_direction, dsrfm.open_direction),
            ("action", 6, "increase", "decrease"),
        )


class SemanticQualityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = load_trajectory_contract(_registry()["DEM_pickplace"])
        self.timestamps = np.arange(8) / 10
        self.arm = CanonicalArmTrajectory(
            action_position=np.column_stack((np.arange(8) * 0.01, np.zeros(8), np.zeros(8))),
            action_orientation=np.tile([0.0, 0.0, 0.0, 1.0], (8, 1)),
            action_gripper=np.zeros((8, 1)),
            joint_position=np.column_stack((np.arange(8) * 0.02, np.zeros((8, 6)))),
        )

    def test_quaternion_sign_flip_has_zero_rotation(self) -> None:
        steps = quaternion_angular_steps(
            np.asarray([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]])
        )
        np.testing.assert_allclose(steps, [0.0])

    def test_absolute_ee_metrics_are_computed_per_arm(self) -> None:
        quality = self._analyze(self.contract)

        assert quality.translation_path_length is not None
        assert quality.joint_path_length is not None
        self.assertGreater(quality.translation_path_length, 0.0)
        self.assertEqual(quality.rotation_path_length, 0.0)
        self.assertGreater(quality.joint_path_length, 0.0)
        self.assertEqual(
            quality.not_applicable,
            {
                "gripper_chatter_rate": (
                    "binary commands encode intended state changes, not actuator chatter"
                )
            },
        )

    def test_delta_ee_metrics_are_not_applied_as_absolute_poses(self) -> None:
        delta_contract = dataclasses.replace(
            self.contract,
            action=dataclasses.replace(self.contract.action, mode="delta"),
        )

        quality = self._analyze(delta_contract)

        self.assertIsNone(quality.translation_path_length)
        self.assertIsNone(quality.rotation_path_length)
        assert quality.joint_path_length is not None
        self.assertGreater(quality.joint_path_length, 0.0)
        self.assertIn("action mode is delta", quality.not_applicable["translation_metrics"])

    def _analyze(self, contract: TrajectoryContract):
        trajectory = CanonicalTrajectory(
            arms={"left": self.arm},
            timestamps=self.timestamps,
            contract=contract,
        )
        return DatasetQualityAnalyzer().analyze_episode("episode", trajectory).per_arm["left"]


if __name__ == "__main__":
    unittest.main()