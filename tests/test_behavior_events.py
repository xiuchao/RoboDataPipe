from __future__ import annotations

import unittest

import numpy as np
import yaml

from robehavior.activity import analyze_trajectory_activity
from robehavior.events import Event
from robehavior.keyframes import resolve_keyframe_specs
from robehavior.phases import build_phase_event_timeline, segment_behavior_phases
from trajectory.contracts import load_trajectory_contract
from trajectory.models import CanonicalArmTrajectory, CanonicalTrajectory


class EventTimelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open("datasets.yaml") as handle:
            registry = yaml.safe_load(handle)
        cls.contract = load_trajectory_contract(registry["DEM_pickplace"])

    def test_timeline_detects_gripper_events(self) -> None:
        trajectory = CanonicalTrajectory(
            arms={
                "left": CanonicalArmTrajectory(
                    action_gripper=np.asarray([[0.0], [0.0], [1.0], [1.0], [0.0]])
                ),
                "right": CanonicalArmTrajectory(
                    action_gripper=np.zeros((5, 1))
                ),
            },
            timestamps=np.arange(5) * 0.1,
            contract=self.contract,
        )

        phases = segment_behavior_phases(trajectory, analyze_trajectory_activity(trajectory))
        timeline = build_phase_event_timeline(phases, "left")

        self.assertEqual(timeline.events["grasp_start"].local_index, 2)
        self.assertEqual(timeline.events["release_start"].local_index, 4)

    def test_keyframe_policy_only_consumes_event_anchors(self) -> None:
        events = {
            "grasp_start": Event("grasp_start", 10, 1.0, "test"),
            "release_start": Event("release_start", 30, 1.0, "test"),
        }

        specs = resolve_keyframe_specs(events, item_count=40, offset=5)

        self.assertEqual(specs["pre_grasp"][0], 5)
        self.assertEqual(specs["post_grasp"][0], 15)
        self.assertEqual(specs["pre_place"][0], 25)
        self.assertEqual(specs["post_place"][0], 35)

    def test_timeline_keeps_first_repeated_phase_boundary(self) -> None:
        trajectory = CanonicalTrajectory(
            arms={
                "left": CanonicalArmTrajectory(
                    state_gripper=np.asarray([[0.0], [0.2], [0.2], [0.4], [0.4]])
                )
            },
            timestamps=np.arange(5) * 0.1,
            contract=self.contract,
        )
        phases = segment_behavior_phases(trajectory, analyze_trajectory_activity(trajectory))

        timeline = build_phase_event_timeline(phases, "left")

        self.assertEqual(timeline.events["release_start"].local_index, 1)


if __name__ == "__main__":
    unittest.main()