from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from quality.contextual_rules import BehaviorExpectation, behavior_expectations_from_config
from quality.calibration import (
    QualityCalibration,
    apply_quality_calibration,
    calibrate_quality_thresholds,
)
from quality.dashboard import quality_report_to_html
from quality.evaluator import DatasetQualityAnalyzer
from quality.metrics import segmented_trajectory_efficiency
from quality.review import quality_report_to_markdown
from quality.results import EpisodeQuality, QualityReport, TimestampQuality
from quality.rules import assess_learning_quality
from trajectory.contracts import load_trajectory_contract
from trajectory.models import CanonicalArmTrajectory, CanonicalTrajectory


class ContextualQualityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with open("datasets.yaml") as handle:
            registry = yaml.safe_load(handle)
        cls.contract = load_trajectory_contract(registry["DEM_pickplace"])

    def test_static_support_arm_is_only_flagged_when_expected(self) -> None:
        trajectory = _trajectory(self.contract)

        without_expectation = DatasetQualityAnalyzer().analyze_episode("episode", trajectory)
        with_expectation = DatasetQualityAnalyzer(
            (
                BehaviorExpectation(
                    trigger_arm="left",
                    trigger_phase="transport",
                    target_arm="right",
                    expected_role="support",
                    severity="error",
                ),
            )
        ).analyze_episode("episode", trajectory)

        self.assertEqual(without_expectation.findings, [])
        self.assertNotIn("unexpected_static", without_expectation.flags)
        self.assertEqual(len(with_expectation.findings), 1)
        finding = with_expectation.findings[0]
        self.assertEqual(finding.code, "unexpected_static")
        self.assertEqual(finding.arm, "right")
        self.assertEqual((finding.start_frame, finding.stop_frame), (2, 5))
        self.assertIn("unexpected_static", with_expectation.flags)

    def test_passive_arm_is_excluded_from_task_quality_score(self) -> None:
        trajectory = CanonicalTrajectory(
            arms={
                "left": _arm(
                    np.arange(10) * 0.001,
                    [0.0] * 10,
                ),
                "right": _arm(
                    np.arange(10) * 0.01,
                    [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                ),
            },
            timestamps=np.arange(10) * 0.1,
            contract=self.contract,
        )

        assessment = DatasetQualityAnalyzer().analyze_episode("episode", trajectory)
        left = assessment.per_arm["left"]
        right = assessment.per_arm["right"]

        self.assertFalse(left.is_active)
        self.assertIsNone(left.overall_score)
        self.assertIsNone(left.translation_path_length)
        self.assertIsNotNone(left.activity_reason)
        assert left.activity_reason is not None
        self.assertIn("passive/idle motion excluded", left.activity_reason)
        self.assertTrue(right.is_active)
        self.assertIsNotNone(right.overall_score)
        self.assertIsNone(right.gripper_chatter_rate)
        self.assertIn("gripper_chatter_rate", right.not_applicable)
        self.assertEqual(assessment.overall_score, right.overall_score)

    def test_efficiency_is_segmented_by_movement_phase(self) -> None:
        trajectory = CanonicalTrajectory(
            arms={
                "right": _arm(
                    np.asarray([0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 2.0, 1.0]),
                    [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                )
            },
            timestamps=np.arange(8) * 0.1,
            contract=self.contract,
        )

        assessment = DatasetQualityAnalyzer().analyze_episode("episode", trajectory)

        efficiency = assessment.per_arm["right"].trajectory_efficiency
        self.assertIsNotNone(efficiency)
        assert efficiency is not None
        self.assertAlmostEqual(efficiency, 1.0)

    def test_segmented_metrics_exclude_transitions_outside_role(self) -> None:
        positions = np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [10.0, 1.0],
            ]
        )

        efficiency = segmented_trajectory_efficiency(
            positions,
            np.asarray([True, True, False]),
        )

        self.assertAlmostEqual(efficiency, np.sqrt(2.0) / 2.0)

    def test_overall_score_uses_fixed_learning_quality_weights(self) -> None:
        score, flags = assess_learning_quality(-12.96, 0.47, 0.12, -10.79)
        expected = 10.0 * (
            0.35 * (1.0 / (1.0 + np.exp(-(-12.96 + 18.0) / 4.0)))
            + 0.35 * 0.47
            + 0.20 * (1.0 - 0.12)
            + 0.10 * (1.0 / (1.0 + np.exp(-(-10.79 + 18.0) / 4.0)))
        )

        self.assertAlmostEqual(score, expected)
        self.assertEqual(flags, [])

    def test_overall_score_requires_all_learning_quality_components(self) -> None:
        score, _ = assess_learning_quality(-12.96, 0.47, 0.12, None)

        self.assertIsNone(score)

    def test_reference_quantiles_flag_relative_outliers_without_changing_score(self) -> None:
        episodes = []
        for index in range(20):
            arm = EpisodeQuality(
                f"episode_{index:03d}:right",
                is_active=True,
                translation_smoothness=float(index),
                joint_smoothness=float(index + 10),
                trajectory_efficiency=float(index) / 20.0,
                hesitation_fraction=float(index) / 100.0,
                overall_score=7.0,
            )
            episodes.append(
                EpisodeQuality(
                    f"episode_{index:03d}",
                    overall_score=7.0,
                    per_arm={"right": arm},
                )
            )
        report = QualityReport("/tmp/reference", per_episode=episodes, num_episodes=20)
        reference_ids = {episode.episode_id for episode in episodes}

        calibration = calibrate_quality_thresholds(
            report,
            "demo",
            reference_ids,
        )

        thresholds = calibration.per_arm["right"].thresholds
        self.assertAlmostEqual(thresholds["translation_smoothness"].value, 0.95)
        self.assertEqual(thresholds["translation_smoothness"].direction, "below")
        self.assertAlmostEqual(thresholds["hesitation_fraction"].value, 0.1805)
        self.assertEqual(thresholds["hesitation_fraction"].direction, "above")

        apply_quality_calibration(report, calibration)

        self.assertIn(
            "reference_translation_smoothness_below",
            report.per_episode[0].per_arm["right"].flags,
        )
        self.assertIn(
            "reference_hesitation_fraction_above",
            report.per_episode[-1].per_arm["right"].flags,
        )
        self.assertEqual(report.per_episode[0].overall_score, 7.0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            calibration.to_json(path)
            loaded = QualityCalibration.from_json(path)
        self.assertEqual(loaded, calibration)

    def test_assessment_serializes_behavior_profile(self) -> None:
        assessment = DatasetQualityAnalyzer().analyze_episode(
            "episode",
            _trajectory(self.contract),
        )

        payload = assessment.to_dict()

        self.assertTrue(payload["validity"])
        self.assertEqual(
            payload["phases"]["left"][0]["phase"],
            "unclassified_motion",
        )
        self.assertEqual(payload["roles"]["right"][0]["role"], "idle")
        self.assertNotIn("findings", payload)

    def test_episode_serialization_omits_unset_and_empty_fields(self) -> None:
        payload = EpisodeQuality("episode", active_fraction=0.0, is_active=False).to_dict()

        self.assertNotIn("smoothness", payload)
        self.assertNotIn("behavior_summary", payload)
        self.assertNotIn("findings", payload)
        self.assertNotIn("not_applicable", payload)
        self.assertNotIn("learning_quality", payload)
        self.assertNotIn("diagnostics", payload)
        self.assertNotIn("smoothness", payload)
        self.assertNotIn("cartesian_path_length", payload)
        self.assertEqual(payload["active_fraction"], 0.0)
        self.assertIs(payload["is_active"], False)

    def test_arm_metrics_are_serialized_by_purpose(self) -> None:
        quality = EpisodeQuality(
            "episode:right",
            translation_smoothness=-12.96,
            joint_smoothness=-10.79,
            joint_path_length=5.95,
            translation_path_length=1.29,
            rotation_path_length=5.79,
            angular_speed_mean=0.38,
            angular_acceleration_rms=4.95,
            trajectory_efficiency=0.47,
            hesitation_fraction=0.12,
        )

        payload = quality.to_dict()

        self.assertEqual(
            set(payload["learning_quality"]),
            {
                "translation_smoothness",
                "joint_smoothness",
                "trajectory_efficiency",
                "hesitation_fraction",
            },
        )
        self.assertEqual(
            set(payload["diagnostics"]),
            {
                "joint_path_length",
                "translation_path_length",
                "rotation_path_length",
                "angular_speed_mean",
                "angular_acceleration_rms",
            },
        )
        self.assertNotIn("translation_smoothness", payload)
        self.assertNotIn("joint_path_length", payload)

    def test_dataset_report_markdown_summarizes_sample_and_review_priority(self) -> None:
        first_arm = EpisodeQuality(
            "episode_000:right",
            is_active=True,
            translation_smoothness=-13.0,
            joint_smoothness=-11.0,
            trajectory_efficiency=0.4,
            hesitation_fraction=0.2,
            joint_path_length=6.0,
            translation_path_length=1.5,
            rotation_path_length=5.0,
            angular_speed_mean=0.5,
            angular_acceleration_rms=6.0,
            timestamp=TimestampQuality(dt_mean=0.1, dt_std=0.01, jitter_ratio=0.1),
            overall_score=6.0,
        )
        second_arm = EpisodeQuality(
            "episode_001:right",
            is_active=True,
            translation_smoothness=-11.0,
            joint_smoothness=-9.0,
            trajectory_efficiency=0.8,
            hesitation_fraction=0.1,
            joint_path_length=4.0,
            translation_path_length=1.0,
            rotation_path_length=3.0,
            angular_speed_mean=0.3,
            angular_acceleration_rms=4.0,
            overall_score=8.0,
        )
        report = QualityReport(
            dataset_path="/tmp/dataset",
            num_episodes=2,
            overall_score=7.0,
            per_episode=[
                EpisodeQuality("episode_000", overall_score=6.0, per_arm={"right": first_arm}),
                EpisodeQuality("episode_001", overall_score=8.0, per_arm={"right": second_arm}),
            ],
        )

        markdown = quality_report_to_markdown(report, "demo", sample=2)

        self.assertIn("First 2 episodes (sample)", markdown)
        self.assertIn("| Overall score | 2 | 7.00 | 7.00", markdown)
        self.assertIn("## Diagnostic Distribution", markdown)
        self.assertIn("| Joint path length | 2 | 0 | 5.00 |", markdown)
        self.assertIn("| Timestamp jitter ratio | 1 | 1 | 0.10 |", markdown)
        self.assertIn("do not contribute to `overall_score`", markdown)
        self.assertIn("Episodes outside P10-P90", markdown)
        self.assertIn("1. `episode_000`: score `6.00`", markdown)
        self.assertIn("not the full dataset", markdown)

        dashboard = quality_report_to_html(report, "demo", sample=2)

        self.assertIn("demo Dataset Quality", dashboard)
        self.assertIn("Dataset overview", dashboard)
        self.assertIn("Diagnostic distributions", dashboard)
        self.assertIn("Review queue", dashboard)
        self.assertIn("ep000", dashboard)
        self.assertNotIn("episode_000", dashboard)
        self.assertIn("Plotly.newPlot", dashboard)

    def test_expectations_are_loaded_from_dataset_config(self) -> None:
        expectations = behavior_expectations_from_config(
            {
                "quality": {
                    "behavior_expectations": [
                        {
                            "when": {"arm": "left", "phase": "transport"},
                            "expect": {"arm": "right", "role": "support"},
                            "severity": "error",
                        }
                    ]
                }
            }
        )

        self.assertEqual(
            expectations,
            (
                BehaviorExpectation(
                    "left", "transport", "right", "support", "error"
                ),
            ),
        )

    def test_invalid_expectation_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown behavior phase"):
            BehaviorExpectation(
                "left",
                cast(Any, "flying"),
                "right",
                "support",
            )

    def test_report_serialization_rounds_all_floats_to_two_places(self) -> None:
        arm = EpisodeQuality("episode:left", overall_score=1.2345)
        episode = EpisodeQuality(
            "episode",
            behavior_summary={"left": {"activity": 0.12345}},
            overall_score=7.456,
            per_arm={"left": arm},
        )
        report = QualityReport(
            "dataset",
            overall_score=8.7654,
            per_episode=[episode],
            per_arm_summary={"left": {"score": 4.5678}},
        )

        payload = report.to_dict()

        self.assertEqual(payload["overall_score"], 8.77)
        self.assertEqual(payload["per_episode"][0]["overall_score"], 7.46)
        self.assertEqual(
            payload["per_episode"][0]["behavior_summary"]["left"]["activity"],
            0.12,
        )
        self.assertEqual(
            payload["per_episode"][0]["per_arm"]["left"]["overall_score"],
            1.23,
        )
        self.assertEqual(payload["per_arm_summary"]["left"]["score"], 4.57)


def _trajectory(contract) -> CanonicalTrajectory:
    left_x = np.asarray([0.0, 0.01, 0.02, 0.03, 0.04, 0.04])
    return CanonicalTrajectory(
        arms={
            "left": _arm(left_x, [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]),
            "right": _arm(np.zeros(6), [0.0] * 6),
        },
        timestamps=np.arange(6) * 0.1,
        contract=contract,
    )


def _arm(x: np.ndarray, gripper: list[float]) -> CanonicalArmTrajectory:
    return CanonicalArmTrajectory(
        action_position=np.column_stack((x, np.zeros(len(x)), np.zeros(len(x)))),
        action_gripper=np.asarray(gripper)[:, None],
        joint_position=np.column_stack((x, np.zeros((len(x), 6)))),
    )


if __name__ == "__main__":
    unittest.main()