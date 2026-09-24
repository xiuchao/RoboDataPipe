from __future__ import annotations

from collections import defaultdict

import numpy as np

from quality.metrics import (
    active_path_length,
    masked_gripper_chatter_rate,
    rotation_metrics,
    segmented_hesitation_fraction,
    segmented_log_dimensionless_jerk,
    segmented_trajectory_efficiency,
    timestamp_quality,
)
from quality.contextual_rules import BehaviorExpectation, evaluate_behavior_expectations
from quality.results import EpisodeQuality, QualityReport
from quality.rules import assess_learning_quality
from robehavior.activity import ArmActivity
from robehavior.phases import ArmPhaseTimeline
from robehavior.profile import analyze_behavior
from robehavior.roles import ArmRoleTimeline
from trajectory.contracts import TrajectoryContract
from trajectory.models import CanonicalArmTrajectory, CanonicalTrajectory


class DatasetQualityAnalyzer:
    def __init__(self, expectations: tuple[BehaviorExpectation, ...] = ()) -> None:
        self.expectations = expectations

    def analyze_episode(self, episode_id: str, trajectory: CanonicalTrajectory) -> EpisodeQuality:
        episode = EpisodeQuality(episode_id=episode_id)
        behavior = analyze_behavior(trajectory)
        episode.behavior_summary = behavior.activity.to_summary()
        episode.phases = {
            arm_name: list(timeline.segments)
            for arm_name, timeline in behavior.phases.per_arm.items()
        }
        episode.roles = {
            arm_name: list(timeline.segments)
            for arm_name, timeline in behavior.roles.per_arm.items()
        }
        episode.findings = evaluate_behavior_expectations(
            behavior.phases,
            behavior.roles,
            self.expectations,
        )
        for arm_name, arm in trajectory.arms.items():
            arm_quality = self._analyze_arm(
                f"{episode_id}:{arm_name}",
                arm,
                behavior.activity.per_arm[arm_name],
                behavior.phases.per_arm[arm_name],
                behavior.roles.per_arm[arm_name],
                trajectory.timestamps,
                trajectory.contract,
            )
            episode.per_arm[arm_name] = arm_quality

        episode.num_frames = max((arm_quality.num_frames for arm_quality in episode.per_arm.values()), default=0)

        active_scores = [arm_quality.overall_score for arm_quality in episode.per_arm.values() if arm_quality.overall_score is not None]
        if active_scores:
            episode.overall_score = float(np.mean(active_scores))
        episode.flags = list(
            dict.fromkeys(
                [flag for arm in episode.per_arm.values() for flag in arm.flags]
                + [finding.code for finding in episode.findings]
            )
        )
        return episode

    def build_report(self, dataset_path: str, episodes: list[EpisodeQuality]) -> QualityReport:
        report = QualityReport(dataset_path=dataset_path)
        report.per_episode = episodes
        report.num_episodes = len(episodes)
        scores = [episode.overall_score for episode in episodes if episode.overall_score is not None]
        if scores:
            report.overall_score = float(np.mean(scores))

        flagged: dict[str, list[str]] = defaultdict(list)
        per_arm_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for episode in episodes:
            for flag in episode.flags:
                flagged[flag].append(episode.episode_id)
            for arm_name, arm_quality in episode.per_arm.items():
                if arm_quality.overall_score is not None:
                    per_arm_metrics[arm_name]["overall_score"].append(arm_quality.overall_score)
                if arm_quality.trajectory_efficiency is not None:
                    per_arm_metrics[arm_name]["trajectory_efficiency"].append(arm_quality.trajectory_efficiency)
                if arm_quality.hesitation_fraction is not None:
                    per_arm_metrics[arm_name]["hesitation_fraction"].append(arm_quality.hesitation_fraction)

        report.flagged_episodes = dict(flagged)
        report.per_arm_summary = {
            arm_name: {
                metric_name: float(np.mean(values))
                for metric_name, values in metric_map.items()
                if values
            }
            for arm_name, metric_map in per_arm_metrics.items()
        }
        return report

    def _analyze_arm(
        self,
        episode_id: str,
        arm: CanonicalArmTrajectory,
        activity: ArmActivity,
        phases: ArmPhaseTimeline,
        roles: ArmRoleTimeline,
        timestamps: np.ndarray | None,
        contract: TrajectoryContract | None,
    ) -> EpisodeQuality:
        quality = EpisodeQuality(episode_id=episode_id)
        quality.num_frames = max(
            (
                len(signal)
                for signal in (
                    arm.action_position,
                    arm.action_orientation,
                    arm.joint_position,
                )
                if signal is not None
            ),
            default=0,
        )

        dt = _mean_timestep(timestamps)
        absolute_ee = (
            contract is not None
            and contract.action.space == "end_effector"
            and contract.action.mode == "absolute"
        )
        position = arm.action_position if absolute_ee else None
        orientation = arm.action_orientation if absolute_ee else None
        task_active = np.isin(roles.labels, ("actor", "support"))
        movement_phase = np.isin(
            phases.labels,
            ("unclassified_motion", "transport", "retreat", "manipulate"),
        )
        task_movement = task_active & movement_phase
        quality.active_fraction = (
            float(np.mean(task_active)) if len(task_active) else 0.0
        )
        quality.is_active = bool(np.any(task_active))
        quality.activity_reason = (
            f"task-active during {quality.active_fraction:.1%} of transitions"
            if quality.is_active
            else "no actor or support intervals; passive/idle motion excluded"
        )
        if not quality.is_active:
            quality.not_applicable["task_quality_metrics"] = (
                "arm has no actor or support intervals"
            )
            return quality

        if arm.joint_position is not None:
            quality.joint_smoothness = segmented_log_dimensionless_jerk(
                arm.joint_position, dt, task_movement
            )
            quality.joint_path_length = active_path_length(
                arm.joint_position,
                task_movement,
            )

        if position is not None:
            quality.translation_smoothness = segmented_log_dimensionless_jerk(
                position, dt, task_movement
            )
            quality.translation_path_length = active_path_length(
                position,
                task_movement,
            )
            quality.trajectory_efficiency = segmented_trajectory_efficiency(
                position,
                task_movement,
            )
            quality.hesitation_fraction = segmented_hesitation_fraction(
                position,
                dt,
                task_movement,
            )
        elif arm.action_position is not None:
            quality.not_applicable["translation_metrics"] = _ee_metric_reason(contract)

        quaternion_rotation = (
            contract is not None
            and contract.action.rotation is not None
            and contract.action.rotation.representation == "quaternion"
        )
        if orientation is not None and quaternion_rotation:
            (
                quality.rotation_path_length,
                quality.angular_speed_mean,
                quality.angular_acceleration_rms,
            ) = rotation_metrics(orientation, dt, task_movement)
        elif arm.action_orientation is not None:
            quality.not_applicable["rotation_metrics"] = _ee_metric_reason(contract)

        binary_gripper_command = (
            contract is not None
            and contract.gripper_action.representation == "binary_command"
        )
        if arm.action_gripper is not None and binary_gripper_command:
            quality.not_applicable["gripper_chatter_rate"] = (
                "binary commands encode intended state changes, not actuator chatter"
            )
        elif arm.action_gripper is not None and quality.num_frames > 1:
            quality.gripper_chatter_rate = masked_gripper_chatter_rate(
                arm.action_gripper,
                dt,
                task_active,
            )

        quality.smoothness = quality.translation_smoothness
        quality.cartesian_path_length = quality.translation_path_length

        quality.timestamp = timestamp_quality(timestamps) if timestamps is not None else None
        quality.overall_score, quality.flags = assess_learning_quality(
            quality.translation_smoothness,
            quality.trajectory_efficiency,
            quality.hesitation_fraction,
            quality.joint_smoothness,
        )
        if quality.overall_score is None:
            quality.not_applicable["overall_score"] = (
                "requires translation smoothness, joint smoothness, "
                "trajectory efficiency, and hesitation fraction"
            )
        return quality


def _mean_timestep(timestamps: np.ndarray | None) -> float:
    if timestamps is not None and len(timestamps) > 1:
        return float(np.mean(np.diff(timestamps)))
    return 0.1


def _ee_metric_reason(contract: TrajectoryContract | None) -> str:
    if contract is None:
        return "trajectory has no action contract"
    if contract.action.space != "end_effector":
        return f"action space is {contract.action.space}, not end_effector"
    if contract.action.mode != "absolute":
        return f"action mode is {contract.action.mode}; absolute pose is required"
    return "action rotation is not a supported quaternion representation"