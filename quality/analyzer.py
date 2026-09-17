from __future__ import annotations

from collections import defaultdict

import numpy as np

from quality.metrics import (
    gripper_chatter_rate,
    hesitation_fraction,
    log_dimensionless_jerk,
    path_length,
    quality_score,
    timestamp_quality,
    trajectory_efficiency,
)
from quality.models import CanonicalArmTrajectory, CanonicalTrajectory, EpisodeQuality, QualityReport


class DatasetQualityAnalyzer:
    def analyze_episode(self, episode_id: str, trajectory: CanonicalTrajectory) -> EpisodeQuality:
        episode = EpisodeQuality(episode_id=episode_id)
        for arm_name, arm in trajectory.arms.items():
            arm_quality = self._analyze_arm(f"{episode_id}:{arm_name}", arm, trajectory.timestamps)
            episode.per_arm[arm_name] = arm_quality

        episode.num_frames = max((arm_quality.num_frames for arm_quality in episode.per_arm.values()), default=0)

        active_scores = [arm_quality.overall_score for arm_quality in episode.per_arm.values() if arm_quality.overall_score is not None]
        if active_scores:
            episode.overall_score = float(np.mean(active_scores))
        episode.flags = list(dict.fromkeys(flag for arm in episode.per_arm.values() for flag in arm.flags))
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
        timestamps: np.ndarray | None,
    ) -> EpisodeQuality:
        motion_signal = arm.motion_signal
        quality = EpisodeQuality(episode_id=episode_id)
        if motion_signal is not None:
            quality.num_frames = len(motion_signal)
        elif arm.position is not None:
            quality.num_frames = len(arm.position)

        dt = _mean_timestep(timestamps)

        if motion_signal is not None:
            quality.smoothness = log_dimensionless_jerk(motion_signal, dt)
            quality.joint_path_length = path_length(motion_signal)

        if arm.position is not None:
            quality.cartesian_path_length = path_length(arm.position)
            quality.trajectory_efficiency = trajectory_efficiency(arm.position)
            quality.hesitation_fraction = hesitation_fraction(arm.position, dt)

        if arm.gripper is not None and quality.num_frames > 1:
            duration = max((quality.num_frames - 1) * dt, 1e-6)
            quality.gripper_chatter_rate = gripper_chatter_rate(arm.gripper, duration)

        quality.timestamp = timestamp_quality(timestamps)
        quality.overall_score, quality.flags = quality_score(
            quality.smoothness,
            quality.trajectory_efficiency,
            quality.hesitation_fraction,
            quality.gripper_chatter_rate,
            quality.timestamp.jitter_ratio if quality.timestamp else None,
        )
        quality.is_active, quality.activity_reason = _classify_activity(quality.joint_path_length)
        if not quality.is_active:
            quality.overall_score = None
        return quality


def _mean_timestep(timestamps: np.ndarray | None) -> float:
    if timestamps is not None and len(timestamps) > 1:
        return float(np.mean(np.diff(timestamps)))
    return 0.1


def _classify_activity(joint_path_length: float | None) -> tuple[bool, str]:
    if joint_path_length is None:
        return True, "no joint path measurement"
    if joint_path_length < 0.5:
        return False, f"joint path {joint_path_length:.3f} below activity threshold"
    return True, "joint motion exceeds activity threshold"