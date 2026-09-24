from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from quality.results import QualityReport


LOWER_IS_BAD = (
    "translation_smoothness",
    "joint_smoothness",
    "trajectory_efficiency",
)
HIGHER_IS_BAD = ("hesitation_fraction",)


@dataclass(frozen=True)
class MetricThreshold:
    metric: str
    direction: str
    quantile: float
    value: float
    reference_count: int


@dataclass(frozen=True)
class ArmCalibration:
    arm_name: str
    episode_count: int
    thresholds: dict[str, MetricThreshold]


@dataclass(frozen=True)
class QualityCalibration:
    dataset_name: str
    lower_quantile: float
    upper_quantile: float
    reference_episode_ids: tuple[str, ...]
    per_arm: dict[str, ArmCalibration]

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "QualityCalibration":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        per_arm = {
            arm_name: ArmCalibration(
                arm_name=arm_payload["arm_name"],
                episode_count=int(arm_payload["episode_count"]),
                thresholds={
                    metric: MetricThreshold(**threshold)
                    for metric, threshold in arm_payload["thresholds"].items()
                },
            )
            for arm_name, arm_payload in payload["per_arm"].items()
        }
        return cls(
            dataset_name=payload["dataset_name"],
            lower_quantile=float(payload["lower_quantile"]),
            upper_quantile=float(payload["upper_quantile"]),
            reference_episode_ids=tuple(payload["reference_episode_ids"]),
            per_arm=per_arm,
        )


def calibrate_quality_thresholds(
    report: QualityReport,
    dataset_name: str,
    reference_episode_ids: set[str],
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    min_reference_count: int = 20,
) -> QualityCalibration:
    if not 0.0 < lower_quantile < upper_quantile < 1.0:
        raise ValueError("quantiles must satisfy 0 < lower < upper < 1")
    if min_reference_count < 2:
        raise ValueError("min_reference_count must be at least 2")
    available_ids = {episode.episode_id for episode in report.per_episode}
    unknown_ids = sorted(reference_episode_ids - available_ids)
    if unknown_ids:
        raise ValueError(f"reference episodes not found in report: {unknown_ids}")

    per_arm_values: dict[str, dict[str, list[float]]] = {}
    per_arm_episodes: dict[str, set[str]] = {}
    for episode in report.per_episode:
        if episode.episode_id not in reference_episode_ids or not episode.validity:
            continue
        for arm_name, arm in episode.per_arm.items():
            if arm.is_active is not True:
                continue
            per_arm_episodes.setdefault(arm_name, set()).add(episode.episode_id)
            metric_values = per_arm_values.setdefault(arm_name, {})
            for metric in LOWER_IS_BAD + HIGHER_IS_BAD:
                value = getattr(arm, metric)
                if value is not None and np.isfinite(value):
                    metric_values.setdefault(metric, []).append(float(value))

    per_arm: dict[str, ArmCalibration] = {}
    for arm_name, metric_values in per_arm_values.items():
        thresholds: dict[str, MetricThreshold] = {}
        for metric, values in metric_values.items():
            if len(values) < min_reference_count:
                continue
            lower_is_bad = metric in LOWER_IS_BAD
            quantile = lower_quantile if lower_is_bad else upper_quantile
            thresholds[metric] = MetricThreshold(
                metric=metric,
                direction="below" if lower_is_bad else "above",
                quantile=quantile,
                value=float(np.quantile(values, quantile)),
                reference_count=len(values),
            )
        if thresholds:
            per_arm[arm_name] = ArmCalibration(
                arm_name=arm_name,
                episode_count=len(per_arm_episodes[arm_name]),
                thresholds=thresholds,
            )
    if not per_arm:
        raise ValueError(
            "no arm has enough valid, active reference values; "
            f"need at least {min_reference_count} per metric"
        )
    return QualityCalibration(
        dataset_name=dataset_name,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        reference_episode_ids=tuple(sorted(reference_episode_ids)),
        per_arm=per_arm,
    )


def apply_quality_calibration(
    report: QualityReport,
    calibration: QualityCalibration,
) -> None:
    for episode in report.per_episode:
        for arm_name, arm in episode.per_arm.items():
            arm_calibration = calibration.per_arm.get(arm_name)
            if arm_calibration is None or arm.is_active is not True:
                continue
            for metric, threshold in arm_calibration.thresholds.items():
                value = getattr(arm, metric)
                if value is None or not np.isfinite(value):
                    continue
                outside = (
                    value < threshold.value
                    if threshold.direction == "below"
                    else value > threshold.value
                )
                if outside:
                    arm.flags.append(f"reference_{metric}_{threshold.direction}")
            arm.flags = list(dict.fromkeys(arm.flags))
        episode.flags = list(
            dict.fromkeys(
                [flag for arm in episode.per_arm.values() for flag in arm.flags]
                + [finding.code for finding in episode.findings]
            )
        )
    flagged: dict[str, list[str]] = {}
    for episode in report.per_episode:
        for flag in episode.flags:
            flagged.setdefault(flag, []).append(episode.episode_id)
    report.flagged_episodes = flagged