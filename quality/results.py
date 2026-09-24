from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from robehavior.phases import BehaviorPhase
from robehavior.roles import RoleSegment


@dataclass
class TimestampQuality:
    dt_mean: float
    dt_std: float
    jitter_ratio: float


@dataclass(frozen=True)
class QualityFinding:
    code: str
    severity: str
    arm: str
    start_frame: int
    stop_frame: int
    expected: str
    observed: str
    context: str
    message: str


@dataclass
class EpisodeQuality:
    episode_id: str
    validity: bool = True
    num_frames: int = 0
    behavior_summary: dict[str, dict[str, float]] = field(default_factory=dict)
    phases: dict[str, list[BehaviorPhase]] = field(default_factory=dict)
    roles: dict[str, list[RoleSegment]] = field(default_factory=dict)
    findings: list[QualityFinding] = field(default_factory=list)
    active_fraction: float | None = None
    smoothness: float | None = None
    translation_smoothness: float | None = None
    joint_smoothness: float | None = None
    joint_path_length: float | None = None
    cartesian_path_length: float | None = None
    translation_path_length: float | None = None
    rotation_path_length: float | None = None
    angular_speed_mean: float | None = None
    angular_acceleration_rms: float | None = None
    trajectory_efficiency: float | None = None
    hesitation_fraction: float | None = None
    gripper_chatter_rate: float | None = None
    timestamp: TimestampQuality | None = None
    overall_score: float | None = None
    is_active: bool | None = None
    activity_reason: str | None = None
    not_applicable: dict[str, str] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    per_arm: dict[str, "EpisodeQuality"] = field(default_factory=dict)

    def to_dict(self) -> dict:
        learning_quality = _compact_mapping({
            "translation_smoothness": self.translation_smoothness,
            "joint_smoothness": self.joint_smoothness,
            "trajectory_efficiency": self.trajectory_efficiency,
            "hesitation_fraction": self.hesitation_fraction,
            "gripper_chatter_rate": self.gripper_chatter_rate,
        })
        diagnostics = _compact_mapping({
            "joint_path_length": self.joint_path_length,
            "translation_path_length": self.translation_path_length,
            "rotation_path_length": self.rotation_path_length,
            "angular_speed_mean": self.angular_speed_mean,
            "angular_acceleration_rms": self.angular_acceleration_rms,
            "jitter_ratio": (
                self.timestamp.jitter_ratio if self.timestamp is not None else None
            ),
        })
        out = {
            "episode_id": self.episode_id,
            "validity": self.validity,
            "num_frames": self.num_frames,
            "behavior_summary": self.behavior_summary,
            "phases": {
                arm_name: [asdict(segment) for segment in segments]
                for arm_name, segments in self.phases.items()
            },
            "roles": {
                arm_name: [asdict(segment) for segment in segments]
                for arm_name, segments in self.roles.items()
            },
            "findings": [asdict(finding) for finding in self.findings],
            "active_fraction": self.active_fraction,
            "learning_quality": learning_quality,
            "diagnostics": diagnostics,
            "overall_score": self.overall_score,
            "is_active": self.is_active,
            "activity_reason": self.activity_reason,
            "not_applicable": self.not_applicable,
            "flags": self.flags,
        }
        if self.per_arm:
            out["per_arm"] = {name: value.to_dict() for name, value in self.per_arm.items()}
        return {
            key: value
            for key, value in out.items()
            if value is not None and value != {} and value != []
        }


def _compact_mapping(values: dict[str, float | None]) -> dict[str, float]:
    return {key: value for key, value in values.items() if value is not None}


@dataclass
class QualityReport:
    dataset_path: str
    num_episodes: int = 0
    overall_score: float = 0.0
    computed_at: str = ""
    per_episode: list[EpisodeQuality] = field(default_factory=list)
    per_arm_summary: dict[str, dict[str, float]] = field(default_factory=dict)
    flagged_episodes: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.computed_at:
            self.computed_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return _round_floats({
            "dataset_path": self.dataset_path,
            "num_episodes": self.num_episodes,
            "overall_score": self.overall_score,
            "computed_at": self.computed_at,
            "per_episode": [episode.to_dict() for episode in self.per_episode],
            "per_arm_summary": self.per_arm_summary,
            "flagged_episodes": self.flagged_episodes,
        })

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)


def _round_floats(value, digits: int = 2):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {key: _round_floats(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item, digits) for item in value]
    return value