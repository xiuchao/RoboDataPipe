from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CanonicalArmTrajectory:
    position: np.ndarray | None = None
    orientation: np.ndarray | None = None
    gripper: np.ndarray | None = None
    joint_position: np.ndarray | None = None
    state_gripper: np.ndarray | None = None

    @property
    def motion_signal(self) -> np.ndarray | None:
        return self.joint_position if self.joint_position is not None else self.position


@dataclass(frozen=True)
class CanonicalTrajectory:
    arms: dict[str, CanonicalArmTrajectory] = field(default_factory=dict)
    timestamps: np.ndarray | None = None


@dataclass
class TimestampQuality:
    dt_mean: float
    dt_std: float
    jitter_ratio: float


@dataclass
class EpisodeQuality:
    episode_id: str
    num_frames: int = 0
    smoothness: float | None = None
    joint_path_length: float | None = None
    cartesian_path_length: float | None = None
    trajectory_efficiency: float | None = None
    hesitation_fraction: float | None = None
    gripper_chatter_rate: float | None = None
    timestamp: TimestampQuality | None = None
    overall_score: float | None = None
    is_active: bool | None = None
    activity_reason: str | None = None
    flags: list[str] = field(default_factory=list)
    per_arm: dict[str, "EpisodeQuality"] = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {
            "episode_id": self.episode_id,
            "num_frames": self.num_frames,
            "smoothness": self.smoothness,
            "joint_path_length": self.joint_path_length,
            "cartesian_path_length": self.cartesian_path_length,
            "trajectory_efficiency": self.trajectory_efficiency,
            "hesitation_fraction": self.hesitation_fraction,
            "gripper_chatter_rate": self.gripper_chatter_rate,
            "overall_score": self.overall_score,
            "is_active": self.is_active,
            "activity_reason": self.activity_reason,
            "flags": self.flags,
        }
        if self.timestamp is not None:
            out["jitter_ratio"] = self.timestamp.jitter_ratio
        if self.per_arm:
            out["per_arm"] = {name: value.to_dict() for name, value in self.per_arm.items()}
        return out


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
        return {
            "dataset_path": self.dataset_path,
            "num_episodes": self.num_episodes,
            "overall_score": round(self.overall_score, 3),
            "computed_at": self.computed_at,
            "per_episode": [episode.to_dict() for episode in self.per_episode],
            "per_arm_summary": self.per_arm_summary,
            "flagged_episodes": self.flagged_episodes,
        }

    def to_json(self, path: str | Path) -> None:
        with open(path, "w") as handle:
            json.dump(self.to_dict(), handle, indent=2)