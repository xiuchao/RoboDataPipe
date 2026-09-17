from __future__ import annotations

import pandas as pd

from quality.models import QualityReport


def quality_report_to_dataframe(report: QualityReport) -> pd.DataFrame:
    rows: list[dict] = []
    for episode in report.per_episode:
        row = {
            "episode_id": episode.episode_id,
            "episode_index": _episode_index(episode.episode_id),
            "length": episode.num_frames,
            "overall_score": episode.overall_score,
            "num_flags": len(episode.flags),
            "flags": ",".join(episode.flags),
        }
        for arm_name, arm in episode.per_arm.items():
            prefix = f"{arm_name}_"
            row[f"{prefix}is_active"] = arm.is_active
            row[f"{prefix}smoothness"] = arm.smoothness
            row[f"{prefix}joint_path_length"] = arm.joint_path_length
            row[f"{prefix}cartesian_path_length"] = arm.cartesian_path_length
            row[f"{prefix}trajectory_efficiency"] = arm.trajectory_efficiency
            row[f"{prefix}hesitation_fraction"] = arm.hesitation_fraction
            row[f"{prefix}gripper_chatter_rate"] = arm.gripper_chatter_rate
            row[f"{prefix}jitter_ratio"] = (
                arm.timestamp.jitter_ratio if arm.timestamp is not None else None
            )
            row[f"{prefix}flags"] = ",".join(arm.flags)
        rows.append(row)
    return pd.DataFrame(rows)


def _episode_index(episode_id: str) -> int | None:
    if episode_id.startswith("episode_"):
        suffix = episode_id.removeprefix("episode_")
        if suffix.isdigit():
            return int(suffix)
    return None