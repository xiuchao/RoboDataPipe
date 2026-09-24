from __future__ import annotations

import numpy as np
import pandas as pd

from quality.results import QualityReport


def quality_report_to_markdown(
    report: QualityReport,
    dataset_name: str,
    *,
    sample: int = 0,
    review_limit: int = 3,
) -> str:
    episodes = report.per_episode
    valid_count = sum(episode.validity for episode in episodes)
    scored = [episode for episode in episodes if episode.overall_score is not None]
    scope = (
        f"First {report.num_episodes} episodes (sample)"
        if sample > 0
        else f"All {report.num_episodes} episodes"
    )
    lines = [
        f"# {dataset_name} Dataset Quality Report",
        "",
        f"- Dataset: `{report.dataset_path}`",
        f"- Scope: {scope}",
        f"- Generated: `{report.computed_at}`",
        f"- Valid episodes: `{valid_count}/{report.num_episodes}`",
        f"- Scored episodes: `{len(scored)}/{report.num_episodes}`",
        f"- Dataset score: `{report.overall_score:.2f}/10`",
        "",
        "## Metric Summary",
        "",
        "| Metric | Count | Mean | Median | Worst episode | Best episode |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    metric_specs = (
        ("Overall score", "overall_score", False),
        ("Translation smoothness", "translation_smoothness", False),
        ("Joint smoothness", "joint_smoothness", False),
        ("Trajectory efficiency", "trajectory_efficiency", False),
        ("Hesitation fraction", "hesitation_fraction", True),
    )
    for label, metric_name, higher_is_worse in metric_specs:
        values = _report_metric_values(report, metric_name)
        if not values:
            continue
        ordered = sorted(values, reverse=higher_is_worse)
        worst_value, worst_episode = ordered[0]
        best_value, best_episode = ordered[-1]
        numeric_values = [value for value, _ in values]
        lines.append(
            f"| {label} | {len(values)} | {np.mean(numeric_values):.2f} | "
            f"{np.median(numeric_values):.2f} | `{worst_episode}` ({worst_value:.2f}) | "
            f"`{best_episode}` ({best_value:.2f}) |"
        )

    diagnostic_specs = (
        ("Joint path length", "joint_path_length"),
        ("Translation path length", "translation_path_length"),
        ("Rotation path length", "rotation_path_length"),
        ("Angular speed mean", "angular_speed_mean"),
        ("Angular acceleration RMS", "angular_acceleration_rms"),
        ("Timestamp jitter ratio", "jitter_ratio"),
    )
    active_arm_count = sum(
        arm.is_active is True
        for episode in episodes
        for arm in episode.per_arm.values()
    )
    diagnostic_tails: list[str] = []
    lines.extend([
        "",
        "## Diagnostic Distribution",
        "",
        "Diagnostics describe motion and sampling distributions and do not contribute to "
        "`overall_score`. Low or high values are context for review, not automatically "
        "better or worse.",
        "",
        "| Metric | Count | Missing | Median | P10 | P90 | Minimum | Maximum |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ])
    for label, metric_name in diagnostic_specs:
        values = _report_metric_values(report, metric_name)
        missing = active_arm_count - len(values)
        if not values:
            lines.append(f"| {label} | 0 | {missing} | N/A | N/A | N/A | N/A | N/A |")
            continue
        numeric_values = np.asarray([value for value, _ in values])
        p10, p90 = np.percentile(numeric_values, [10, 90])
        minimum, minimum_episode = min(values)
        maximum, maximum_episode = max(values)
        lines.append(
            f"| {label} | {len(values)} | {missing} | {np.median(numeric_values):.2f} | "
            f"{p10:.2f} | {p90:.2f} | `{minimum_episode}` ({minimum:.2f}) | "
            f"`{maximum_episode}` ({maximum:.2f}) |"
        )
        low_episodes = [
            episode_id
            for value, episode_id in values
            if value < p10 and not np.isclose(value, p10, atol=0.005)
        ]
        high_episodes = [
            episode_id
            for value, episode_id in values
            if value > p90 and not np.isclose(value, p90, atol=0.005)
        ]
        diagnostic_tails.append(
            f"- {label}: low `{', '.join(low_episodes) or 'none'}`; "
            f"high `{', '.join(high_episodes) or 'none'}`"
        )

    lines.extend([
        "",
        "### Diagnostic Distribution Tails",
        "",
        "Episodes outside P10-P90 are listed for comparison; they are not automatic "
        "quality failures.",
        "",
        *diagnostic_tails,
    ])

    lines.extend(["", "## Automatic Flags", ""])
    if report.flagged_episodes:
        for flag, episode_ids in sorted(report.flagged_episodes.items()):
            lines.append(f"- `{flag}`: {len(episode_ids)} ({', '.join(episode_ids)})")
    else:
        lines.append("No episodes crossed the configured absolute flag thresholds.")

    lines.extend(["", "## Manual Review Priority", ""])
    for rank, episode in enumerate(
        sorted(scored, key=lambda item: item.overall_score)[:review_limit],
        start=1,
    ):
        lines.append(f"{rank}. `{episode.episode_id}`: score `{episode.overall_score:.2f}`")
    if not scored:
        lines.append("No scored episodes are available for ranking.")

    if sample > 0:
        lines.extend([
            "",
            "> This report covers a leading sample, not the full dataset. It validates the",
            "> analysis workflow but should not be treated as the final dataset assessment.",
        ])
    return "\n".join(lines) + "\n"


def _report_metric_values(
    report: QualityReport,
    metric_name: str,
) -> list[tuple[float, str]]:
    if metric_name == "overall_score":
        return [
            (episode.overall_score, episode.episode_id)
            for episode in report.per_episode
            if episode.overall_score is not None
        ]
    return [
        (value, episode.episode_id)
        for episode in report.per_episode
        for arm in episode.per_arm.values()
        if arm.is_active
        if (value := _arm_metric_value(arm, metric_name)) is not None
    ]


def _arm_metric_value(arm, metric_name: str) -> float | None:
    if metric_name == "jitter_ratio":
        return arm.timestamp.jitter_ratio if arm.timestamp is not None else None
    return getattr(arm, metric_name)


def quality_report_to_dataframe(report: QualityReport) -> pd.DataFrame:
    rows: list[dict] = []
    for episode in report.per_episode:
        behavior_columns = {
            f"{arm_name}_{metric_name}": value
            for arm_name, metrics in episode.behavior_summary.items()
            for metric_name, value in metrics.items()
        }
        row = {
            "episode_id": episode.episode_id,
            "episode_index": _episode_index(episode.episode_id),
            "length": episode.num_frames,
            "overall_score": episode.overall_score,
            "num_flags": len(episode.flags),
            "flags": ",".join(episode.flags),
            "num_findings": len(episode.findings),
            "findings": ",".join(finding.code for finding in episode.findings),
            **behavior_columns,
        }
        for arm_name, arm in episode.per_arm.items():
            prefix = f"{arm_name}_"
            row[f"{prefix}is_active"] = arm.is_active
            row[f"{prefix}active_fraction"] = arm.active_fraction
            row[f"{prefix}smoothness"] = arm.smoothness
            row[f"{prefix}translation_smoothness"] = arm.translation_smoothness
            row[f"{prefix}joint_smoothness"] = arm.joint_smoothness
            row[f"{prefix}joint_path_length"] = arm.joint_path_length
            row[f"{prefix}cartesian_path_length"] = arm.cartesian_path_length
            row[f"{prefix}translation_path_length"] = arm.translation_path_length
            row[f"{prefix}rotation_path_length"] = arm.rotation_path_length
            row[f"{prefix}angular_speed_mean"] = arm.angular_speed_mean
            row[f"{prefix}angular_acceleration_rms"] = arm.angular_acceleration_rms
            row[f"{prefix}trajectory_efficiency"] = arm.trajectory_efficiency
            row[f"{prefix}hesitation_fraction"] = arm.hesitation_fraction
            row[f"{prefix}gripper_chatter_rate"] = arm.gripper_chatter_rate
            row[f"{prefix}jitter_ratio"] = (
                arm.timestamp.jitter_ratio if arm.timestamp is not None else None
            )
            row[f"{prefix}flags"] = ",".join(arm.flags)
        rows.append(row)
    return pd.DataFrame(rows)


def rank_suspicious_episodes(metrics: pd.DataFrame) -> pd.DataFrame:
    ranked = metrics.copy()
    ranked["score_length_extreme"] = _two_sided_percentile_score(ranked["length"])

    smoothness_columns = [column for column in ranked if column.endswith("smoothness")]
    if smoothness_columns:
        ranked["worst_smoothness"] = ranked[smoothness_columns].min(axis=1, skipna=True)
        ranked["score_jerk"] = _percentile_score(
            ranked["worst_smoothness"], higher_is_worse=False
        )
    else:
        ranked["score_jerk"] = 0.0

    efficiency_columns = [
        column for column in ranked if column.endswith("trajectory_efficiency")
    ]
    if efficiency_columns:
        ranked["worst_efficiency"] = ranked[efficiency_columns].min(axis=1, skipna=True)
        ranked["score_efficiency"] = _percentile_score(
            ranked["worst_efficiency"], higher_is_worse=False
        )
    else:
        ranked["score_efficiency"] = 0.0

    hesitation_columns = [
        column for column in ranked if column.endswith("hesitation_fraction")
    ]
    if hesitation_columns:
        ranked["worst_hesitation"] = ranked[hesitation_columns].max(axis=1, skipna=True)
        ranked["score_hesitation"] = _percentile_score(ranked["worst_hesitation"])
    else:
        ranked["score_hesitation"] = 0.0

    chatter_columns = [
        column for column in ranked if column.endswith("gripper_chatter_rate")
    ]
    if chatter_columns:
        ranked["worst_chatter"] = ranked[chatter_columns].max(axis=1, skipna=True)
        ranked["score_gripper_toggles"] = _percentile_score(ranked["worst_chatter"])
    else:
        ranked["score_gripper_toggles"] = 0.0

    path_columns = [
        column for column in ranked if column.endswith("cartesian_path_length")
    ]
    if path_columns:
        ranked["max_path_len"] = ranked[path_columns].max(axis=1, skipna=True)
        ranked["score_path_len"] = _percentile_score(ranked["max_path_len"])
    else:
        ranked["score_path_len"] = 0.0

    has_nan_action = ranked.get("has_nan_action", pd.Series(False, index=ranked.index))
    has_nan_state = ranked.get("has_nan_state", pd.Series(False, index=ranked.index))
    ranked["score_nan"] = (
        has_nan_action.astype(float) + has_nan_state.astype(float)
    ).clip(0.0, 1.0)

    ranked["suspicious_score"] = (
        0.25 * ranked["score_jerk"]
        + 0.20 * ranked["score_length_extreme"]
        + 0.20 * ranked["score_gripper_toggles"]
        + 0.15 * ranked["score_hesitation"]
        + 0.15 * ranked["score_path_len"]
        + 0.10 * ranked["score_efficiency"]
        + 0.10 * ranked["score_nan"]
    )
    return ranked.sort_values("suspicious_score", ascending=False).reset_index(drop=True)


def _episode_index(episode_id: str) -> int | None:
    if episode_id.startswith("episode_"):
        suffix = episode_id.removeprefix("episode_")
        if suffix.isdigit():
            return int(suffix)
    return None


def _percentile_score(series: pd.Series, *, higher_is_worse: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    ranks = values.rank(pct=True)
    return ranks.fillna(0.0) if higher_is_worse else (1.0 - ranks).fillna(0.0)


def _two_sided_percentile_score(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    percentiles = values.rank(pct=True).fillna(0.5)
    return (2.0 * np.abs(percentiles - 0.5)).clip(0.0, 1.0)