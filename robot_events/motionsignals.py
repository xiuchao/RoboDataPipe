import numpy as np
import pandas as pd
import subprocess
import time
from pathlib import Path
from typing import Any

from dataloader import DATASET_REGISTRY, load_lerobot_dataset
from keyframes import (
    GripperSignalSpec,
    build_episode_index,
    detect_gripper_close,
    detect_gripper_open,
    gripper_signal_spec_from_config,
    gripper_signal,
    item_frame_index,
    item_timestamp,
    resolve_gripper_signal_spec,
    scalar,
    to_numpy,
)


def _load_episode_items(ds: Any, global_indices: list[int]) -> list[dict[str, Any]]:
    hf_dataset = getattr(ds, "hf_dataset", None)
    if hf_dataset is None:
        return [ds[i] for i in global_indices]

    batch = hf_dataset.select(global_indices)
    return [
        {column: batch[column][i] for column in batch.column_names}
        for i in range(len(global_indices))
    ]


def compute_action_state_metrics(ds: Any) -> pd.DataFrame:
    """
    Compute episode-level action/state quality metrics.

    Expected dataset fields:
        episode_index
        frame_index
        action
        observation.state

    Returns:
        metrics_df: one row per episode
    """
    hf_dataset = getattr(ds, "hf_dataset", None)
    available_columns = hf_dataset.column_names if hf_dataset is not None else list(ds[0].keys())
    required_cols = ["episode_index", "action", "observation.state"]
    missing = [c for c in required_cols if c not in available_columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    rows = []

    for episode_slice in build_episode_index(ds):
        items = _load_episode_items(ds, episode_slice.global_indices)
        if not items:
            continue

        actions = np.stack([to_numpy(item["action"]) for item in items]).astype(np.float32)
        states = np.stack([to_numpy(item["observation.state"]) for item in items]).astype(np.float32)

        ee_action = actions[:, :6]
        gripper_cmd = actions[:, 6]

        ee_pos = states[:, 6:9]
        ee_z = states[:, 8]
        gripper_state = states[:, 12]
        gripper_cmd_state = states[:, 13]

        action_vel = np.diff(ee_action, axis=0)
        action_acc = np.diff(action_vel, axis=0)
        action_jerk = np.diff(action_acc, axis=0)

        ee_delta = np.diff(ee_pos, axis=0)

        rows.append({
            "episode_index": int(episode_slice.episode_index),
            "length": int(len(items)),

            # action quality
            "action_norm_mean": float(np.linalg.norm(ee_action, axis=1).mean()),
            "action_norm_max": float(np.linalg.norm(ee_action, axis=1).max()),
            "action_vel_mean": float(np.linalg.norm(action_vel, axis=1).mean()) if len(action_vel) else 0.0,
            "action_vel_max": float(np.linalg.norm(action_vel, axis=1).max()) if len(action_vel) else 0.0,
            "action_acc_mean": float(np.linalg.norm(action_acc, axis=1).mean()) if len(action_acc) else 0.0,
            "action_acc_max": float(np.linalg.norm(action_acc, axis=1).max()) if len(action_acc) else 0.0,
            "action_jerk_mean": float(np.linalg.norm(action_jerk, axis=1).mean()) if len(action_jerk) else 0.0,
            "action_jerk_max": float(np.linalg.norm(action_jerk, axis=1).max()) if len(action_jerk) else 0.0,

            # gripper behavior
            "gripper_toggles_action": float(np.abs(np.diff(gripper_cmd)).sum()) if len(gripper_cmd) > 1 else 0.0,
            "gripper_toggles_state": float(np.abs(np.diff(gripper_cmd_state)).sum()) if len(gripper_cmd_state) > 1 else 0.0,
            "gripper_state_start": float(gripper_state[0]),
            "gripper_state_end": float(gripper_state[-1]),
            "gripper_cmd_start": float(gripper_cmd[0]),
            "gripper_cmd_end": float(gripper_cmd[-1]),

            # EE trajectory
            "ee_x_start": float(ee_pos[0, 0]),
            "ee_y_start": float(ee_pos[0, 1]),
            "ee_z_start": float(ee_pos[0, 2]),
            "ee_x_end": float(ee_pos[-1, 0]),
            "ee_y_end": float(ee_pos[-1, 1]),
            "ee_z_end": float(ee_pos[-1, 2]),
            "ee_z_max": float(ee_z.max()),
            "ee_z_lift": float(ee_z.max() - ee_z[0]),
            "ee_path_len": float(np.linalg.norm(ee_delta, axis=1).sum()) if len(ee_delta) else 0.0,
            "ee_displacement": float(np.linalg.norm(ee_pos[-1] - ee_pos[0])),

            # sanity
            "has_nan_action": bool(np.isnan(actions).any()),
            "has_nan_state": bool(np.isnan(states).any()),
        })

    metrics_df = pd.DataFrame(rows)
    return metrics_df


def select_key_frames_before_pick_and_place(
    ds,
    episode_index,
    cfg: dict[str, Any] | None = None,
    gripper_signal_spec: GripperSignalSpec | None = None,
    signal_source="action",
    gripper_dim=-1,
    transition_threshold=0.5,
    frames_before=1,
    direction="increase",
    open_direction=None,
    side: str | None = None,
):
    del transition_threshold

    if frames_before < 1:
        raise ValueError("frames_before must be >= 1")

    episode_slices = build_episode_index(ds)
    episode_slice = next(
        (episode_slice for episode_slice in episode_slices if episode_slice.episode_index == episode_index),
        None,
    )
    if episode_slice is None:
        raise ValueError(f"episode {episode_index} not found")

    items = _load_episode_items(ds, episode_slice.global_indices)
    actions = np.stack([to_numpy(item["action"]) for item in items]).astype(np.float32)

    if len(actions) < 2:
        raise ValueError(f"episode {episode_index} is too short to select key frames")

    if cfg is not None:
        resolved_spec = gripper_signal_spec_from_config(
            cfg,
            source=signal_source,
            side=side,
            dim=gripper_dim,
            close_direction=direction,
            open_direction=open_direction,
        )
    else:
        resolved_spec = resolve_gripper_signal_spec(
            spec=gripper_signal_spec,
            signal_source=signal_source,
            gripper_dim=gripper_dim,
            close_direction=direction,
            open_direction=open_direction,
        )

    signal = gripper_signal(items, resolved_spec)
    pick_event = detect_gripper_close(signal, direction=resolved_spec.close_direction)

    if pick_event.local_index + 1 < len(signal):
        place_event_base = detect_gripper_open(
            signal[pick_event.local_index:],
            direction=resolved_spec.open_direction,
        )
        place_local_index = place_event_base.local_index + pick_event.local_index
    else:
        place_event_base = detect_gripper_open(signal, direction=resolved_spec.open_direction)
        place_local_index = place_event_base.local_index

    def build_key_frame(transition_idx, label):
        key_frame_idx = max(0, int(transition_idx) - (frames_before - 1))
        item = items[key_frame_idx]
        return {
            "label": label,
            "episode_index": int(episode_index),
            "frame_index": item_frame_index(item, key_frame_idx),
            "dataset_index": int(episode_slice.global_indices[key_frame_idx]),
            "timestamp": item_timestamp(item),
            "action": actions[key_frame_idx].tolist(),
            "observation_state": np.asarray(item["observation.state"], dtype=np.float32).tolist(),
        }

    return {
        "before_pick": build_key_frame(pick_event.local_index, "before_pick"),
        "before_place": build_key_frame(place_local_index, "before_place"),
    }


# ranks suspiciousness of episodes based on metrics. 
def _percentile_score(series, higher_is_worse=True):
    """
    Convert a numeric series to [0, 1] percentile score.
    1 means more suspicious.
    """
    s = pd.to_numeric(series, errors="coerce")
    ranks = s.rank(pct=True)

    if higher_is_worse:
        return ranks.fillna(0.0)
    else:
        return (1.0 - ranks).fillna(0.0)

def _two_sided_percentile_score(series):
    """
    High score for both unusually small and unusually large values.
    Median-ish values get lower scores.
    """
    s = pd.to_numeric(series, errors="coerce")
    pct = s.rank(pct=True).fillna(0.5)
    return (2.0 * np.abs(pct - 0.5)).clip(0.0, 1.0)

def rank_suspicious_episodes(metrics_df):
    """
    Add suspiciousness scores to episode-level metrics.

    Returns:
        ranked_df sorted by suspicious_score descending.
    """
    df = metrics_df.copy()

    df["score_length_extreme"] = _two_sided_percentile_score(df["length"])
    df["score_jerk"] = _percentile_score(df["action_jerk_max"], higher_is_worse=True)
    df["score_action_norm"] = _percentile_score(df["action_norm_max"], higher_is_worse=True)
    df["score_gripper_toggles"] = _percentile_score(df["gripper_toggles_action"], higher_is_worse=True)
    df["score_path_len"] = _percentile_score(df["ee_path_len"], higher_is_worse=True)

    df["score_nan"] = (
        df["has_nan_action"].astype(float) + df["has_nan_state"].astype(float)
    ).clip(0.0, 1.0)

    # Weighted suspicious score. Tune later after looking at a few episodes.
    df["suspicious_score"] = (
        0.25 * df["score_jerk"]
        + 0.20 * df["score_length_extreme"]
        + 0.20 * df["score_gripper_toggles"]
        + 0.15 * df["score_path_len"]
        + 0.10 * df["score_action_norm"]
        + 0.10 * df["score_nan"]
    )

    df = df.sort_values("suspicious_score", ascending=False).reset_index(drop=True)
    return df


# auto view episodes with lerobot-dataset-viz. 
def make_review_commands(
    ranked_df,
    n=30,
    episode_col="episode_index",
):
    review_df = ranked_df.head(n).copy()
    review_df["viz_command"] = review_df[episode_col].apply(
        lambda x: f"viz_ep {int(x)}"
    )
    return review_df

def auto_view_episodes(
    ranked_df,
    root,
    repo_id="ases200q2/UR5_RTDE_SpaceMouse_EE_pick_and_place_object_Easy_filtered",
    episode_col="episode_index",
    n=20,
    start_rank=0,
    sleep_after_launch=2.0,
):
    """
    Open suspicious episodes one by one with lerobot-dataset-viz.

    Usage:
        auto_view_episodes(ranked_df, ROOT, n=20)

    Controls:
        Press Enter: close current viewer and open next episode
        Type q + Enter: quit
    """
    root = Path(root)

    review_df = ranked_df.iloc[start_rank:start_rank + n].copy()

    for rank, row in review_df.iterrows():
        ep_idx = int(row[episode_col])

        print("\n" + "=" * 80)
        print(f"Rank: {rank}")
        print(f"Episode: {ep_idx}")

        for col in [
            "suspicious_score",
            "length",
            "action_jerk_max",
            "gripper_toggles_action",
            "ee_path_len",
            "action_norm_max",
        ]:
            if col in row:
                print(f"{col}: {row[col]}")

        cmd = [
            "lerobot-dataset-viz",
            "--repo-id", repo_id,
            "--root", str(root),
            "--mode", "local",
            "--episode-index", str(ep_idx),
        ]

        print("\nLaunching:")
        print(" ".join(cmd))

        proc = subprocess.Popen(cmd)
        time.sleep(sleep_after_launch)

        user_input = input("\nPress Enter for next episode, or type q to quit: ").strip()

        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        if user_input.lower() == "q":
            print("Stopped review.")
            break

# --------------------------------------------------------------
if __name__ == "__main__":

    DATASET_NAME = "DSRFM_easy"
    ds, cfg = load_lerobot_dataset(DATASET_NAME, registry_path=DATASET_REGISTRY)
    root = Path(cfg["root"])
    hf_dataset = getattr(ds, "hf_dataset", None)

    print({"dataset": DATASET_NAME, "root": str(root), "repo_id": cfg["repo_id"]})
    print(f"episodes: {len(build_episode_index(ds))}")
    print(f"frames: {len(ds)}")
    if hf_dataset is not None:
        print(hf_dataset.column_names)

    # get episode-level metrics and save to CSV
    metrics_df = compute_action_state_metrics(ds)
    out_path = root / "quality_action_state.csv"
    metrics_df.to_csv(out_path, index=False)

    print(metrics_df.describe())
    print("\nHigh jerk episodes:")
    print(metrics_df.sort_values("action_jerk_max", ascending=False).head(20))

    print("\nVery short episodes:")
    print(metrics_df.sort_values("length").head(20))

    print("\nMany gripper toggles:")
    print(metrics_df.sort_values("gripper_toggles_action", ascending=False).head(20))
    print("Saved:", out_path)


    # rank suspicious episodes based on metrics
    ranked_df = rank_suspicious_episodes(metrics_df)
    ranked_path = root / "quality_ranked_suspicious.csv"
    ranked_df.to_csv(ranked_path, index=False)

    print("\nMost suspicious episodes:")
    cols = [
        "episode_index",
        "suspicious_score",
        "length",
        "action_jerk_max",
        "gripper_toggles_action",
        "ee_path_len",
        "action_norm_max",
        "has_nan_action",
        "has_nan_state",
    ]
    print(ranked_df[cols].head(30))
    print("Saved:", ranked_path)


    # review df
    review_df = make_review_commands(ranked_df, n=30)
    review_df.to_csv(root / "review_commands.csv", index=False)

    print(review_df[
        ["episode_index", "suspicious_score", "length", "action_jerk_max", "viz_command"]
    ])

    auto_view_episodes(
    ranked_df,
    root=root,
    n=20,
    start_rank=20,
    )

