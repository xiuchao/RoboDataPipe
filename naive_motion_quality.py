import numpy as np
from pathlib import Path
from io import BytesIO
import json
import re
import pandas as pd
import subprocess
import time
from functools import lru_cache

import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import snapshot_download


def load_lerobot_parquet(root):
    """
    Load a local LeRobot-style dataset from:
        root/
          data/chunk-xxx/file-xxx.parquet
          meta/info.json

    Returns:
        df: pandas DataFrame with all frames
        info: dict, dataset metadata if available
    """
    root = Path(root)

    info_path = root / "meta" / "info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())
    else:
        info = {}

    parquet_files = sorted((root / "data").glob("chunk-*/*.parquet"))

    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")

    tables = []
    for p in parquet_files:
        table = pq.read_table(p)
        tables.append(table.to_pandas())

    df = pd.concat(tables, ignore_index=True)

    # Keep frame order stable
    sort_cols = []
    if "episode_index" in df.columns:
        sort_cols.append("episode_index")
    if "frame_index" in df.columns:
        sort_cols.append("frame_index")
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df, info

def compute_action_state_metrics(df):
    """
    Compute episode-level action/state quality metrics.

    Expected columns:
        episode_index
        frame_index
        action: shape [7]
        observation.state: shape [14]

    Returns:
        metrics_df: one row per episode
    """
    required_cols = ["episode_index", "action", "observation.state"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    rows = []

    for ep_idx, ep in df.groupby("episode_index"):
        if "frame_index" in ep.columns:
            ep = ep.sort_values("frame_index")

        actions = np.stack(ep["action"].to_numpy()).astype(np.float32)
        states = np.stack(ep["observation.state"].to_numpy()).astype(np.float32)

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
            "episode_index": int(ep_idx),
            "length": int(len(ep)),

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
    gripper_dim=-1,
    transition_threshold=0.5,
    frames_before=1,
):
    ex = ds.filter(lambda x: x["episode_index"] == episode_index)
    actions = np.asarray(ex["action"], dtype=np.float32)

    if len(actions) < 2:
        raise ValueError(f"episode {episode_index} is too short to select key frames")

    if frames_before < 1:
        raise ValueError("frames_before must be >= 1")

    gripper = actions[:, gripper_dim]
    transitions = np.diff(gripper)

    pick_candidates = np.flatnonzero(transitions > transition_threshold)
    if len(pick_candidates) == 0:
        raise ValueError(f"episode {episode_index} has no pick transition")

    place_candidates = np.flatnonzero(transitions < -transition_threshold)
    place_candidates = place_candidates[place_candidates > pick_candidates[0]]
    if len(place_candidates) == 0:
        raise ValueError(f"episode {episode_index} has no place transition after pick")

    def build_key_frame(transition_idx, label):
        key_frame_idx = max(0, int(transition_idx) - (frames_before - 1))
        return {
            "label": label,
            "episode_index": int(episode_index),
            "frame_index": int(ex["frame_index"][key_frame_idx]),
            "dataset_index": int(ex["index"][key_frame_idx]),
            "timestamp": float(ex["timestamp"][key_frame_idx]),
            "action": actions[key_frame_idx].tolist(),
            "observation_state": np.asarray(
                ex["observation.state"][key_frame_idx], dtype=np.float32
            ).tolist(),
        }

    return {
        "before_pick": build_key_frame(pick_candidates[0], "before_pick"),
        "before_place": build_key_frame(place_candidates[0], "before_place"),
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

# --------------------------------------------------------------------------
if __name__ == "__main__":

    ROOT = Path("/data/xiuchao/biArm/DEM/ur5_easy_filtered")
    # load dataset and print some info
    df, info = load_lerobot_parquet(ROOT)
    print(info)
    print(df.shape)
    print(df.columns.tolist())
    print(df.iloc[0]["action"])
    print(df.iloc[0]["observation.state"])
    breakpoint()

    # get episode-level metrics and save to CSV
    metrics_df = compute_action_state_metrics(df)
    out_path = ROOT / "quality_action_state.csv"
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
    ranked_path = ROOT / "quality_ranked_suspicious.csv"
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
    review_df.to_csv(ROOT / "review_commands.csv", index=False)

    print(review_df[
        ["episode_index", "suspicious_score", "length", "action_jerk_max", "viz_command"]
    ])

    auto_view_episodes(
    ranked_df,
    root=ROOT,
    n=20,
    start_rank=20,
    )

