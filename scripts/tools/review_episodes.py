import argparse
import numpy as np
import pandas as pd
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_io import DATASET_REGISTRY, build_episode_index, load_episode_items, load_lerobot_dataset
from quality import (
    analyze_loaded_dataset_quality,
    quality_report_to_dataframe,
    rank_suspicious_episodes,
)


__all__ = [
    "compute_action_state_metrics",
    "rank_suspicious_episodes",
    "make_review_commands",
    "auto_view_episodes",
]


def compute_action_state_metrics(ds: Any, cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Compute episode-level quality metrics using the DEM quality analyzer.

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

    report = analyze_loaded_dataset_quality(ds, cfg)
    metrics_df = quality_report_to_dataframe(report)

    nan_rows = []
    for episode_slice in build_episode_index(ds):
        items = load_episode_items(ds, episode_slice.episode_index)
        if not items:
            continue
        actions = np.stack([np.asarray(item["action"]) for item in items]).astype(np.float32)
        states = np.stack([np.asarray(item["observation.state"]) for item in items]).astype(np.float32)
        nan_rows.append(
            {
                "episode_index": int(episode_slice.episode_index),
                "has_nan_action": bool(np.isnan(actions).any()),
                "has_nan_state": bool(np.isnan(states).any()),
            }
        )
    if nan_rows:
        nan_df = pd.DataFrame(nan_rows)
        metrics_df = metrics_df.merge(nan_df, on="episode_index", how="left")

    return metrics_df

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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank and review episodes by trajectory quality.")
    parser.add_argument("--dataset", default="DSRFM_easy", help="Dataset nickname in datasets.yaml.")
    parser.add_argument("--registry", default=DATASET_REGISTRY, help="Path to datasets.yaml.")
    parser.add_argument("--skip-view", action="store_true", help="Write reports without launching episode viewers.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ds, cfg = load_lerobot_dataset(args.dataset, registry_path=args.registry)
    root = Path(cfg["root"])
    hf_dataset = getattr(ds, "hf_dataset", None)

    print({"dataset": args.dataset, "root": str(root), "repo_id": cfg["repo_id"]})
    print(f"episodes: {len(build_episode_index(ds))}")
    print(f"frames: {len(ds)}")
    if hf_dataset is not None:
        print(hf_dataset.column_names)

    # get episode-level metrics and save to CSV
    metrics_df = compute_action_state_metrics(ds, cfg)
    out_path = root / "quality_action_state.csv"
    metrics_df.to_csv(out_path, index=False)

    print(metrics_df.describe(include="all"))
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
        "worst_smoothness",
        "worst_chatter",
        "worst_hesitation",
        "worst_efficiency",
        "has_nan_action",
        "has_nan_state",
    ]
    print(ranked_df[cols].head(30))
    print("Saved:", ranked_path)


    # review df
    review_df = make_review_commands(ranked_df, n=30)
    review_df.to_csv(root / "review_commands.csv", index=False)

    print(review_df[
        ["episode_index", "suspicious_score", "length", "worst_smoothness", "viz_command"]
    ])

    if not args.skip_view:
        auto_view_episodes(
            ranked_df,
            root=root,
            n=20,
            start_rank=20,
        )


if __name__ == "__main__":
    main()
