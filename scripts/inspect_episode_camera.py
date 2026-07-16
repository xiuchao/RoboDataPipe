from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import DATASET_REGISTRY, load_lerobot_dataset
from robot_events.registry import resolve_episode_cameras


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset nickname in datasets.yaml, e.g. DEM_pickplace.",
    )
    parser.add_argument(
        "--registry",
        default=DATASET_REGISTRY,
        help="Path to datasets.yaml.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        required=True,
        help="Episode index to inspect.",
    )
    parser.add_argument(
        "--keyframe-type",
        default="gripper_close",
        help="Keyframe context for camera choice, e.g. gripper_close or episode_end.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ds, cfg = load_lerobot_dataset(args.dataset, registry_path=args.registry)
    cameras, motion = resolve_episode_cameras(
        ds,
        cfg,
        args.episode,
        cameras=None,
        keyframe_type=args.keyframe_type,
    )

    print(f"[dataset] {args.dataset}")
    print(f"[episode] {args.episode}")
    print(f"[keyframe_type] {args.keyframe_type}")
    print(f"[selected_cameras] {cameras}")
    if motion is not None:
        print(f"[moving_arm] {motion['moving_arm']}")
        print(f"[left_score] {motion['left_score']}")
        print(f"[right_score] {motion['right_score']}")