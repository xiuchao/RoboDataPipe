from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import DATASET_REGISTRY, load_lerobot_dataset
from project_paths import KEYFRAME_OUTPUT_DIR
from robot_events.keyframes import extract_keyframes_for_episode, gripper_signal_spec_from_config, save_keyframes_json
from robot_events.registry import resolve_episode_cameras


DEFAULT_KEYFRAMES = [
    "episode_start",
    "pre_grasp",
    "gripper_close",
    "post_grasp",
    "pre_place",
    "gripper_open",
    "post_place",
    "episode_end",
]
DEFAULT_KEYFRAME_OUT = str(KEYFRAME_OUTPUT_DIR)


def resolve_camera_context_keyframe(keyframe_types: list[str]) -> str | None:
    preferred_order = [
        "gripper_close",
        "pre_grasp",
        "post_grasp",
        "gripper_open",
        "pre_place",
        "post_place",
        "episode_end",
        "episode_start",
    ]
    wanted = set(keyframe_types)
    for keyframe_type in preferred_order:
        if keyframe_type in wanted:
            return keyframe_type
    return keyframe_types[0] if keyframe_types else None

def normalize_output_root(out: str | Path) -> Path:
    out_path = Path(out)
    for prefix in ("output_keyframes_", "out_keyframes_"):
        if out_path.name.startswith(prefix):
            suffix = out_path.name.removeprefix(prefix)
            normalized_root = Path(DEFAULT_KEYFRAME_OUT)
            return normalized_root / suffix if suffix else normalized_root
    if out_path.name == "out_keyframes":
        return Path(DEFAULT_KEYFRAME_OUT)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset nickname in datasets.yaml, e.g. DSRFM_easy",
    )
    parser.add_argument(
        "--registry",
        default=DATASET_REGISTRY,
        help="Path to datasets.yaml",
    )
    parser.add_argument(
        "--episode",
        type=int,
        required=True,
        help="Episode index",
    )
    parser.add_argument(
        "--keyframes",
        nargs="+",
        default=DEFAULT_KEYFRAMES,
        help=f"Keyframe types. Default: {DEFAULT_KEYFRAMES}",
    )
    parser.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        default=None,
        help="Camera key. Can be repeated. Use --camera auto to force adapter-based camera selection.",
    )
    parser.add_argument(
        "--out",
        default=DEFAULT_KEYFRAME_OUT,
        help="Output directory",
    )
    parser.add_argument(
        "--signal-source",
        default=None,
        help="Signal source key, e.g. observation.state or action. Defaults to datasets.yaml if set.",
    )
    parser.add_argument(
        "--gripper-dim",
        type=int,
        default=None,
        help="Gripper dimension in signal vector. Defaults to datasets.yaml if set.",
    )
    parser.add_argument(
        "--direction",
        choices=["decrease", "increase"],
        default=None,
        help="For gripper_close: whether close is signal decrease or increase. Defaults to datasets.yaml if set.",
    )
    parser.add_argument(
        "--gripper-side",
        choices=["left", "right"],
        default=None,
        help="Optional gripper side for bimanual datasets. Defaults to the config-preferred index order.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=5,
        help="Frame offset for pre/post keyframes",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Optional moving average window for gripper signal",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ds, cfg = load_lerobot_dataset(
        args.dataset,
        registry_path=args.registry,
    )

    requested_cameras = args.cameras
    if requested_cameras is not None and "auto" in requested_cameras:
        if len(requested_cameras) != 1:
            raise ValueError("--camera auto cannot be combined with explicit camera names")
        requested_cameras = None

    camera_context_keyframe = resolve_camera_context_keyframe(args.keyframes)
    cameras, motion = resolve_episode_cameras(
        ds,
        cfg,
        args.episode,
        requested_cameras,
        keyframe_type=camera_context_keyframe,
    )

    gripper_signal_spec = gripper_signal_spec_from_config(
        cfg,
        source=args.signal_source,
        side=args.gripper_side,
        dim=args.gripper_dim,
        close_direction=args.direction,
    )

    output_root = normalize_output_root(args.out)
    out_dir = output_root / args.dataset / f"ep_{args.episode:03d}"


    keyframes = extract_keyframes_for_episode(
        ds=ds,
        dataset_name=args.dataset,
        episode_index=args.episode,
        keyframe_types=args.keyframes,
        cameras=cameras,
        out_dir=out_dir,
        gripper_signal_spec=gripper_signal_spec,
        offset=args.offset,
        smooth_window=args.smooth_window,
    )

    json_path = out_dir / f"keyframes.json"
    save_keyframes_json(keyframes, json_path)

    print(f"[done] saved {len(keyframes)} keyframes")
    print(f"[out]  {out_dir}")
    print(f"[json] {json_path}")
    if motion is not None:
        print(f"[moving_arm] {motion['moving_arm']}")
        print(f"[left_score] {motion['left_score']}")
        print(f"[right_score] {motion['right_score']}")
        print(f"[camera_context_keyframe] {camera_context_keyframe}")
        print(f"[selected_cameras] {cameras}")
    print(f"[gripper_signal_spec] {gripper_signal_spec}")

    for kf in keyframes:
        print(
            f"{kf.keyframe_type:16s} "
            f"ep={kf.episode_index} "
            f"frame={kf.frame_index} "
            f"time={kf.timestamp} "
            f"score={kf.score}"
        )

