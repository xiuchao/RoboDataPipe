from __future__ import annotations

import argparse
from pathlib import Path

from dataloader import load_lerobot_dataset
from keyframes import extract_keyframes_for_episode, save_keyframes_json


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


def normalize_output_root(out: str | Path) -> Path:
    out_path = Path(out)
    for prefix in ("output_keyframes_", "out_keyframes_"):
        if out_path.name.startswith(prefix):
            suffix = out_path.name.removeprefix(prefix)
            normalized_root = out_path.parent / "output_keyframes"
            return normalized_root / suffix if suffix else normalized_root
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
        default="/data/xiuchao/biArm/DEM/datasets.yaml",
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
        help="Camera key. Can be repeated.",
    )
    parser.add_argument(
        "--out",
        default="/data/xiuchao/biArm/DEM/output_keyframes",
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

    cameras = args.cameras
    if cameras is None:
        cameras = cfg.get("cameras")
    if cameras is None:
        default_camera = cfg.get("default_camera")
        if default_camera is None:
            raise ValueError(
                "No camera specified. Use --camera or set cameras/default_camera in datasets.yaml"
            )
        cameras = [default_camera]

    signal_source = args.signal_source or cfg.get("signal_source", "observation.state")
    gripper_dim = args.gripper_dim if args.gripper_dim is not None else cfg.get("gripper_dim", -1)
    direction = args.direction or cfg.get("direction", "decrease")

    output_root = normalize_output_root(args.out)
    out_dir = output_root / args.dataset / f"ep_{args.episode:03d}"


    keyframes = extract_keyframes_for_episode(
        ds=ds,
        dataset_name=args.dataset,
        episode_index=args.episode,
        keyframe_types=args.keyframes,
        cameras=cameras,
        out_dir=out_dir,
        signal_source=signal_source,
        gripper_dim=gripper_dim,
        direction=direction,
        offset=args.offset,
        smooth_window=args.smooth_window,
    )

    json_path = out_dir / f"keyframes.json"
    save_keyframes_json(keyframes, json_path)

    print(f"[done] saved {len(keyframes)} keyframes")
    print(f"[out]  {out_dir}")
    print(f"[json] {json_path}")

    for kf in keyframes:
        print(
            f"{kf.keyframe_type:16s} "
            f"ep={kf.episode_index} "
            f"frame={kf.frame_index} "
            f"time={kf.timestamp} "
            f"score={kf.score}"
        )

