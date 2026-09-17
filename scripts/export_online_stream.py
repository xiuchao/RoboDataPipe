from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import DATASET_REGISTRY, load_lerobot_dataset
from robot_events.keyframes import build_episode_index, image_to_pil, item_frame_index, item_timestamp, to_numpy
from robot_events.registry import resolve_episode_cameras


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset nickname in datasets.yaml, e.g. DEM_pickplace.")
    parser.add_argument(
        "--episode",
        type=int,
        action="append",
        dest="episodes",
        default=None,
        help="Optional episode index to export. Can be repeated. Defaults to all episodes.",
    )
    parser.add_argument("--registry", default=DATASET_REGISTRY, help="Path to datasets.yaml.")
    parser.add_argument("--camera", action="append", dest="cameras", default=None, help="Camera key to export. Can be repeated.")
    parser.add_argument("--output-dir", required=True, help="Directory for the exported JSONL and copied frame images.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing exported files.")
    return parser.parse_args()


def resolve_episode_indices(ds: Any, episodes: list[int] | None) -> list[int]:
    available = [int(episode_slice.episode_index) for episode_slice in build_episode_index(ds)]
    if episodes is None:
        return available
    requested = list(dict.fromkeys(int(episode_index) for episode_index in episodes))
    available_set = set(available)
    missing = [episode_index for episode_index in requested if episode_index not in available_set]
    if missing:
        raise ValueError(f"episode(s) not found: {missing}")
    return requested


def load_episode_items(ds: Any, episode_index: int) -> tuple[list[int], list[dict[str, Any]]]:
    episode_slice = next(
        (episode_slice for episode_slice in build_episode_index(ds) if episode_slice.episode_index == episode_index),
        None,
    )
    if episode_slice is None:
        raise ValueError(f"episode {episode_index} not found")

    hf_dataset = getattr(ds, "hf_dataset", None)
    if hf_dataset is None:
        items = [ds[i] for i in episode_slice.global_indices]
    else:
        batch = hf_dataset.select(episode_slice.global_indices)
        items = [
            {column: batch[column][i] for column in batch.column_names}
            for i in range(len(episode_slice.global_indices))
        ]
    return episode_slice.global_indices, items


def export_online_stream(
    *,
    dataset_name: str,
    episode_index: int,
    registry_path: str,
    cameras: list[str] | None,
    output_dir: str | Path,
    overwrite: bool,
) -> Path:
    ds, cfg = load_lerobot_dataset(dataset_name, registry_path=registry_path)
    resolved_cameras, _ = resolve_episode_cameras(ds, cfg, episode_index, cameras, keyframe_type="gripper_fully_open")
    global_indices, items = load_episode_items(ds, episode_index)

    output_dir = Path(output_dir)
    image_dir = output_dir / f"ep_{episode_index:03d}" / "images"
    jsonl_path = output_dir / f"ep_{episode_index:03d}.jsonl"
    if jsonl_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {jsonl_path}. Use --overwrite to replace it.")

    image_dir.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for local_index, item in enumerate(items):
            frame_index = item_frame_index(item, local_index)
            dataset_item = ds[global_indices[local_index]]
            images: dict[str, str] = {}
            for camera in resolved_cameras:
                safe_camera = camera.replace(".", "_").replace("/", "_")
                image_path = image_dir / f"frame_{frame_index:06d}_{safe_camera}.jpg"
                image_to_pil(dataset_item[camera]).save(image_path)
                images[camera] = str(image_path)

            payload = {
                "episode_index": int(episode_index),
                "frame_index": int(frame_index),
                "timestamp": item_timestamp(item),
                "action": to_numpy(item["action"]).astype(float).tolist(),
                "observation.state": to_numpy(item["observation.state"]).astype(float).tolist(),
                "images": images,
            }
            handle.write(json.dumps(payload) + "\n")

    return jsonl_path


def main() -> None:
    args = parse_args()
    ds, _ = load_lerobot_dataset(args.dataset, registry_path=args.registry)
    episode_indices = resolve_episode_indices(ds, args.episodes)
    for episode_index in episode_indices:
        jsonl_path = export_online_stream(
            dataset_name=args.dataset,
            episode_index=episode_index,
            registry_path=args.registry,
            cameras=args.cameras,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
        print(f"[jsonl] {jsonl_path}")


if __name__ == "__main__":
    main()
