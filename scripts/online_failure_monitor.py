from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import DATASET_REGISTRY, load_registry
from robot_events.online_monitor import OnlineFailureMonitor, OnlineMonitorConfig, iter_jsonl_samples
from vlm.qwen_vl_config import PROMPT_MODES


def default_cameras_for_dataset(dataset: str, registry_path: str) -> list[str]:
    registry = load_registry(registry_path)
    dataset_cfg = registry.get(dataset, {})
    cameras_cfg = dataset_cfg.get("cameras", {}) if isinstance(dataset_cfg, dict) else {}
    robot_type = str(dataset_cfg.get("robot_type", "")) if isinstance(dataset_cfg, dict) else ""

    preferred_keys = ["global", "both", "top", "front", "left", "right", "default"]
    ordered: list[str] = []
    for key in preferred_keys:
        value = cameras_cfg.get(key) if isinstance(cameras_cfg, dict) else None
        if isinstance(value, list):
            for camera in value:
                if isinstance(camera, str) and camera not in ordered:
                    ordered.append(camera)
        elif isinstance(value, str) and value not in ordered:
            ordered.append(value)

    if robot_type in {"aibot2", "agibot2"}:
        for camera in [
            "observation.images.camera_top",
            "observation.images.camera_front",
            "observation.images.camera_left",
            "observation.images.camera_right",
        ]:
            if camera not in ordered:
                ordered.append(camera)

    if ordered:
        return ordered
    return ["observation.images.camera_top", "observation.images.camera_left"]


def resolve_cameras_for_samples(requested_cameras: list[str], samples: list[dict[str, Any]]) -> list[str]:
    if not samples:
        return requested_cameras
    first_sample = samples[0]
    available = {key for key in first_sample.keys() if isinstance(key, str) and key.startswith("observation.images.")}
    images = first_sample.get("images")
    if isinstance(images, dict):
        for key in images.keys():
            if isinstance(key, str) and key.startswith("observation.images."):
                available.add(key)
    resolved = [camera for camera in requested_cameras if camera in available]
    return resolved or sorted(available)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Dataset nickname in datasets.yaml, e.g. DEM_pickplace.")
    parser.add_argument("--input-jsonl", required=True, help="Path to a JSONL stream file. Each line must contain action/state arrays and image paths.")
    parser.add_argument("--registry", default=DATASET_REGISTRY, help="Path to datasets.yaml.")
    parser.add_argument(
        "--prompt-mode",
        choices=sorted(PROMPT_MODES.keys()),
        default="shelf_placement_after_release",
        help="Prompt template to use for online failure monitoring.",
    )
    parser.add_argument("--question", default=None, help="Optional explicit question override.")
    parser.add_argument("--camera", action="append", dest="cameras", default=None, help="Camera key to include. Can be repeated.")
    parser.add_argument("--model", default=None, help="Optional model override.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature. Use 0 for greedy decoding.")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto", help="Torch dtype for model loading.")
    parser.add_argument("--signal-source", default=None, help="Optional gripper-open detection source, e.g. action.")
    parser.add_argument("--release-gripper-source", default=None, help="Optional fully-open detection source, e.g. observation.state.")
    parser.add_argument("--release-open-tolerance", type=float, default=None, help="Optional open-value tolerance for fully-open detection.")
    parser.add_argument("--release-stationary-window", type=int, default=None, help="Reserved for future online release-state checks.")
    parser.add_argument("--release-stationary-displacement-threshold", type=float, default=None, help="Reserved for future online release-state checks.")
    parser.add_argument("--approach-source", default=None, help="Optional position source for right_arm_start_moving detection, e.g. action.")
    parser.add_argument("--approach-window", type=int, default=None, help="Window size in frames for right_arm_start_moving detection.")
    parser.add_argument("--approach-displacement-threshold", type=float, default=None, help="Minimum displacement needed to mark right_arm_start_moving.")
    parser.add_argument("--max-buffer-frames", type=int, default=64, help="Maximum number of recent frames to keep in memory.")
    parser.add_argument("--output-json", default=None, help="Optional path to save emitted alerts as a JSON array.")
    parser.add_argument("--output-txt", default=None, help="Optional path to save a concise human-readable event summary.")
    return parser.parse_args()


def build_monitor_config(args: argparse.Namespace) -> OnlineMonitorConfig:
    kwargs: dict[str, Any] = {
        "dataset_name": args.dataset,
        "registry_path": args.registry,
        "prompt_mode": args.prompt_mode,
        "question": args.question,
        "cameras": tuple(args.cameras or default_cameras_for_dataset(args.dataset, args.registry)),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "dtype_name": args.dtype,
        "signal_source": args.signal_source,
        "release_gripper_source": args.release_gripper_source,
        "release_open_tolerance": args.release_open_tolerance,
        "release_stationary_window": args.release_stationary_window,
        "release_stationary_displacement_threshold": args.release_stationary_displacement_threshold,
        "approach_source": args.approach_source,
        "approach_window": args.approach_window,
        "approach_displacement_threshold": args.approach_displacement_threshold,
        "max_buffer_frames": args.max_buffer_frames,
    }
    if args.model is not None:
        kwargs["model_name"] = args.model
    return OnlineMonitorConfig(**kwargs)


def format_event_summary(event_log: list[dict[str, Any]], alerts: list[dict[str, Any]]) -> str:
    lines = [
        f"[events] {len(event_log)}",
        f"[alerts] {len(alerts)}",
    ]
    for event in event_log:
        summary = f"[event] {event['event']} frame={event['frame_index']} timestamp={event.get('timestamp')}"
        if event.get("placement_status") is not None:
            summary += f" placement_status={event.get('placement_status')} confidence={event.get('confidence')}"
        lines.append(summary)

    for alert in alerts:
        lines.append(
            f"[alert] {alert['alert_type']} frame={alert['frame_index']} timestamp={alert.get('timestamp')} message={alert['message']}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    samples = list(iter_jsonl_samples(args.input_jsonl))
    args.cameras = resolve_cameras_for_samples(list(args.cameras or default_cameras_for_dataset(args.dataset, args.registry)), samples)
    monitor = OnlineFailureMonitor(build_monitor_config(args))
    event_log: list[dict[str, Any]] = []
    alerts: list[dict[str, Any]] = []

    for sample in samples:
        update = monitor.add_sample(sample)
        if update is None:
            continue

        for event in update.get("events", []):
            event_log.append(event)
            print(
                f"[event] {event['event']} frame={event['frame_index']} "
                f"timestamp={event.get('timestamp')}"
            )

        for alert in update.get("alerts", []):
            alerts.append(alert)
            print(f"[alert] {alert['alert_type']} frame={alert['frame_index']} message={alert['message']}")
            print(json.dumps(alert, indent=2))

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"events": event_log, "alerts": alerts}, indent=2))
        print(f"[json] {output_path}")

    if args.output_txt is not None:
        summary_path = Path(args.output_txt)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(format_event_summary(event_log, alerts))
        print(f"[txt] {summary_path}")


if __name__ == "__main__":
    main()
