from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image


def scalar(value: Any) -> int | float | str:
	if hasattr(value, "item"):
		return value.item()
	return value


def to_numpy(value: Any) -> np.ndarray:
	if hasattr(value, "detach"):
		value = value.detach()
	if hasattr(value, "cpu"):
		value = value.cpu()
	if hasattr(value, "numpy"):
		return value.numpy()
	return np.asarray(value)


def item_frame_index(item: dict[str, Any], fallback: int) -> int:
	if "frame_index" in item:
		return int(scalar(item["frame_index"]))
	if "index" in item:
		return int(scalar(item["index"]))
	return fallback


def item_timestamp(item: dict[str, Any], fallback: float | None = None) -> float | None:
	if "timestamp" in item:
		return float(scalar(item["timestamp"]))
	return fallback


@dataclass(frozen=True)
class EpisodeSlice:
	episode_index: int
	global_indices: list[int]


def build_episode_index(dataset: Any) -> list[EpisodeSlice]:
	hf_dataset = getattr(dataset, "hf_dataset", None)
	if hf_dataset is not None and "episode_index" in hf_dataset.column_names:
		episode_values = [int(scalar(value)) for value in hf_dataset["episode_index"]]
		if not episode_values:
			return []

		episode_slices: list[EpisodeSlice] = []
		start = 0
		current_episode = episode_values[0]
		for i, episode_index in enumerate(episode_values[1:], start=1):
			if episode_index != current_episode:
				episode_slices.append(EpisodeSlice(current_episode, list(range(start, i))))
				start = i
				current_episode = episode_index
		episode_slices.append(EpisodeSlice(current_episode, list(range(start, len(episode_values)))))
		return episode_slices

	groups: dict[int, list[int]] = {}
	for i in range(len(dataset)):
		item = dataset[i]
		episode_index = int(scalar(item["episode_index"]))
		groups.setdefault(episode_index, []).append(i)
	return [EpisodeSlice(ep, idxs) for ep, idxs in sorted(groups.items())]


@dataclass(frozen=True)
class Event:
	name: str
	local_index: int
	score: float
	method: str
	signal_before: float | None = None
	signal_after: float | None = None


def _largest_change(signal: np.ndarray, direction: str) -> Event:
	if len(signal) < 2:
		raise ValueError("Need at least two frames to detect a transition.")

	diff = np.diff(signal)
	if direction == "decrease":
		local_index = int(np.argmin(diff) + 1)
		score = float(-np.min(diff))
	elif direction == "increase":
		local_index = int(np.argmax(diff) + 1)
		score = float(np.max(diff))
	else:
		raise ValueError("direction must be 'decrease' or 'increase'.")

	return Event(
		name="transition",
		local_index=local_index,
		score=score,
		method=f"largest_{direction}",
		signal_before=float(signal[local_index - 1]),
		signal_after=float(signal[local_index]),
	)

Direction = Literal["increase", "decrease"]


@dataclass(frozen=True)
class GripperSignalSpec:
    source: str
    dim: int
    close_direction: Direction
    open_direction: Direction


def _opposite_direction(direction: Direction) -> Direction:
	return "increase" if direction == "decrease" else "decrease"


def resolve_gripper_signal_spec(
	spec: GripperSignalSpec | None = None,
	*,
	signal_source: str = "observation.state",
	gripper_dim: int = -1,
	close_direction: Direction = "decrease",
	open_direction: Direction | None = None,
) -> GripperSignalSpec:
	if spec is not None:
		return spec

	return GripperSignalSpec(
		source=signal_source,
		dim=gripper_dim,
		close_direction=close_direction,
		open_direction=open_direction or _opposite_direction(close_direction),
	)


def gripper_signal_spec_from_config(
	cfg: dict[str, Any],
	*,
	source: str | None = None,
	side: str | None = None,
	dim: int | None = None,
	close_direction: Direction | None = None,
	open_direction: Direction | None = None,
) -> GripperSignalSpec:
	resolved_source = source or cfg.get("signal_source", "observation.state")
	source_cfg = cfg.get(resolved_source, {})
	if not isinstance(source_cfg, dict):
		source_cfg = {}

	gripper_root = cfg.get("gripper", {})
	if not isinstance(gripper_root, dict):
		gripper_root = {}

	resolved_side = side or cfg.get("gripper_side")

	gripper_cfg_key = "state" if resolved_source == "observation.state" else resolved_source
	gripper_cfg = gripper_root.get(gripper_cfg_key, {})
	if not isinstance(gripper_cfg, dict):
		gripper_cfg = {}

	resolved_dim = dim
	if resolved_dim is None and resolved_side is not None and f"{resolved_side}_index" in gripper_cfg:
		resolved_dim = int(gripper_cfg[f"{resolved_side}_index"])
	if resolved_dim is None and "index" in gripper_cfg:
		resolved_dim = int(gripper_cfg["index"])
	if resolved_dim is None and "right_index" in gripper_cfg:
		resolved_dim = int(gripper_cfg["right_index"])
	if resolved_dim is None and "left_index" in gripper_cfg:
		resolved_dim = int(gripper_cfg["left_index"])
	if resolved_dim is None:
		resolved_dim = int(cfg.get("gripper_dim", -1))

	resolved_close_direction = close_direction or source_cfg.get("close_direction") or cfg.get("direction", "decrease")
	resolved_open_direction = open_direction or source_cfg.get("open_direction")

	return resolve_gripper_signal_spec(
		signal_source=resolved_source,
		gripper_dim=resolved_dim,
		close_direction=resolved_close_direction,
		open_direction=resolved_open_direction,
	)


def gripper_signal(
    items: list[dict[str, Any]],
	spec: GripperSignalSpec | None = None,
	*,
	source: str = "observation.state",
	dim: int = -1,
) -> np.ndarray:
	if not items:
		raise ValueError(
			"Cannot build a gripper signal from empty items."
		)

	resolved_spec = spec or resolve_gripper_signal_spec(
		signal_source=source,
		gripper_dim=dim,
	)

	values = np.stack(
		[to_numpy(item[resolved_spec.source]) for item in items]
 	)

	if values.ndim == 1:
		return values.astype(float)

	if resolved_spec.dim >= values.shape[1] or resolved_spec.dim < -values.shape[1]:
		raise IndexError(
			f"Gripper dim {resolved_spec.dim} is invalid for "
			f"signal shape {values.shape}."
		)

	return values[:, resolved_spec.dim].astype(float)


def detect_gripper_close(
    signal: np.ndarray,
    direction: Direction,
) -> Event:
    signal = np.asarray(signal, dtype=float)
    event = _largest_change(signal, direction)

    return Event(
        "gripper_close",
        event.local_index,
        event.score,
        event.method,
        event.signal_before,
        event.signal_after,
    )


def detect_gripper_open(
    signal: np.ndarray,
    direction: Direction,
) -> Event:
    signal = np.asarray(signal, dtype=float)
    event = _largest_change(signal, direction)

    return Event(
        "gripper_open",
        event.local_index,
        event.score,
        event.method,
        event.signal_before,
        event.signal_after,
    )


def image_to_pil(image: Any) -> Image.Image:
	arr = to_numpy(image)
	if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
		arr = np.moveaxis(arr, 0, -1)
	if arr.dtype != np.uint8:
		if arr.max(initial=0) <= 1.0:
			arr = arr * 255.0
		arr = np.clip(arr, 0, 255).astype(np.uint8)
	if arr.ndim == 2:
		return Image.fromarray(arr)
	if arr.ndim == 3 and arr.shape[-1] == 1:
		return Image.fromarray(arr[..., 0])
	return Image.fromarray(arr)


def save_item_image(item: dict[str, Any], camera: str, path: Path) -> None:
	if camera not in item:
		raise KeyError(f"Camera field not found in item: {camera}")
	path.parent.mkdir(parents=True, exist_ok=True)
	image_to_pil(item[camera]).save(path)


@dataclass(frozen=True)
class Keyframe:
	keyframe_type: str
	episode_index: int
	local_index: int
	frame_index: int
	timestamp: float | None
	score: float
	method: str
	images: dict[str, str]
	signal_value: float | None = None


def _find_episode_slice(ds: Any, episode_index: int) -> EpisodeSlice:
	for episode_slice in build_episode_index(ds):
		if episode_slice.episode_index == episode_index:
			return episode_slice
	raise ValueError(f"episode {episode_index} not found")


def _smooth_signal(signal: np.ndarray, window: int) -> np.ndarray:
	signal = np.asarray(signal, dtype=float)
	if window <= 1 or len(signal) == 0:
		return signal
	kernel = np.ones(window, dtype=float) / float(window)
	return np.convolve(signal, kernel, mode="same")


def _clamp_index(index: int, length: int) -> int:
	return max(0, min(int(index), length - 1))


def _save_keyframe_images(
	ds: Any,
	global_index: int,
	cameras: list[str],
	out_dir: Path,
	keyframe_type: str,
	frame_index: int,
) -> dict[str, str]:
	item = ds[global_index]
	saved: dict[str, str] = {}
	for camera in cameras:
		safe_camera = camera.replace(".", "_").replace("/", "_")
		path = out_dir / f"{keyframe_type}_{safe_camera}_frame_{frame_index:06d}.jpg"
		save_item_image(item, camera, path)
		saved[camera] = str(path)
	return saved


def _make_keyframe(
	ds: Any,
	keyframe_type: str,
	episode_index: int,
	local_index: int,
	global_indices: list[int],
	items: list[dict[str, Any]],
	cameras: list[str],
	out_dir: Path,
	score: float,
	method: str,
	signal: np.ndarray | None,
) -> Keyframe:
	item = items[local_index]
	global_index = global_indices[local_index]
	frame_index = item_frame_index(item, local_index)
	timestamp = item_timestamp(item)
	images = _save_keyframe_images(ds, global_index, cameras, out_dir, keyframe_type, frame_index)
	signal_value = None if signal is None else float(signal[local_index])
	return Keyframe(
		keyframe_type=keyframe_type,
		episode_index=episode_index,
		local_index=local_index,
		frame_index=frame_index,
		timestamp=timestamp,
		score=float(score),
		method=method,
		images=images,
		signal_value=signal_value,
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


def extract_keyframes_for_episode(
	ds: Any,
	dataset_name: str,
	episode_index: int,
	keyframe_types: list[str],
	cameras: list[str],
	out_dir: str | Path,
	gripper_signal_spec: GripperSignalSpec | None = None,
	signal_source: str = "observation.state",
	gripper_dim: int = -1,
	direction: Direction = "decrease",
	open_direction: Direction | None = None,
	offset: int = 5,
	smooth_window: int = 1,
) -> list[Keyframe]:
	del dataset_name

	episode_slice = _find_episode_slice(ds, episode_index)
	items = _load_episode_items(ds, episode_slice.global_indices)
	if not items:
		raise ValueError(f"episode {episode_index} is empty")

	out_dir = Path(out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)
	resolved_spec = resolve_gripper_signal_spec(
		spec=gripper_signal_spec,
		signal_source=signal_source,
		gripper_dim=gripper_dim,
		close_direction=direction,
		open_direction=open_direction,
	)

	signal = _smooth_signal(
		gripper_signal(items, resolved_spec),
		smooth_window,
	)

	close_event = detect_gripper_close(signal, direction=resolved_spec.close_direction)
	if close_event.local_index + 1 < len(signal):
		open_event_base = detect_gripper_open(
			signal[close_event.local_index:],
			direction=resolved_spec.open_direction,
		)
		open_event = Event(
			name=open_event_base.name,
			local_index=open_event_base.local_index + close_event.local_index,
			score=open_event_base.score,
			method=open_event_base.method,
			signal_before=open_event_base.signal_before,
			signal_after=open_event_base.signal_after,
		)
	else:
		open_event = detect_gripper_open(signal, direction=resolved_spec.open_direction)

	keyframe_specs: dict[str, tuple[int, float, str]] = {
		"episode_start": (0, 0.0, "episode_boundary"),
		"pre_grasp": (_clamp_index(close_event.local_index - offset, len(items)), close_event.score, "offset_before_gripper_close"),
		"gripper_close": (close_event.local_index, close_event.score, close_event.method),
		"post_grasp": (_clamp_index(close_event.local_index + offset, len(items)), close_event.score, "offset_after_gripper_close"),
		"pre_place": (_clamp_index(open_event.local_index - offset, len(items)), open_event.score, "offset_before_gripper_open"),
		"gripper_open": (open_event.local_index, open_event.score, open_event.method),
		"post_place": (_clamp_index(open_event.local_index + offset, len(items)), open_event.score, "offset_after_gripper_open"),
		"episode_end": (len(items) - 1, 0.0, "episode_boundary"),
	}

	keyframes: list[Keyframe] = []
	for keyframe_type in keyframe_types:
		if keyframe_type not in keyframe_specs:
			available = ", ".join(sorted(keyframe_specs.keys()))
			raise ValueError(f"Unknown keyframe type '{keyframe_type}'. Available: {available}")
		local_index, score, method = keyframe_specs[keyframe_type]
		keyframes.append(
			_make_keyframe(
				ds=ds,
				keyframe_type=keyframe_type,
				episode_index=episode_index,
				local_index=local_index,
				global_indices=episode_slice.global_indices,
				items=items,
				cameras=cameras,
				out_dir=out_dir,
				score=score,
				method=method,
				signal=signal,
			)
		)

	return keyframes


def save_keyframes_json(keyframes: list[Keyframe], path: str | Path) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	payload = [asdict(keyframe) for keyframe in keyframes]
	path.write_text(json.dumps(payload, indent=2))
