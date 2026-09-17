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


@dataclass(frozen=True)
class GripperValueSpec:
	open_value: float
	closed_value: float
	larger_means: str | None = None


@dataclass(frozen=True)
class PositionSignalSpec:
	source: str
	slice_start: int
	slice_stop: int


@dataclass(frozen=True)
class RetreatBehaviorSpec:
	position_signal: PositionSignalSpec
	start_event: str = "gripper_close"
	window: int = 5
	displacement_threshold: float = 0.01


@dataclass(frozen=True)
class ReleaseBehaviorSpec:
	position_signal: PositionSignalSpec
	fully_open_signal: GripperSignalSpec
	gripper_value: GripperValueSpec
	open_event: str = "gripper_fully_open"
	retreat_event: str | None = None
	stationary_window: int = 1
	stationary_displacement_threshold: float = 0.01
	open_tolerance: float = 0.05
	fallback_delay_frames: int | None = None


@dataclass(frozen=True)
class EpisodeBehaviorContext:
	gripper_signal: np.ndarray
	events: dict[str, Event]
	position_signal: np.ndarray | None = None


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


def gripper_value_spec_from_config(
	cfg: dict[str, Any],
	*,
	source: str | None = None,
	side: str | None = None,
) -> GripperValueSpec:
	resolved_source = source or cfg.get("signal_source", "observation.state")
	gripper_root = cfg.get("gripper", {})
	if not isinstance(gripper_root, dict):
		gripper_root = {}

	gripper_cfg_key = "state" if resolved_source == "observation.state" else resolved_source
	gripper_cfg = gripper_root.get(gripper_cfg_key, {})
	if not isinstance(gripper_cfg, dict):
		gripper_cfg = {}

	if "open_value" not in gripper_cfg or "closed_value" not in gripper_cfg:
		raise ValueError(f"Missing gripper open/closed values for source {resolved_source!r}")

	del side
	return GripperValueSpec(
		open_value=float(gripper_cfg["open_value"]),
		closed_value=float(gripper_cfg["closed_value"]),
		larger_means=gripper_cfg.get("larger_means"),
	)


def _normalize_slice(raw_slice: Any, *, name: str) -> tuple[int, int]:
	if not isinstance(raw_slice, (list, tuple)) or len(raw_slice) != 2:
		raise ValueError(f"{name} must be a 2-item slice, got {raw_slice!r}")
	start = int(raw_slice[0])
	stop = int(raw_slice[1])
	if stop <= start:
		raise ValueError(f"{name} must have stop > start, got {raw_slice!r}")
	return start, stop


def _resolve_arm_slice(
	cfg: dict[str, Any],
	*,
	side: str | None,
	gripper_dim: int | None,
) -> tuple[int, int] | None:
	layout = cfg.get("action_layout", {})
	if not isinstance(layout, dict):
		return None

	if side is not None:
		arm_slice = layout.get(f"{side}_slice")
		if arm_slice is not None:
			return _normalize_slice(arm_slice, name=f"action_layout.{side}_slice")

	if gripper_dim is None:
		return None

	for candidate in ("left", "right"):
		arm_slice = layout.get(f"{candidate}_slice")
		if arm_slice is None:
			continue
		start, stop = _normalize_slice(arm_slice, name=f"action_layout.{candidate}_slice")
		if start <= gripper_dim < stop:
			return start, stop

	return None


def position_signal_spec_from_config(
	cfg: dict[str, Any],
	*,
	gripper_signal_spec: GripperSignalSpec | None = None,
	source: str | None = None,
	side: str | None = None,
	slice_start: int | None = None,
	slice_stop: int | None = None,
) -> PositionSignalSpec:
	retreat_cfg = cfg.get("retreat", {})
	if not isinstance(retreat_cfg, dict):
		retreat_cfg = {}

	resolved_source = source or retreat_cfg.get("source")
	if resolved_source is None:
		resolved_source = "action" if isinstance(cfg.get("action_layout"), dict) else "observation.state"

	if (slice_start is None) != (slice_stop is None):
		raise ValueError("slice_start and slice_stop must be provided together")

	if slice_start is not None and slice_stop is not None:
		return PositionSignalSpec(
			source=resolved_source,
			slice_start=int(slice_start),
			slice_stop=int(slice_stop),
		)

	configured_slice = retreat_cfg.get("position_slice")
	if configured_slice is not None:
		start, stop = _normalize_slice(configured_slice, name="retreat.position_slice")
		return PositionSignalSpec(
			source=resolved_source,
			slice_start=start,
			slice_stop=stop,
		)

	if resolved_source == "action":
		layout = cfg.get("action_layout", {})
		if not isinstance(layout, dict):
			raise ValueError("Action-based retreat detection requires action_layout in datasets.yaml")

		pos_start, pos_stop = _normalize_slice(
			layout.get("pos_slice", [0, 3]),
			name="action_layout.pos_slice",
		)
		arm_slice = _resolve_arm_slice(
			cfg,
			side=side or retreat_cfg.get("side"),
			gripper_dim=None if gripper_signal_spec is None else gripper_signal_spec.dim,
		)
		if arm_slice is None:
			return PositionSignalSpec(
				source=resolved_source,
				slice_start=pos_start,
				slice_stop=pos_stop,
			)

		return PositionSignalSpec(
			source=resolved_source,
			slice_start=arm_slice[0] + pos_start,
			slice_stop=arm_slice[0] + pos_stop,
		)

	observation_layout = cfg.get("observation_layout", {})
	if not isinstance(observation_layout, dict):
		observation_layout = {}

	start, stop = _normalize_slice(
		observation_layout.get("ee_pos_slice", [6, 9]),
		name="observation_layout.ee_pos_slice",
	)
	return PositionSignalSpec(
		source=resolved_source,
		slice_start=start,
		slice_stop=stop,
	)


def retreat_behavior_spec_from_config(
	cfg: dict[str, Any],
	*,
	gripper_signal_spec: GripperSignalSpec | None = None,
	position_source: str | None = None,
	side: str | None = None,
	position_slice_start: int | None = None,
	position_slice_stop: int | None = None,
	window: int | None = None,
	displacement_threshold: float | None = None,
	start_event: str = "gripper_close",
) -> RetreatBehaviorSpec:
	retreat_cfg = cfg.get("retreat", {})
	if not isinstance(retreat_cfg, dict):
		retreat_cfg = {}

	position_signal = position_signal_spec_from_config(
		cfg,
		gripper_signal_spec=gripper_signal_spec,
		source=position_source,
		side=side,
		slice_start=position_slice_start,
		slice_stop=position_slice_stop,
	)

	return RetreatBehaviorSpec(
		position_signal=position_signal,
		start_event=start_event,
		window=int(window if window is not None else retreat_cfg.get("window", 5)),
		displacement_threshold=float(
			displacement_threshold
			if displacement_threshold is not None
			else retreat_cfg.get("displacement_threshold", 0.01)
		),
	)


def release_behavior_spec_from_config(
	cfg: dict[str, Any],
	*,
	gripper_signal_spec: GripperSignalSpec,
	position_source: str | None = None,
	side: str | None = None,
	position_slice_start: int | None = None,
	position_slice_stop: int | None = None,
	stationary_window: int | None = None,
	stationary_displacement_threshold: float | None = None,
	open_tolerance: float | None = None,
	fully_open_source: str | None = None,
	open_event: str = "gripper_fully_open",
	retreat_event: str | None = None,
) -> ReleaseBehaviorSpec:
	release_cfg = cfg.get("release", {})
	if not isinstance(release_cfg, dict):
		release_cfg = {}

	tretreat_cfg = cfg.get("retreat", {})
	if not isinstance(tretreat_cfg, dict):
		tretreat_cfg = {}

	resolved_fully_open_source = fully_open_source or release_cfg.get("gripper_source", "observation.state")
	fully_open_signal = gripper_signal_spec_from_config(
		cfg,
		source=resolved_fully_open_source,
		side=side,
	)

	position_signal = position_signal_spec_from_config(
		cfg,
		gripper_signal_spec=gripper_signal_spec,
		source=position_source,
		side=side,
		slice_start=position_slice_start,
		slice_stop=position_slice_stop,
	)
	gripper_value = gripper_value_spec_from_config(
		cfg,
		source=fully_open_signal.source,
		side=side,
	)
	gripper_range = abs(gripper_value.closed_value - gripper_value.open_value)
	default_tolerance = max(gripper_range * 0.05, 1e-6)
	fallback_delay_frames = release_cfg.get("fallback_delay_frames")
	if fallback_delay_frames is None:
		gripper_root = cfg.get("gripper", {})
		if isinstance(gripper_root, dict):
			fallback_delay_frames = gripper_root.get("estimated_delay_frames")

	return ReleaseBehaviorSpec(
		position_signal=position_signal,
		fully_open_signal=fully_open_signal,
		gripper_value=gripper_value,
		open_event=open_event,
		retreat_event=retreat_event,
		stationary_window=int(
			stationary_window
			if stationary_window is not None
			else release_cfg.get("stationary_window", 1)
		),
		stationary_displacement_threshold=float(
			stationary_displacement_threshold
			if stationary_displacement_threshold is not None
			else release_cfg.get(
				"stationary_displacement_threshold",
				tretreat_cfg.get("displacement_threshold", 0.01),
			)
		),
		open_tolerance=float(
			open_tolerance
			if open_tolerance is not None
			else release_cfg.get("open_tolerance", default_tolerance)
		),
		fallback_delay_frames=None if fallback_delay_frames is None else int(fallback_delay_frames),
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


def position_signal(
	items: list[dict[str, Any]],
	spec: PositionSignalSpec,
) -> np.ndarray:
	if not items:
		raise ValueError("Cannot build a position signal from empty items.")

	values = np.stack([to_numpy(item[spec.source]) for item in items])
	if values.ndim == 1:
		raise ValueError(f"Position source {spec.source!r} must be vector-valued.")

	if spec.slice_stop > values.shape[1] or spec.slice_start < -values.shape[1]:
		raise IndexError(
			f"Position slice [{spec.slice_start}, {spec.slice_stop}) is invalid for "
			f"signal shape {values.shape}."
		)

	return values[:, spec.slice_start:spec.slice_stop].astype(float)


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


def find_retreat_start(
	position: np.ndarray,
	start_idx: int,
	window: int = 5,
	displacement_threshold: float = 0.01,
) -> Event | None:
	position = np.asarray(position, dtype=float)
	if position.ndim != 2:
		raise ValueError(f"position must be 2D, got shape {position.shape}")
	if len(position) == 0:
		return None
	if window < 1:
		raise ValueError("window must be >= 1")

	stop = len(position) - window
	for t in range(max(int(start_idx), 0), stop):
		displacement = float(np.linalg.norm(position[t + window] - position[t]))
		if displacement > displacement_threshold:
			return Event(
				name="retreat_start",
				local_index=t,
				score=displacement,
				method=f"displacement_over_threshold(window={window}, threshold={displacement_threshold})",
			)

	return None


def find_release_keyframe(
	gripper_signal: np.ndarray,
	position: np.ndarray,
	*,
	open_event: Event,
	retreat_event: Event,
	open_value: float,
	open_tolerance: float,
	stationary_window: int,
	stationary_displacement_threshold: float,
) -> Event | None:
	gripper_signal = np.asarray(gripper_signal, dtype=float)
	position = np.asarray(position, dtype=float)
	if position.ndim != 2:
		raise ValueError(f"position must be 2D, got shape {position.shape}")
	if stationary_window < 1:
		raise ValueError("stationary_window must be >= 1")

	start_idx = max(int(open_event.local_index), 0)
	end_idx = min(int(retreat_event.local_index), len(gripper_signal) - 1)
	if end_idx < start_idx:
		return None

	for t in range(end_idx, start_idx - 1, -1):
		if abs(float(gripper_signal[t]) - open_value) > open_tolerance:
			continue

		horizon = min(t + stationary_window, int(retreat_event.local_index), len(position) - 1)
		if horizon <= t:
			continue

		displacement = float(np.linalg.norm(position[horizon] - position[t]))
		if displacement <= stationary_displacement_threshold:
			return Event(
				name="release_keyframe",
				local_index=t,
				score=displacement,
				method=(
					"latest_open_stationary_before_retreat"
					f"(window={stationary_window}, threshold={stationary_displacement_threshold}, "
					f"open_value={open_value}, open_tolerance={open_tolerance})"
				),
			)

	return None


def find_gripper_fully_open(
	gripper_signal: np.ndarray,
	*,
	start_idx: int,
	open_value: float,
	open_tolerance: float,
	fallback_delay_frames: int | None = None,
) -> Event | None:
	gripper_signal = np.asarray(gripper_signal, dtype=float)
	for t in range(max(int(start_idx), 0), len(gripper_signal)):
		if abs(float(gripper_signal[t]) - open_value) <= open_tolerance:
			return Event(
				name="gripper_fully_open",
				local_index=t,
				score=float(abs(float(gripper_signal[t]) - open_value)),
				method=f"first_open_value_match(open_value={open_value}, open_tolerance={open_tolerance})",
			)
	if fallback_delay_frames is not None:
		fallback_index = min(max(int(start_idx), 0) + int(fallback_delay_frames), len(gripper_signal) - 1)
		return Event(
			name="gripper_fully_open",
			local_index=fallback_index,
			score=float(abs(float(gripper_signal[fallback_index]) - open_value)),
			method=f"estimated_delay_fallback(delay_frames={fallback_delay_frames})",
		)
	return None


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


def _build_behavior_context(
	items: list[dict[str, Any]],
	*,
	gripper_signal_spec: GripperSignalSpec,
	retreat_behavior_spec: RetreatBehaviorSpec | None,
	release_behavior_spec: ReleaseBehaviorSpec | None,
	smooth_window: int,
) -> EpisodeBehaviorContext:
	signal = _smooth_signal(
		gripper_signal(items, gripper_signal_spec),
		smooth_window,
	)

	close_event = detect_gripper_close(signal, direction=gripper_signal_spec.close_direction)
	events: dict[str, Event] = {
		close_event.name: close_event,
	}

	if close_event.local_index + 1 < len(signal):
		open_event_base = detect_gripper_open(
			signal[close_event.local_index:],
			direction=gripper_signal_spec.open_direction,
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
		open_event = detect_gripper_open(signal, direction=gripper_signal_spec.open_direction)
	events[open_event.name] = open_event

	position = None
	if retreat_behavior_spec is not None:
		anchor_event = events.get(retreat_behavior_spec.start_event)
		if anchor_event is None:
			raise ValueError(
				f"Retreat behavior depends on missing start event {retreat_behavior_spec.start_event!r}"
			)
		position = position_signal(items, retreat_behavior_spec.position_signal)
		retreat_event = find_retreat_start(
			position,
			start_idx=anchor_event.local_index,
			window=retreat_behavior_spec.window,
			displacement_threshold=retreat_behavior_spec.displacement_threshold,
		)
		if retreat_event is not None:
			events[retreat_event.name] = retreat_event

	if release_behavior_spec is not None:
		transition_open_event = events.get("gripper_open")
		if transition_open_event is None:
			raise ValueError("Release behavior depends on missing gripper_open event")
		fully_open_signal = _smooth_signal(
			gripper_signal(items, release_behavior_spec.fully_open_signal),
			smooth_window,
		)

		fully_open_event = find_gripper_fully_open(
			fully_open_signal,
			start_idx=transition_open_event.local_index,
			open_value=release_behavior_spec.gripper_value.open_value,
			open_tolerance=release_behavior_spec.open_tolerance,
			fallback_delay_frames=release_behavior_spec.fallback_delay_frames,
		)
		if fully_open_event is not None:
			events[fully_open_event.name] = fully_open_event

		open_event = events.get(release_behavior_spec.open_event)
		if open_event is None:
			raise ValueError(
				f"Release behavior depends on missing open event {release_behavior_spec.open_event!r}"
			)
		retreat_event = None
		if release_behavior_spec.retreat_event is not None:
			retreat_event = events.get(release_behavior_spec.retreat_event)
		release_position = position
		if release_position is None:
			release_position = position_signal(items, release_behavior_spec.position_signal)

		if retreat_event is None or retreat_event.local_index <= open_event.local_index:
			retreat_event = find_retreat_start(
				release_position,
				start_idx=open_event.local_index,
				window=release_behavior_spec.stationary_window,
				displacement_threshold=release_behavior_spec.stationary_displacement_threshold,
			)
			if retreat_event is None:
				retreat_event = Event(
					name="release_motion_boundary",
					local_index=len(signal) - 1,
					score=0.0,
					method="episode_end_boundary",
				)

		release_event = find_release_keyframe(
			gripper_signal=signal,
			position=release_position,
			open_event=open_event,
			retreat_event=retreat_event,
			open_value=release_behavior_spec.gripper_value.open_value,
			open_tolerance=release_behavior_spec.open_tolerance,
			stationary_window=release_behavior_spec.stationary_window,
			stationary_displacement_threshold=release_behavior_spec.stationary_displacement_threshold,
		)
		if release_event is not None:
			events[release_event.name] = release_event
		if position is None:
			position = release_position

	return EpisodeBehaviorContext(
		gripper_signal=signal,
		events=events,
		position_signal=position,
	)


def _resolve_keyframe_specs(
	context: EpisodeBehaviorContext,
	*,
	item_count: int,
	offset: int,
) -> dict[str, tuple[int, float, str]]:
	close_event = context.events["gripper_close"]
	open_event = context.events["gripper_open"]

	keyframe_specs: dict[str, tuple[int, float, str]] = {
		"episode_start": (0, 0.0, "episode_boundary"),
		"pre_grasp": (_clamp_index(close_event.local_index - offset, item_count), close_event.score, "offset_before_gripper_close"),
		"gripper_close": (close_event.local_index, close_event.score, close_event.method),
		"post_grasp": (_clamp_index(close_event.local_index + offset, item_count), close_event.score, "offset_after_gripper_close"),
		"pre_place": (_clamp_index(open_event.local_index - offset, item_count), open_event.score, "offset_before_gripper_open"),
		"gripper_open": (open_event.local_index, open_event.score, open_event.method),
		"post_place": (_clamp_index(open_event.local_index + offset, item_count), open_event.score, "offset_after_gripper_open"),
		"episode_end": (item_count - 1, 0.0, "episode_boundary"),
	}

	fully_open_event = context.events.get("gripper_fully_open")
	if fully_open_event is not None:
		keyframe_specs["gripper_fully_open"] = (
			fully_open_event.local_index,
			fully_open_event.score,
			fully_open_event.method,
		)

	retreat_event = context.events.get("retreat_start")
	if retreat_event is not None:
		keyframe_specs["pre_retreat"] = (
			_clamp_index(retreat_event.local_index - offset, item_count),
			retreat_event.score,
			"offset_before_retreat_start",
		)
		keyframe_specs["retreat_start"] = (
			retreat_event.local_index,
			retreat_event.score,
			retreat_event.method,
		)
		keyframe_specs["post_retreat"] = (
			_clamp_index(retreat_event.local_index + offset, item_count),
			retreat_event.score,
			"offset_after_retreat_start",
		)

	release_event = context.events.get("release_keyframe")
	if release_event is not None:
		keyframe_specs["release_keyframe"] = (
			release_event.local_index,
			release_event.score,
			release_event.method,
		)

	return keyframe_specs


def extract_keyframes_for_episode(
	ds: Any,
	dataset_name: str,
	episode_index: int,
	keyframe_types: list[str],
	cameras: list[str],
	out_dir: str | Path,
	gripper_signal_spec: GripperSignalSpec | None = None,
	retreat_behavior_spec: RetreatBehaviorSpec | None = None,
	release_behavior_spec: ReleaseBehaviorSpec | None = None,
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
	context = _build_behavior_context(
		items,
		gripper_signal_spec=resolved_spec,
		retreat_behavior_spec=retreat_behavior_spec,
		release_behavior_spec=release_behavior_spec,
		smooth_window=smooth_window,
	)
	keyframe_specs = _resolve_keyframe_specs(
		context,
		item_count=len(items),
		offset=offset,
	)
	supported_keyframe_types = {
		"episode_start",
		"pre_grasp",
		"gripper_close",
		"post_grasp",
		"gripper_fully_open",
		"release_keyframe",
		"pre_retreat",
		"retreat_start",
		"post_retreat",
		"pre_place",
		"gripper_open",
		"post_place",
		"episode_end",
	}

	keyframes: list[Keyframe] = []
	for keyframe_type in keyframe_types:
		if keyframe_type not in keyframe_specs:
			if keyframe_type not in supported_keyframe_types:
				available = ", ".join(sorted(supported_keyframe_types))
				raise ValueError(f"Unknown keyframe type '{keyframe_type}'. Available: {available}")
			available = ", ".join(sorted(keyframe_specs.keys()))
			raise ValueError(
				f"Keyframe type '{keyframe_type}' is supported but unavailable for this episode. "
				f"Available for this episode: {available}"
			)
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
				signal=context.gripper_signal,
			)
		)

	return keyframes


def save_keyframes_json(keyframes: list[Keyframe], path: str | Path) -> None:
	path = Path(path)
	path.parent.mkdir(parents=True, exist_ok=True)
	payload = [asdict(keyframe) for keyframe in keyframes]
	path.write_text(json.dumps(payload, indent=2))
