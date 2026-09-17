from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from dataloader import DATASET_REGISTRY, load_registry
from robot_events.keyframes import (
    Event,
    detect_gripper_open,
    find_retreat_start,
    find_gripper_fully_open,
    gripper_signal,
    gripper_value_spec_from_config,
    gripper_signal_spec_from_config,
    item_frame_index,
    item_timestamp,
    position_signal,
    position_signal_spec_from_config,
    release_behavior_spec_from_config,
)
from vlm.qwen_vl_config import DEFAULT_MODEL
from vlm.qwen_vl_qa import answer_question_about_image_entries, load_qwen_model


_VLM_CACHE_LOCK = threading.Lock()
_VLM_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


@dataclass(frozen=True)
class OnlineAlertRule:
    alert_statuses: tuple[str, ...] = ("still_held", "dropped_outside", "missed_compartment", "uncertain")


@dataclass(frozen=True)
class OnlineMonitorConfig:
    dataset_name: str
    registry_path: str = DATASET_REGISTRY
    prompt_mode: str = "shelf_placement_after_release"
    question: str | None = None
    cameras: tuple[str, ...] = (
        "observation.images.camera_top",
        "observation.images.camera_left",
    )
    model_name: str = DEFAULT_MODEL
    max_new_tokens: int = 256
    temperature: float = 0.0
    dtype_name: str = "auto"
    shot_mode: str = "zeroshot"
    signal_source: str | None = None
    release_gripper_source: str | None = None
    release_open_tolerance: float | None = None
    release_stationary_window: int | None = None
    release_stationary_displacement_threshold: float | None = None
    approach_side: str = "right"
    approach_source: str = "action"
    approach_window: int | None = None
    approach_displacement_threshold: float | None = None
    retreat_source: str | None = None
    retreat_window: int | None = None
    retreat_displacement_threshold: float | None = None
    max_buffer_frames: int = 64
    alert_rule: OnlineAlertRule = OnlineAlertRule()


@dataclass(frozen=True)
class LoggedEvent:
    event: str
    episode_index: int | None
    frame_index: int
    timestamp: float | None
    score: float
    method: str


class OnlineFailureMonitor:
    def __init__(self, config: OnlineMonitorConfig):
        self.config = config
        registry = load_registry(config.registry_path)
        if config.dataset_name not in registry:
            available = ", ".join(sorted(registry.keys()))
            raise KeyError(f"Unknown dataset '{config.dataset_name}'. Available: {available}")

        self.dataset_cfg = registry[config.dataset_name]
        self.robot_type = str(self.dataset_cfg.get("robot_type", "unknown"))
        self.gripper_signal_spec = gripper_signal_spec_from_config(
            self.dataset_cfg,
            source=config.signal_source,
        )
        open_signal_source = config.signal_source if config.signal_source is not None else "action"
        self.start_open_signal_spec = gripper_signal_spec_from_config(
            self.dataset_cfg,
            source=open_signal_source,
        )
        self.release_behavior_spec = release_behavior_spec_from_config(
            self.dataset_cfg,
            gripper_signal_spec=self.gripper_signal_spec,
            fully_open_source=config.release_gripper_source,
            stationary_window=config.release_stationary_window,
            stationary_displacement_threshold=config.release_stationary_displacement_threshold,
            open_tolerance=config.release_open_tolerance,
        )
        self.start_open_gripper_value = gripper_value_spec_from_config(
            self.dataset_cfg,
            source=open_signal_source,
            side=config.approach_side,
        )
        self.initial_state_signal_spec = self.release_behavior_spec.fully_open_signal
        self.initial_state_gripper_value = self.release_behavior_spec.gripper_value
        retreat_cfg = self.dataset_cfg.get("retreat", {})
        if not isinstance(retreat_cfg, dict):
            retreat_cfg = {}

        self.approach_position_spec = position_signal_spec_from_config(
            self.dataset_cfg,
            gripper_signal_spec=self.gripper_signal_spec,
            source=config.approach_source,
            side=config.approach_side,
        )
        self.left_arm_position_spec = None
        if self.robot_type in {"aibot2", "agibot2"}:
            self.left_arm_position_spec = position_signal_spec_from_config(
                self.dataset_cfg,
                gripper_signal_spec=self.gripper_signal_spec,
                source=config.approach_source,
                side="left",
            )
        self.approach_window = int(config.approach_window if config.approach_window is not None else retreat_cfg.get("window", 5))
        self.approach_displacement_threshold = float(
            config.approach_displacement_threshold
            if config.approach_displacement_threshold is not None
            else retreat_cfg.get("displacement_threshold", 0.01)
        )

        self.retreat_position_spec = position_signal_spec_from_config(
            self.dataset_cfg,
            gripper_signal_spec=self.gripper_signal_spec,
            source=config.retreat_source,
            side=config.approach_side,
        )
        self.retreat_window = int(config.retreat_window if config.retreat_window is not None else retreat_cfg.get("window", 5))
        self.retreat_displacement_threshold = float(
            config.retreat_displacement_threshold
            if config.retreat_displacement_threshold is not None
            else retreat_cfg.get("displacement_threshold", 0.01)
        )

        self.processor: Any | None = None
        self.model: Any | None = None
        self.vlm_disabled = False
        self.vlm_error: str | None = None
        self.buffer: deque[dict[str, Any]] = deque(maxlen=config.max_buffer_frames)
        self.emitted_event_keys: set[tuple[str, int | None, int]] = set()
        self.approach_start_event: LoggedEvent | None = None
        self.approach_stop_event: LoggedEvent | None = None
        self.open_event: LoggedEvent | None = None
        self.fully_open_event: LoggedEvent | None = None
        self.retreat_event: LoggedEvent | None = None
        self.status_event_key: tuple[str, int | None, int] | None = None
        self.initial_state_event: LoggedEvent | None = None
        self.left_arm_still_event: LoggedEvent | None = None

    def add_sample(self, sample: dict[str, Any]) -> dict[str, Any] | None:
        normalized = normalize_online_sample(sample)
        self.buffer.append(normalized)
        return self._collect_updates()

    def _collect_updates(self) -> dict[str, Any] | None:
        items = list(self.buffer)
        updates = {"events": [], "alerts": []}

        left_arm_still_record = self._maybe_capture_left_arm_still(items)
        if left_arm_still_record is not None:
            updates["events"].append(left_arm_still_record)

        initial_state_record = self._maybe_capture_initial_state(items)
        if initial_state_record is not None:
            updates["events"].append(initial_state_record)

        if len(self.buffer) < 2:
            return updates if updates["events"] or updates["alerts"] else None

        open_record = self._maybe_capture_gripper_open(items)
        if open_record is not None:
            updates["events"].append(open_record)

        approach_record = self._maybe_capture_approach_start(items)
        if approach_record is not None:
            updates["events"].append(approach_record)

        approach_stop_record = self._maybe_capture_approach_stop(items)
        if approach_stop_record is not None:
            updates["events"].append(approach_stop_record)

        if self.open_event is None:
            return updates if updates["events"] or updates["alerts"] else None

        fully_open_record = self._maybe_capture_gripper_fully_open(items)
        if fully_open_record is not None:
            updates["events"].append(fully_open_record)

        if self.fully_open_event is None:
            return updates if updates["events"] or updates["alerts"] else None

        retreat_record = self._maybe_capture_retreat_start(items)
        if retreat_record is not None:
            updates["events"].append(retreat_record)

        status_update = self._maybe_capture_object_status(items)
        if status_update is not None:
            updates["events"].append(status_update)
            if status_update.get("placement_status") in self.config.alert_rule.alert_statuses:
                updates["alerts"].append(
                    {
                        "alert": True,
                        "alert_type": status_update["placement_status"],
                        **status_update,
                    }
                )

        return updates if updates["events"] or updates["alerts"] else None

    def _maybe_capture_approach_start(self, items: list[dict[str, Any]]) -> dict[str, Any] | None:
        if self.approach_start_event is not None:
            return None
        approach_signal = position_signal(items, self.approach_position_spec)
        approach_event = self._detect_motion_start(approach_signal)
        if approach_event is None:
            return None
        self.approach_start_event = self._freeze_event(items, approach_event)
        return self._event_dict(self.approach_start_event)

    def _detect_motion_start(self, position: Any) -> Event | None:
        position = np.asarray(position, dtype=float)
        if len(position) < 2:
            return None
        for local_index in range(1, len(position)):
            step_displacement = float(np.linalg.norm(position[local_index] - position[local_index - 1]))
            if step_displacement > self.approach_displacement_threshold:
                return Event(
                    name="right_arm_start_moving",
                    local_index=local_index,
                    score=step_displacement,
                    method=f"step_displacement_over_threshold(threshold={self.approach_displacement_threshold})",
                )
        return None

    def _maybe_capture_approach_stop(self, items: list[dict[str, Any]]) -> dict[str, Any] | None:
        if self.approach_stop_event is not None or self.approach_start_event is None:
            return None
        approach_signal = position_signal(items, self.approach_position_spec)
        start_local_index = self._find_local_index(items, self.approach_start_event.frame_index)
        if start_local_index is None:
            return None
        stop_event = self._detect_motion_stop(approach_signal, start_local_index)
        if stop_event is None:
            return None
        self.approach_stop_event = self._freeze_event(items, stop_event)
        return self._event_dict(self.approach_stop_event)

    def _detect_motion_stop(self, position: Any, start_local_index: int) -> Event | None:
        position = np.asarray(position, dtype=float)
        if len(position) < 3:
            return None
        step_displacements = np.linalg.norm(np.diff(position, axis=0), axis=1)
        still_window = max(2, min(self.approach_window, len(step_displacements)))
        confirmation_window = max(still_window, min(still_window * 2, len(step_displacements)))
        search_start = max(int(start_local_index) + 1, 1)
        search_stop = len(position) - still_window + 1
        for local_index in range(search_start, search_stop):
            confirmation_steps = step_displacements[local_index:local_index + confirmation_window]
            if len(confirmation_steps) < confirmation_window:
                continue
            window_steps = confirmation_steps[:still_window]
            if len(window_steps) < still_window:
                continue
            if float(np.max(confirmation_steps)) <= self.approach_displacement_threshold:
                return Event(
                    name="right_arm_stop_moving",
                    local_index=local_index,
                    score=float(np.max(confirmation_steps)),
                    method=(
                        "confirmed_sustained_step_displacement_below_threshold"
                        f"(window={still_window}, confirmation_window={confirmation_window}, "
                        f"threshold={self.approach_displacement_threshold})"
                    ),
                )
        return None

    def _maybe_capture_initial_state(self, items: list[dict[str, Any]]) -> dict[str, Any] | None:
        if self.initial_state_event is not None or not items:
            return None
        first_item = items[0]
        initial_event = LoggedEvent(
            event="initial_state",
            episode_index=normalized_episode_index(first_item),
            frame_index=item_frame_index(first_item, 0),
            timestamp=item_timestamp(first_item),
            score=1.0,
            method="initial_sample_snapshot",
        )
        self.initial_state_event = initial_event
        payload = self._event_dict(initial_event)
        if payload is None:
            return None
        payload["message"] = self._build_initial_state_message(first_item)
        return payload

    def _build_initial_state_message(self, item: dict[str, Any]) -> str:
        if self.robot_type in {"aibot2", "agibot2"}:
            gripper_state = self._describe_initial_gripper_state(item)
            camera_count = len(self.config.cameras)
            return (
                "Initial bimanual state: "
                f"left arm state pending confirmation, right arm gripper is {gripper_state}, "
                f"monitoring {camera_count} camera views."
            )
        return "Initial state captured."

    def _maybe_capture_left_arm_still(self, items: list[dict[str, Any]]) -> dict[str, Any] | None:
        if self.left_arm_still_event is not None or not items:
            return None
        if self.robot_type not in {"aibot2", "agibot2"}:
            return None
        if self.left_arm_position_spec is None:
            return None
        if not self._left_arm_is_still(items):
            return None
        first_item = items[0]
        still_event = LoggedEvent(
            event="left_arm_still_confirmed",
            episode_index=normalized_episode_index(first_item),
            frame_index=item_frame_index(first_item, 0),
            timestamp=item_timestamp(first_item),
            score=1.0,
            method=(
                "sustained_step_displacement_below_threshold"
                f"(window={self._left_arm_still_window(items)}, threshold={self.approach_displacement_threshold})"
            ),
        )
        self.left_arm_still_event = still_event
        payload = self._event_dict(still_event)
        if payload is None:
            return None
        payload["message"] = "Left arm stationary start state confirmed from motion signals."
        return payload

    def _left_arm_is_still(self, items: list[dict[str, Any]]) -> bool:
        if self.left_arm_position_spec is None:
            return False
        left_signal = np.asarray(position_signal(items, self.left_arm_position_spec), dtype=float)
        if len(left_signal) < 2:
            return False
        step_displacements = np.linalg.norm(np.diff(left_signal, axis=0), axis=1)
        still_window = self._left_arm_still_window(items)
        if len(step_displacements) < still_window:
            return False
        return float(np.max(step_displacements[:still_window])) <= self.approach_displacement_threshold

    def _left_arm_still_window(self, items: list[dict[str, Any]]) -> int:
        return max(2, min(self.approach_window, max(len(items) - 1, 1)))

    def _describe_initial_gripper_state(self, item: dict[str, Any]) -> str:
        source = self.initial_state_signal_spec.source
        signal = item.get(source)
        if signal is None:
            return "in its starting state"
        values = list(signal)
        dim = self.initial_state_signal_spec.dim
        if dim >= len(values) or dim < -len(values):
            return "in its starting state"
        gripper_value = float(values[dim])
        distance_to_open = abs(gripper_value - self.initial_state_gripper_value.open_value)
        distance_to_closed = abs(gripper_value - self.initial_state_gripper_value.closed_value)
        return "closed" if distance_to_closed <= distance_to_open else "open"

    def _maybe_capture_gripper_open(self, items: list[dict[str, Any]]) -> dict[str, Any] | None:
        if self.open_event is not None:
            return None
        command_signal = gripper_signal(items, self.start_open_signal_spec)
        open_event = _rename_event(
            detect_gripper_open(command_signal, direction=self.start_open_signal_spec.open_direction),
            "gripper_start_opening",
        )
        if open_event.local_index <= 0 or open_event.score <= 0.0:
            return None
        self.open_event = self._freeze_event(items, open_event)
        return self._event_dict(self.open_event)

    def _maybe_capture_gripper_fully_open(self, items: list[dict[str, Any]]) -> dict[str, Any] | None:
        if self.fully_open_event is not None or self.open_event is None:
            return None
        release_spec = self.release_behavior_spec
        fully_open_signal = gripper_signal(items, release_spec.fully_open_signal)
        open_local_index = self._find_local_index(items, self.open_event.frame_index)
        if open_local_index is None:
            return None

        actual_event = find_gripper_fully_open(
            fully_open_signal,
            start_idx=open_local_index,
            open_value=release_spec.gripper_value.open_value,
            open_tolerance=release_spec.open_tolerance,
            fallback_delay_frames=None,
        )
        if actual_event is not None:
            self.fully_open_event = self._freeze_event(items, actual_event)
            return self._event_dict(self.fully_open_event)

        if release_spec.fallback_delay_frames is None:
            return None
        fallback_index = open_local_index + int(release_spec.fallback_delay_frames)
        if fallback_index >= len(fully_open_signal):
            return None
        fallback_event = Event(
            name="gripper_fully_open",
            local_index=fallback_index,
            score=float(abs(float(fully_open_signal[fallback_index]) - release_spec.gripper_value.open_value)),
            method=f"estimated_delay_fallback(delay_frames={release_spec.fallback_delay_frames})",
        )
        self.fully_open_event = self._freeze_event(items, fallback_event)
        return self._event_dict(self.fully_open_event)

    def _maybe_capture_retreat_start(self, items: list[dict[str, Any]]) -> dict[str, Any] | None:
        if self.retreat_event is not None or self.fully_open_event is None:
            return None
        retreat_signal = position_signal(items, self.retreat_position_spec)
        fully_open_local_index = self._find_local_index(items, self.fully_open_event.frame_index)
        if fully_open_local_index is None:
            return None
        retreat_event = _rename_event(
            find_retreat_start(
                retreat_signal,
                start_idx=fully_open_local_index,
                window=self.retreat_window,
                displacement_threshold=self.retreat_displacement_threshold,
            ),
            "release_retreat_start",
        )
        if retreat_event is None:
            return None
        self.retreat_event = self._freeze_event(items, retreat_event)
        return self._event_dict(self.retreat_event)

    def _maybe_capture_object_status(self, items: list[dict[str, Any]]) -> dict[str, Any] | None:
        if self.open_event is None or self.fully_open_event is None:
            return None

        status_key = build_event_key("object_in_shelf_status", self.fully_open_event.episode_index, self.fully_open_event.frame_index)
        if self.status_event_key == status_key:
            return None

        open_local_index = self._find_local_index(items, self.open_event.frame_index)
        fully_open_local_index = self._find_local_index(items, self.fully_open_event.frame_index)
        if open_local_index is None or fully_open_local_index is None:
            return None

        image_entries = build_image_entries_from_buffer(
            items,
            cameras=list(self.config.cameras),
            event_map={
                "gripper_open": open_local_index,
                "gripper_fully_open": fully_open_local_index,
            },
        )
        if not image_entries:
            return None

        if not self._ensure_vlm_loaded():
            status_event = {
                "event": "object_in_shelf_status",
                "episode_index": self.fully_open_event.episode_index,
                "frame_index": self.fully_open_event.frame_index,
                "timestamp": self.fully_open_event.timestamp,
                "local_index": fully_open_local_index,
                "placement_status": "vlm_unavailable",
                "confidence": None,
                "images": image_entries,
                "result": {"error": self.vlm_error},
                "online_inference_seconds": 0.0,
                "message": f"object QA unavailable: {self.vlm_error}",
            }
            self.status_event_key = status_key
            self.emitted_event_keys.add(status_key)
            return status_event

        inference_start = perf_counter()
        try:
            result = answer_question_about_image_entries(
                image_entries,
                self.config.question,
                keyframe_types=["gripper_open", "gripper_fully_open"],
                model_name=self.config.model_name,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                dtype_name=self.config.dtype_name,
                prompt_mode=self.config.prompt_mode,
                processor=self.processor,
                model=self.model,
            )
        except Exception as exc:
            self.vlm_disabled = True
            self.vlm_error = f"{type(exc).__name__}: {exc}"
            result = {
                "parsed_answer": {
                    "placement_status": "vlm_unavailable",
                    "confidence": None,
                },
                "error": self.vlm_error,
            }
        elapsed = perf_counter() - inference_start

        parsed = result.get("parsed_answer") or {}
        status_event = {
            "event": "object_in_shelf_status",
            "episode_index": self.fully_open_event.episode_index,
            "frame_index": self.fully_open_event.frame_index,
            "timestamp": self.fully_open_event.timestamp,
            "local_index": fully_open_local_index,
            "placement_status": parsed.get("placement_status"),
            "confidence": parsed.get("confidence"),
            "images": image_entries,
            "result": result,
            "online_inference_seconds": elapsed,
            "message": build_alert_message(parsed),
        }
        self.status_event_key = status_key
        self.emitted_event_keys.add(status_key)
        return status_event

    def _ensure_vlm_loaded(self) -> bool:
        if self.vlm_disabled:
            return False
        if self.processor is not None and self.model is not None:
            return True
        cache_key = (self.config.model_name, self.config.dtype_name)
        cached = _VLM_CACHE.get(cache_key)
        if cached is not None:
            self.processor, self.model = cached
            return True
        try:
            with _VLM_CACHE_LOCK:
                cached = _VLM_CACHE.get(cache_key)
                if cached is None:
                    cached = load_qwen_model(self.config.model_name, self.config.dtype_name)
                    _VLM_CACHE[cache_key] = cached
                self.processor, self.model = cached
        except Exception as exc:
            self.vlm_disabled = True
            self.vlm_error = f"{type(exc).__name__}: {exc}"
            self.processor = None
            self.model = None
            return False
        return True

    def _freeze_event(self, items: list[dict[str, Any]], event: Event) -> LoggedEvent:
        item = items[event.local_index]
        return LoggedEvent(
            event=event.name,
            episode_index=normalized_episode_index(item),
            frame_index=item_frame_index(item, event.local_index),
            timestamp=item_timestamp(item),
            score=float(event.score),
            method=event.method,
        )

    def _event_dict(self, event: LoggedEvent) -> dict[str, Any] | None:
        key = build_event_key(event.event, event.episode_index, event.frame_index)
        if key in self.emitted_event_keys:
            return None
        self.emitted_event_keys.add(key)
        return {
            "event": event.event,
            "episode_index": event.episode_index,
            "frame_index": event.frame_index,
            "timestamp": event.timestamp,
            "score": event.score,
            "method": event.method,
        }

    def _find_local_index(self, items: list[dict[str, Any]], frame_index: int) -> int | None:
        for local_index, item in enumerate(items):
            if item_frame_index(item, local_index) == frame_index:
                return local_index
        return None


def normalize_online_sample(sample: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(sample)
    if "images" in normalized:
        images = normalized.pop("images")
        if not isinstance(images, dict):
            raise TypeError("sample['images'] must be a dict when provided")
        for camera, image_path in images.items():
            normalized[camera] = str(image_path)
    return normalized


def normalized_episode_index(item: dict[str, Any]) -> int | None:
    value = item.get("episode_index")
    if value is None:
        return None
    return int(value)


def build_event_key(event_name: str, episode_index: int | None, frame_index: int) -> tuple[str, int | None, int]:
    return event_name, episode_index, int(frame_index)


def _rename_event(event: Event | None, event_name: str) -> Event | None:
    if event is None:
        return None
    return Event(
        name=event_name,
        local_index=event.local_index,
        score=event.score,
        method=event.method,
        signal_before=event.signal_before,
        signal_after=event.signal_after,
    )


def build_image_entries_from_buffer(
    items: list[dict[str, Any]],
    *,
    cameras: list[str],
    event_map: dict[str, int],
) -> list[dict[str, Any]]:
    image_entries: list[dict[str, Any]] = []
    for keyframe_type, local_index in event_map.items():
        if local_index < 0 or local_index >= len(items):
            continue
        item = items[local_index]
        for camera in cameras:
            image_path = item.get(camera)
            if image_path is None:
                raise KeyError(f"Camera '{camera}' not found in online sample for keyframe '{keyframe_type}'")
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Online image path does not exist: {path}")
            image_entries.append(
                {
                    "keyframe_type": keyframe_type,
                    "camera": camera,
                    "path": str(path),
                }
            )
    return image_entries


def build_alert_message(parsed_answer: dict[str, Any]) -> str:
    placement_status = parsed_answer.get("placement_status", "unknown")
    confidence = parsed_answer.get("confidence")
    return f"placement_status={placement_status}, confidence={confidence}"


def iter_jsonl_samples(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(payload, dict):
                raise TypeError(f"Expected a JSON object on line {line_number} of {path}")
            yield payload
