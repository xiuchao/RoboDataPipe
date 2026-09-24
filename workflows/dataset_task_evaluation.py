from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from dataset_io import DATASET_REGISTRY, build_episode_index, load_lerobot_dataset
from project_paths import KEYFRAME_OUTPUT_DIR
from robehavior.keyframes import (
    extract_keyframes_for_episode,
    gripper_signal_spec_from_config,
    release_behavior_spec_from_config,
    save_keyframes_json,
)
from robehavior.registry import resolve_episode_cameras
from vlm.qwen_vl_config import DEFAULT_MODEL, PROMPT_MODES, resolve_model_name, resolve_question
from vlm.qwen_vl_qa import answer_question_about_keyframes, collect_demonstrations, load_qwen_model


RELEASE_KEYFRAMES = {"gripper_fully_open", "release_keyframe"}


@dataclass(frozen=True)
class DatasetTaskEvaluationConfig:
    dataset_name: str
    registry_path: str | Path = DATASET_REGISTRY
    question: str | None = None
    cameras: list[str] | None = None
    keyframe_types: list[str] | None = None
    episodes: list[int] | None = None
    demo_upright: str | Path | None = None
    demo_non_upright: str | Path | None = None
    shot_mode: str = "fewshot"
    keyframe_out: str | Path = KEYFRAME_OUTPUT_DIR
    model_name: str = DEFAULT_MODEL
    max_new_tokens: int = 256
    temperature: float = 0.0
    dtype_name: str = "auto"
    prompt_mode: str = "qa"
    offset: int = 5
    smooth_window: int = 1
    retreat_source: str | None = None
    release_gripper_source: str | None = None
    release_open_tolerance: float | None = None
    release_stationary_window: int | None = None
    release_stationary_displacement_threshold: float | None = None


def resolve_evaluation_keyframes(keyframe_types: list[str] | None, prompt_mode: str) -> list[str]:
    if keyframe_types is not None:
        return keyframe_types

    default_keyframes = PROMPT_MODES[prompt_mode].default_keyframe_types
    if default_keyframes is not None:
        return list(default_keyframes)
    return ["episode_start"]


def resolve_episode_indices(dataset: Any, episodes: list[int] | None) -> list[int]:
    available = [episode_slice.episode_index for episode_slice in build_episode_index(dataset)]
    if episodes is None:
        return available

    available_set = set(available)
    missing = [episode_index for episode_index in episodes if episode_index not in available_set]
    if missing:
        raise ValueError(f"Episodes not found: {missing}. Available range: {available[:3]}...{available[-3:]}")
    return episodes


def episode_output_dir(keyframe_out: str | Path, dataset_name: str, episode_index: int) -> Path:
    return Path(keyframe_out) / dataset_name / f"ep_{episode_index:03d}"


def extracted_keyframes_ready(
    keyframe_dir: str | Path,
    keyframe_types: list[str],
    cameras: list[str],
) -> bool:
    keyframe_dir = Path(keyframe_dir)
    json_path = keyframe_dir / "keyframes.json"
    if not json_path.exists():
        return False

    try:
        records = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False

    records_by_type = {
        record.get("keyframe_type"): record
        for record in records
        if isinstance(record, dict) and record.get("keyframe_type") is not None
    }
    for keyframe_type in keyframe_types:
        record = records_by_type.get(keyframe_type)
        if record is None:
            return False
        images = record.get("images", {})
        for camera in cameras:
            image_path = images.get(camera)
            if image_path is None:
                return False
            path = Path(image_path)
            if not path.exists() and not (keyframe_dir / path.name).exists():
                return False
    return True


def evaluate_dataset_task_outcomes(config: DatasetTaskEvaluationConfig) -> dict[str, Any]:
    dataset, dataset_config = load_lerobot_dataset(config.dataset_name, registry_path=config.registry_path)
    keyframe_types = resolve_evaluation_keyframes(config.keyframe_types, config.prompt_mode)
    question = resolve_question(config.question, config.prompt_mode)
    model_name = resolve_model_name(config.model_name)
    episode_indices = resolve_episode_indices(dataset, config.episodes)
    demonstrations = collect_demonstrations(
        config.prompt_mode,
        config.shot_mode,
        config.demo_upright,
        config.demo_non_upright,
    )
    processor, model = load_qwen_model(model_name, config.dtype_name)

    results: dict[str, dict[str, Any]] = {}
    non_straight_episodes: list[str] = []
    uncertain_episodes: list[str] = []
    not_inside_episodes: list[str] = []
    total_inference_seconds = 0.0
    total_frame_count = 0

    for episode_index in tqdm(episode_indices, desc="Episodes", unit="ep"):
        episode_key = f"ep{episode_index:03d}"
        cameras, motion = resolve_episode_cameras(
            dataset,
            dataset_config,
            episode_index,
            config.cameras,
            keyframe_type=keyframe_types[-1] if keyframe_types else None,
        )
        keyframe_dir = episode_output_dir(config.keyframe_out, config.dataset_name, episode_index)
        if not extracted_keyframes_ready(keyframe_dir, keyframe_types, cameras):
            gripper_spec = gripper_signal_spec_from_config(dataset_config)
            release_spec = None
            if set(keyframe_types) & RELEASE_KEYFRAMES:
                release_spec = release_behavior_spec_from_config(
                    dataset_config,
                    gripper_signal_spec=gripper_spec,
                    position_source=config.retreat_source,
                    fully_open_source=config.release_gripper_source,
                    stationary_window=config.release_stationary_window,
                    stationary_displacement_threshold=config.release_stationary_displacement_threshold,
                    open_tolerance=config.release_open_tolerance,
                )
            keyframes = extract_keyframes_for_episode(
                ds=dataset,
                dataset_name=config.dataset_name,
                episode_index=episode_index,
                keyframe_types=keyframe_types,
                cameras=cameras,
                out_dir=keyframe_dir,
                gripper_signal_spec=gripper_spec,
                release_behavior_spec=release_spec,
                dataset_config=dataset_config,
                offset=config.offset,
                smooth_window=config.smooth_window,
            )
            save_keyframes_json(keyframes, keyframe_dir / "keyframes.json")

        episode_result = answer_question_about_keyframes(
            keyframe_dir=keyframe_dir,
            question=question,
            cameras=cameras,
            keyframe_types=keyframe_types,
            model_name=model_name,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            dtype_name=config.dtype_name,
            prompt_mode=config.prompt_mode,
            demonstration_entries=demonstrations,
            processor=processor,
            model=model,
        )
        episode_result.update(
            episode_index=episode_index,
            episode_name=episode_key,
            keyframe_dir=str(keyframe_dir),
            selected_cameras=cameras,
        )
        if motion is not None:
            episode_result.update(
                moving_arm=motion["moving_arm"],
                left_score=motion["left_score"],
                right_score=motion["right_score"],
            )
        results[episode_key] = episode_result
        total_inference_seconds += float(episode_result["inference_seconds"])
        total_frame_count += int(episode_result["frame_count"])

        parsed_answer = episode_result.get("parsed_answer")
        if config.prompt_mode == "cylinder_upright" and isinstance(parsed_answer, dict):
            if parsed_answer.get("is_upright") is False:
                non_straight_episodes.append(episode_key)
        if isinstance(parsed_answer, dict):
            placement_status = parsed_answer.get("placement_status")
            if placement_status == "uncertain":
                uncertain_episodes.append(episode_key)
            elif placement_status == "not_inside":
                not_inside_episodes.append(episode_key)

    return {
        "dataset": config.dataset_name,
        "dataset_root": str(dataset_config.get("root")),
        "registry": str(config.registry_path),
        "prompt_mode": config.prompt_mode,
        "shot_mode": config.shot_mode,
        "keyframe_types": keyframe_types,
        "question": question,
        "model": model_name,
        "keyframe_out": str(Path(config.keyframe_out)),
        "demonstrations": demonstrations,
        "total_episodes": len(results),
        "inference_seconds_total": total_inference_seconds,
        "inference_seconds_per_frame_avg": total_inference_seconds / total_frame_count if total_frame_count else 0.0,
        "non_straight_episodes": non_straight_episodes,
        "uncertain_episodes": uncertain_episodes,
        "not_inside_episodes": not_inside_episodes,
        "episodes": results,
    }