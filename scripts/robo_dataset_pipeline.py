from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import DATASET_REGISTRY, load_lerobot_dataset
from project_paths import KEYFRAME_OUTPUT_DIR, QWENVL_OUTPUT_DIR, RESULT_OUTPUT_DIR
from robot_events.keyframes import (
	build_episode_index,
	extract_keyframes_for_episode,
	gripper_signal_spec_from_config,
	release_behavior_spec_from_config,
	retreat_behavior_spec_from_config,
	save_keyframes_json,
)

RETREAT_KEYFRAMES = {"pre_retreat", "retreat_start", "post_retreat"}
RELEASE_KEYFRAMES = {"gripper_fully_open", "release_keyframe"}
from robot_events.registry import resolve_episode_cameras
from vlm.qwen_vl_config import DEFAULT_MODEL, PROMPT_MODES, resolve_model_name, resolve_question
from vlm.qwen_vl_qa import answer_question_about_keyframes, collect_demonstrations, load_qwen_model
from tqdm import tqdm

DEFAULT_KEYFRAME_OUT = str(KEYFRAME_OUTPUT_DIR)
DEFAULT_RESULT_JSON_OUT = str(QWENVL_OUTPUT_DIR)
DEFAULT_RESULT_TEXT_OUT = str(RESULT_OUTPUT_DIR)

def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--dataset",
		required=True,
		help="Dataset nickname in datasets.yaml, e.g. DSRFM_easy.",
	)
	parser.add_argument(
		"--registry",
		default=DATASET_REGISTRY,
		help="Path to datasets.yaml.",
	)
	parser.add_argument(
		"--episode",
		type=int,
		action="append",
		dest="episodes",
		default=None,
		help="Optional episode index to process. Can be repeated. Defaults to all episodes.",
	)
	parser.add_argument(
		"--question",
		default=None,
		help="Question to ask for each episode. Required for --prompt-mode qa.",
	)
	parser.add_argument(
		"--prompt-mode",
		choices=sorted(PROMPT_MODES.keys()),
		default="qa",
		help="Prompt template to use. Default: qa",
	)
	parser.add_argument(
		"--shot-mode",
		choices=["zeroshot", "fewshot"],
		default="fewshot",
		help="Whether to use no demonstrations or include labeled demonstration images. Default: fewshot.",
	)
	parser.add_argument(
		"--camera",
		action="append",
		dest="cameras",
		default=None,
		help="Optional camera key to include. Can be repeated.",
	)
	parser.add_argument(
		"--demo-upright",
		default=None,
		help="Optional labeled upright example image or directory for few-shot calibration.",
	)
	parser.add_argument(
		"--demo-non-upright",
		default=None,
		help="Optional labeled non-upright example image or directory for few-shot calibration.",
	)
	parser.add_argument(
		"--keyframes",
		"--keyframe-type",
		nargs="+",
		dest="keyframe_types",
		default=None,
		help="Keyframe types to extract and evaluate, e.g. --keyframes episode_start.",
	)
	parser.add_argument(
		"--keyframe-out",
		default=DEFAULT_KEYFRAME_OUT,
		help="Directory where extracted keyframes are saved.",
	)
	parser.add_argument(
		"--model",
		default=DEFAULT_MODEL,
		help=(
			f"Model name or alias. Default: {DEFAULT_MODEL}. "
			"Examples: 3b, 7b, Qwen/Qwen2.5-VL-3B-Instruct."
		),
	)
	parser.add_argument(
		"--max-new-tokens",
		type=int,
		default=256,
		help="Maximum number of generated tokens per episode.",
	)
	parser.add_argument(
		"--temperature",
		type=float,
		default=0.0,
		help="Sampling temperature. Use 0 for greedy decoding.",
	)
	parser.add_argument(
		"--dtype",
		choices=["auto", "bfloat16", "float16", "float32"],
		default="auto",
		help="Torch dtype for model loading.",
	)
	parser.add_argument(
		"--offset",
		type=int,
		default=5,
		help="Frame offset for pre/post extracted keyframes.",
	)
	parser.add_argument(
		"--smooth-window",
		type=int,
		default=1,
		help="Optional moving average window for gripper signal.",
	)
	parser.add_argument(
		"--retreat-source",
		default=None,
		help="Optional position source for retreat detection, e.g. action or observation.state.",
	)
	parser.add_argument(
		"--retreat-window",
		type=int,
		default=None,
		help="Window size in frames for retreat-start displacement detection.",
	)
	parser.add_argument(
		"--retreat-displacement-threshold",
		type=float,
		default=None,
		help="Minimum position displacement needed to mark retreat_start.",
	)
	parser.add_argument(
		"--release-open-tolerance",
		type=float,
		default=None,
		help="Tolerance around the configured open_value when resolving release_keyframe.",
	)
	parser.add_argument(
		"--release-gripper-source",
		default=None,
		help="Signal source used for gripper_fully_open, e.g. observation.state or action.",
	)
	parser.add_argument(
		"--release-stationary-window",
		type=int,
		default=None,
		help="Forward window in frames used to confirm the arm is still stationary at release_keyframe.",
	)
	parser.add_argument(
		"--release-stationary-displacement-threshold",
		type=float,
		default=None,
		help="Maximum forward displacement allowed while the arm is still considered stationary at release_keyframe.",
	)
	parser.add_argument(
		"--output-json",
		default=None,
		help="Optional path to save the dataset-level aggregate JSON. If omitted, a timestamped path is generated automatically.",
	)
	parser.add_argument(
		"--output-txt",
		default=None,
		help="Optional path to save the human-readable episode summary text. If omitted, a timestamped path is generated automatically.",
	)
	return parser.parse_args()


def format_summary_text(result: dict[str, Any]) -> str:
	run_timestamp = result.get("run_time") or datetime.now().astimezone().isoformat(timespec="seconds")
	lines = [
		f"[run_time] {run_timestamp}",
		f"[dataset] {result['dataset']}",
		f"[model] {result['model']}",
		f"[prompt_mode] {result['prompt_mode']}",
		f"[shot_mode] {result['shot_mode']}",
		f"[episodes] {result['total_episodes']}",
		f"[inference_seconds_total] {result['inference_seconds_total']:.3f}",
		f"[inference_seconds_per_frame_avg] {result['inference_seconds_per_frame_avg']:.3f}",
		f"[non_straight_episodes] {result['non_straight_episodes']}",
		f"[uncertain_episodes] {result.get('uncertain_episodes', [])}",
		f"[not_inside_episodes] {result.get('not_inside_episodes', [])}",
	]

	if result["question"] is not None:
		lines.insert(5, f"[question] {result['question']}")

	for episode_key, episode_result in result["episodes"].items():
		lines.append(f"[episode] {episode_key}")
		lines.append(f"[episode_inference_seconds] {episode_result['inference_seconds']:.3f}")
		lines.append(f"[episode_inference_seconds_per_frame] {episode_result['inference_seconds_per_frame']:.3f}")
		lines.append(str(episode_result["answer"]))

	return "\n".join(lines) + "\n"


def build_result_basename(result: dict[str, Any], episodes: list[int] | None, run_dt: datetime) -> str:
	keyframe_label = "-".join(result["keyframe_types"]) if result.get("keyframe_types") else "keyframes"
	if episodes is None:
		episode_label = "all"
	elif len(episodes) == 1:
		episode_label = f"ep{episodes[0]:03d}"
	else:
		episode_label = f"{len(episodes)}eps"
	timestamp_label = run_dt.strftime("%m%d_%H%M%S")
	return f"{keyframe_label}_{result['prompt_mode']}_{result['shot_mode']}_{episode_label}_{timestamp_label}"


def resolve_output_paths(
	result: dict[str, Any],
	episodes: list[int] | None,
	run_dt: datetime,
	output_json: str | None,
	output_txt: str | None,
) -> tuple[Path, Path]:
	basename = build_result_basename(result, episodes, run_dt)
	if output_json is None:
		json_path = Path(DEFAULT_RESULT_JSON_OUT) / result["dataset"] / basename.replace(" ", "_")
		json_path = json_path.with_suffix(".json")
	else:
		json_path = Path(output_json)

	if output_txt is None:
		text_path = Path(DEFAULT_RESULT_TEXT_OUT) / f"{result['dataset']}_{basename}_summary.txt"
	else:
		text_path = Path(output_txt)

	return json_path, text_path

def episode_result_key(episode_index: int) -> str:
	return f"ep{episode_index:03d}"

def episode_output_dir(keyframe_out: str | Path, dataset_name: str, episode_index: int) -> Path:
	return Path(keyframe_out) / dataset_name / f"ep_{episode_index:03d}"

def resolve_keyframes_for_pipeline(keyframe_types: list[str] | None, prompt_mode: str) -> list[str]:
	if keyframe_types is not None:
		return keyframe_types

	default_keyframes = PROMPT_MODES[prompt_mode].default_keyframe_types
	if default_keyframes is not None:
		return list(default_keyframes)

	return ["episode_start"]

def resolve_episode_indices(ds: Any, episodes: list[int] | None) -> list[int]:
	available = [episode_slice.episode_index for episode_slice in build_episode_index(ds)]
	if episodes is None:
		return available

	available_set = set(available)
	missing = [episode_index for episode_index in episodes if episode_index not in available_set]
	if missing:
		raise ValueError(f"Episodes not found: {missing}. Available range: {available[:3]}...{available[-3:]}")

	return episodes


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
		records = json.loads(json_path.read_text())
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

def collect_dataset_level_information(
	dataset_name: str,
	registry_path: str | Path = DATASET_REGISTRY,
	question: str | None = None,
	cameras: list[str] | None = None,
	keyframe_types: list[str] | None = None,
	episodes: list[int] | None = None,
	demo_upright: str | Path | None = None,
	demo_non_upright: str | Path | None = None,
	shot_mode: str = "fewshot",
	keyframe_out: str | Path = DEFAULT_KEYFRAME_OUT,
	model_name: str = DEFAULT_MODEL,
	max_new_tokens: int = 256,
	temperature: float = 0.0,
	dtype_name: str = "auto",
	prompt_mode: str = "qa",
	offset: int = 5,
	smooth_window: int = 1,
	retreat_source: str | None = None,
	retreat_window: int | None = None,
	retreat_displacement_threshold: float | None = None,
	release_gripper_source: str | None = None,
	release_open_tolerance: float | None = None,
	release_stationary_window: int | None = None,
	release_stationary_displacement_threshold: float | None = None,
) -> dict[str, Any]:
	ds, cfg = load_lerobot_dataset(dataset_name, registry_path=registry_path)
	resolved_keyframe_types = resolve_keyframes_for_pipeline(keyframe_types, prompt_mode)
	resolved_question = resolve_question(question, prompt_mode)
	resolved_model_name = resolve_model_name(model_name)
	episode_indices = resolve_episode_indices(ds, episodes)
	demonstration_entries = collect_demonstrations(prompt_mode, shot_mode, demo_upright, demo_non_upright)

	processor, model = load_qwen_model(resolved_model_name, dtype_name)
	results: dict[str, dict[str, Any]] = {}
	non_straight_episodes: list[str] = []
	uncertain_episodes: list[str] = []
	not_inside_episodes: list[str] = []
	total_inference_seconds = 0.0
	total_frame_count = 0

	for episode_index in tqdm(episode_indices, desc="Episodes", unit="ep"):
		episode_key = episode_result_key(episode_index)
		resolved_cameras, motion = resolve_episode_cameras(
			ds,
			cfg,
			episode_index,
			cameras,
			keyframe_type=resolved_keyframe_types[-1] if resolved_keyframe_types else None,
		)
		keyframe_dir = episode_output_dir(keyframe_out, dataset_name, episode_index)
		if not extracted_keyframes_ready(keyframe_dir, resolved_keyframe_types, resolved_cameras):
			gripper_signal_spec = gripper_signal_spec_from_config(cfg)
			retreat_behavior_spec = None
			release_behavior_spec = None
			needs_retreat = bool(set(resolved_keyframe_types) & (RETREAT_KEYFRAMES | RELEASE_KEYFRAMES))
			if needs_retreat:
				retreat_behavior_spec = retreat_behavior_spec_from_config(
					cfg,
					gripper_signal_spec=gripper_signal_spec,
					position_source=retreat_source,
					window=retreat_window,
					displacement_threshold=retreat_displacement_threshold,
				)
			if set(resolved_keyframe_types) & RELEASE_KEYFRAMES:
				release_behavior_spec = release_behavior_spec_from_config(
					cfg,
					gripper_signal_spec=gripper_signal_spec,
					position_source=retreat_source,
					fully_open_source=release_gripper_source,
					stationary_window=release_stationary_window,
					stationary_displacement_threshold=release_stationary_displacement_threshold,
					open_tolerance=release_open_tolerance,
				)
			keyframes = extract_keyframes_for_episode(
				ds=ds,
				dataset_name=dataset_name,
				episode_index=episode_index,
				keyframe_types=resolved_keyframe_types,
				cameras=resolved_cameras,
				out_dir=keyframe_dir,
				gripper_signal_spec=gripper_signal_spec,
				retreat_behavior_spec=retreat_behavior_spec,
				release_behavior_spec=release_behavior_spec,
				offset=offset,
				smooth_window=smooth_window,
			)
			save_keyframes_json(keyframes, keyframe_dir / "keyframes.json")

		episode_result = answer_question_about_keyframes(
			keyframe_dir=keyframe_dir,
			question=question,
			cameras=resolved_cameras,
			keyframe_types=resolved_keyframe_types,
			model_name=resolved_model_name,
			max_new_tokens=max_new_tokens,
			temperature=temperature,
			dtype_name=dtype_name,
			prompt_mode=prompt_mode,
			demonstration_entries=demonstration_entries,
			processor=processor,
			model=model,
		)
		episode_result["episode_index"] = episode_index
		episode_result["episode_name"] = episode_key
		episode_result["keyframe_dir"] = str(keyframe_dir)
		episode_result["selected_cameras"] = resolved_cameras
		if motion is not None:
			episode_result["moving_arm"] = motion["moving_arm"]
			episode_result["left_score"] = motion["left_score"]
			episode_result["right_score"] = motion["right_score"]
		results[episode_key] = episode_result
		total_inference_seconds += float(episode_result["inference_seconds"])
		total_frame_count += int(episode_result["frame_count"])

		parsed_answer = episode_result.get("parsed_answer")
		if prompt_mode == "cylinder_upright" and isinstance(parsed_answer, dict):
			if parsed_answer.get("is_upright") is False:
				non_straight_episodes.append(episode_key)
		if isinstance(parsed_answer, dict):
			placement_status = parsed_answer.get("placement_status")
			if placement_status == "uncertain":
				uncertain_episodes.append(episode_key)
			elif placement_status == "not_inside":
				not_inside_episodes.append(episode_key)

	return {
		"dataset": dataset_name,
		"dataset_root": str(cfg.get("root")),
		"registry": str(registry_path),
		"prompt_mode": prompt_mode,
		"shot_mode": shot_mode,
		"keyframe_types": resolved_keyframe_types,
		"question": resolved_question,
		"model": resolved_model_name,
		"keyframe_out": str(Path(keyframe_out)),
		"demonstrations": demonstration_entries,
		"total_episodes": len(results),
		"inference_seconds_total": total_inference_seconds,
		"inference_seconds_per_frame_avg": total_inference_seconds / total_frame_count if total_frame_count else 0.0,
		"non_straight_episodes": non_straight_episodes,
		"uncertain_episodes": uncertain_episodes,
		"not_inside_episodes": not_inside_episodes,
		"episodes": results,
	}

if __name__ == "__main__":
	args = parse_args()
	run_dt = datetime.now().astimezone()
	result = collect_dataset_level_information(
		dataset_name=args.dataset,
		registry_path=args.registry,
		question=args.question,
		cameras=args.cameras,
		keyframe_types=args.keyframe_types,
		episodes=args.episodes,
		demo_upright=args.demo_upright,
		demo_non_upright=args.demo_non_upright,
		keyframe_out=args.keyframe_out,
		model_name=args.model,
		max_new_tokens=args.max_new_tokens,
		temperature=args.temperature,
		dtype_name=args.dtype,
		prompt_mode=args.prompt_mode,
		shot_mode=args.shot_mode,
		offset=args.offset,
		smooth_window=args.smooth_window,
		retreat_source=args.retreat_source,
		retreat_window=args.retreat_window,
		retreat_displacement_threshold=args.retreat_displacement_threshold,
		release_gripper_source=args.release_gripper_source,
		release_open_tolerance=args.release_open_tolerance,
		release_stationary_window=args.release_stationary_window,
		release_stationary_displacement_threshold=args.release_stationary_displacement_threshold,
	)
	result["run_time"] = run_dt.isoformat(timespec="seconds")

	summary_text = format_summary_text(result)
	print(summary_text, end="")

	output_path, text_path = resolve_output_paths(result, args.episodes, run_dt, args.output_json, args.output_txt)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	output_path.write_text(json.dumps(result, indent=2))
	print(f"[json] {output_path}")

	text_path.parent.mkdir(parents=True, exist_ok=True)
	text_path.write_text(summary_text)
	print(f"[txt] {text_path}")
	
    