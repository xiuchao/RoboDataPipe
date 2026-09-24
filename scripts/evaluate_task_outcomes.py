from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_io import DATASET_REGISTRY
from project_paths import KEYFRAME_OUTPUT_DIR, QWENVL_OUTPUT_DIR, RESULT_OUTPUT_DIR
from vlm.qwen_vl_config import DEFAULT_MODEL, PROMPT_MODES
from vlm.task_dashboard import records_from_result, write_task_outcome_dashboard
from workflows.dataset_task_evaluation import DatasetTaskEvaluationConfig, evaluate_dataset_task_outcomes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract episode keyframes and evaluate task outcomes with Qwen-VL."
    )
    parser.add_argument("--dataset", required=True, help="Dataset nickname in datasets.yaml.")
    parser.add_argument("--registry", default=DATASET_REGISTRY, help="Path to datasets.yaml.")
    parser.add_argument(
        "--episode",
        type=int,
        action="append",
        dest="episodes",
        default=None,
        help="Episode index to evaluate. Repeat for multiple episodes; defaults to all.",
    )
    parser.add_argument("--question", default=None, help="Question to ask. Required for --prompt-mode qa.")
    parser.add_argument(
        "--prompt-mode",
        choices=sorted(PROMPT_MODES),
        default="qa",
        help="Prompt template to use. Default: qa.",
    )
    parser.add_argument(
        "--shot-mode",
        choices=["zeroshot", "fewshot"],
        default="fewshot",
        help="Whether to include labeled demonstration images. Default: fewshot.",
    )
    parser.add_argument("--camera", action="append", dest="cameras", default=None, help="Camera key to include.")
    parser.add_argument("--demo-upright", default=None, help="Upright example image or directory.")
    parser.add_argument("--demo-non-upright", default=None, help="Non-upright example image or directory.")
    parser.add_argument(
        "--keyframes",
        "--keyframe-type",
        nargs="+",
        dest="keyframe_types",
        default=None,
        help="Keyframe types to extract and evaluate.",
    )
    parser.add_argument(
        "--keyframe-out",
        default=str(KEYFRAME_OUTPUT_DIR),
        help="Directory where extracted keyframes are saved.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name or alias.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated tokens per episode.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Torch dtype for model loading.",
    )
    parser.add_argument("--offset", type=int, default=5, help="Frame offset for pre/post keyframes.")
    parser.add_argument("--smooth-window", type=int, default=1, help="Gripper signal moving-average window.")
    parser.add_argument("--retreat-source", default=None, help="Position source for retreat detection.")
    parser.add_argument("--release-gripper-source", default=None, help="Signal source for fully-open detection.")
    parser.add_argument("--release-open-tolerance", type=float, default=None, help="Open-value tolerance.")
    parser.add_argument(
        "--release-stationary-window",
        type=int,
        default=None,
        help="Forward window for stationary release confirmation.",
    )
    parser.add_argument(
        "--release-stationary-displacement-threshold",
        type=float,
        default=None,
        help="Maximum displacement during stationary release confirmation.",
    )
    parser.add_argument("--output-json", default=None, help="Aggregate JSON output path.")
    parser.add_argument("--output-txt", default=None, help="Human-readable summary output path.")
    parser.add_argument("--output-html", default=None, help="Task-outcome dashboard output path.")
    return parser.parse_args()


def format_summary_text(result: dict[str, Any]) -> str:
    lines = [
        f"[run_time] {result['run_time']}",
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
        lines.extend(
            [
                f"[episode] {episode_key}",
                f"[episode_inference_seconds] {episode_result['inference_seconds']:.3f}",
                f"[episode_inference_seconds_per_frame] {episode_result['inference_seconds_per_frame']:.3f}",
                str(episode_result["answer"]),
            ]
        )
    return "\n".join(lines) + "\n"


def resolve_output_paths(
    result: dict[str, Any],
    episodes: list[int] | None,
    run_time: datetime,
    output_json: str | None,
    output_txt: str | None,
    output_html: str | None,
) -> tuple[Path, Path, Path]:
    keyframe_label = "-".join(result["keyframe_types"]) if result["keyframe_types"] else "keyframes"
    if episodes is None:
        episode_label = "all"
    elif len(episodes) == 1:
        episode_label = f"ep{episodes[0]:03d}"
    else:
        episode_label = f"{len(episodes)}eps"
    basename = (
        f"{keyframe_label}_{result['prompt_mode']}_{result['shot_mode']}_"
        f"{episode_label}_{run_time.strftime('%m%d_%H%M%S')}"
    )
    json_path = Path(output_json) if output_json else Path(QWENVL_OUTPUT_DIR) / result["dataset"] / f"{basename}.json"
    text_path = (
        Path(output_txt)
        if output_txt
        else Path(RESULT_OUTPUT_DIR) / f"{result['dataset']}_{basename}_summary.txt"
    )
    html_path = Path(output_html) if output_html else text_path.with_suffix(".html")
    return json_path, text_path, html_path


def main() -> None:
    args = parse_args()
    run_time = datetime.now().astimezone()
    result = evaluate_dataset_task_outcomes(
        DatasetTaskEvaluationConfig(
            dataset_name=args.dataset,
            registry_path=args.registry,
            question=args.question,
            cameras=args.cameras,
            keyframe_types=args.keyframe_types,
            episodes=args.episodes,
            demo_upright=args.demo_upright,
            demo_non_upright=args.demo_non_upright,
            shot_mode=args.shot_mode,
            keyframe_out=args.keyframe_out,
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            dtype_name=args.dtype,
            prompt_mode=args.prompt_mode,
            offset=args.offset,
            smooth_window=args.smooth_window,
            retreat_source=args.retreat_source,
            release_gripper_source=args.release_gripper_source,
            release_open_tolerance=args.release_open_tolerance,
            release_stationary_window=args.release_stationary_window,
            release_stationary_displacement_threshold=args.release_stationary_displacement_threshold,
        )
    )
    result["run_time"] = run_time.isoformat(timespec="seconds")

    summary_text = format_summary_text(result)
    print(summary_text, end="")
    json_path, text_path, html_path = resolve_output_paths(
        result,
        args.episodes,
        run_time,
        args.output_json,
        args.output_txt,
        args.output_html,
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[json] {json_path}")
    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(summary_text, encoding="utf-8")
    print(f"[txt] {text_path}")
    write_task_outcome_dashboard(
        records_from_result(result),
        result["dataset"],
        html_path,
        subtitle=f"{result['prompt_mode']} | {result['model']} | {result['run_time']}",
    )
    print(f"[html] {html_path}")


if __name__ == "__main__":
    main()