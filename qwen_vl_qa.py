from __future__ import annotations

import argparse
import json
from time import perf_counter
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_config import (
    DEFAULT_MODEL,
    PROMPT_MODES,
    build_messages_for_mode,
    resolve_default_demonstration_paths,
    resolve_keyframe_types,
    resolve_question,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keyframe-dir",
        required=True,
        help="Episode directory containing keyframes.json.",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Question to ask about the extracted keyframes. Required for --prompt-mode qa.",
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
        help="Optional keyframe types to include, e.g. --keyframes episode_start post_place.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens.",
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
        "--output-json",
        default=None,
        help="Optional path to save the prompt metadata and answer as JSON.",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> str | torch.dtype:
    if dtype_name == "auto":
        return "auto"
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]


def load_keyframe_records(keyframe_dir: str | Path) -> list[dict[str, Any]]:
    keyframe_dir = Path(keyframe_dir)
    json_path = keyframe_dir / "keyframes.json"
    if not json_path.exists():
        raise FileNotFoundError(f"keyframes.json not found under {keyframe_dir}")
    return json.loads(json_path.read_text())


def select_records(
    records: list[dict[str, Any]],
    keyframe_types: list[str] | None,
) -> list[dict[str, Any]]:
    if not keyframe_types:
        return records
    wanted = set(keyframe_types)
    selected = [record for record in records if record.get("keyframe_type") in wanted]
    if not selected:
        available = sorted({record.get("keyframe_type") for record in records})
        raise ValueError(f"No matching keyframes found. Available: {available}")
    return selected


def collect_images(
    keyframe_dir: str | Path,
    records: list[dict[str, Any]],
    cameras: list[str] | None,
) -> list[dict[str, Any]]:
    keyframe_dir = Path(keyframe_dir)
    image_entries: list[dict[str, Any]] = []
    for record in records:
        images = record.get("images", {})
        if cameras is None:
            selected_cameras = list(images.keys())
        else:
            selected_cameras = cameras

        for camera in selected_cameras:
            image_path = images.get(camera)
            if image_path is None:
                available = sorted(images.keys())
                raise KeyError(
                    f"Camera '{camera}' not found for keyframe '{record.get('keyframe_type')}'. Available: {available}"
                )
            path = Path(image_path)
            if not path.exists():
                fallback_path = keyframe_dir / path.name
                if fallback_path.exists():
                    path = fallback_path
                else:
                    raise FileNotFoundError(f"Image not found: {path}")
            image_entries.append(
                {
                    "keyframe_type": record.get("keyframe_type"),
                    "camera": camera,
                    "path": str(path),
                }
            )
    if not image_entries:
        raise ValueError("No images were selected from keyframes.json")
    return image_entries


def collect_demonstrations(
    prompt_mode: str,
    shot_mode: str,
    demo_upright: str | Path | None = None,
    demo_non_upright: str | Path | None = None,
) -> list[dict[str, Any]] | None:
    def expand_demo_paths(path_or_dir: str | Path | None) -> list[Path]:
        if path_or_dir is None:
            return []

        path = Path(path_or_dir)
        if not path.exists():
            raise FileNotFoundError(f"Demonstration image or directory not found: {path}")

        if path.is_dir():
            image_paths = sorted(
                child
                for child in path.iterdir()
                if child.is_file() and child.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
            )
            if not image_paths:
                raise FileNotFoundError(f"No demonstration images found in directory: {path}")
            return image_paths

        return [path]

    demo_upright, demo_non_upright = resolve_default_demonstration_paths(
        prompt_mode,
        shot_mode,
        demo_upright,
        demo_non_upright,
    )
    demonstration_entries: list[dict[str, Any]] = []
    for label, image_path, note in [
        ("upright", demo_upright, "The cylinder is standing upright."),
        ("non_upright", demo_non_upright, "The cylinder is not upright."),
    ]:
        for path in expand_demo_paths(image_path):
            demonstration_entries.append({"label": label, "path": str(path), "note": note})

    return demonstration_entries or None


def load_qwen_model(model_name: str, dtype_name: str):
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=resolve_dtype(dtype_name),
        device_map="auto",
    )
    return processor, model


def generate_answer(
    messages: list[dict[str, Any]],
    *,
    processor: Any,
    model: Any,
    max_new_tokens: int,
    temperature: float,
) -> str:
    prompt_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    model_device = next(model.parameters()).device
    inputs = {name: value.to(model_device) for name, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0.0,
    }
    if temperature > 0.0:
        generation_kwargs["temperature"] = temperature

    generated_ids = model.generate(**inputs, **generation_kwargs)
    generated_trimmed = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs["input_ids"], generated_ids)
    ]
    return processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def parse_structured_answer(answer: str) -> dict[str, Any] | None:
    cleaned = answer.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict):
        return parsed
    return None


def answer_question_about_keyframes(
    keyframe_dir: str | Path,
    question: str | None,
    cameras: list[str] | None = None,
    keyframe_types: list[str] | None = None,
    model_name: str = DEFAULT_MODEL,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    dtype_name: str = "auto",
    prompt_mode: str = "qa",
    demonstration_entries: list[dict[str, Any]] | None = None,
    processor: Any | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    records = load_keyframe_records(keyframe_dir)
    resolved_keyframe_types = resolve_keyframe_types(records, keyframe_types, prompt_mode)
    selected_records = select_records(records, resolved_keyframe_types)
    image_entries = collect_images(keyframe_dir, selected_records, cameras)
    resolved_question = resolve_question(question, prompt_mode)
    messages = build_messages_for_mode(
        image_entries,
        resolved_question,
        prompt_mode,
        demonstration_entries=demonstration_entries,
    )

    if processor is None or model is None:
        processor, model = load_qwen_model(model_name, dtype_name)

    inference_start = perf_counter()
    answer = generate_answer(
        messages,
        processor=processor,
        model=model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    inference_seconds = perf_counter() - inference_start
    frame_count = len(image_entries)

    return {
        "prompt_mode": prompt_mode,
        "keyframe_types": resolved_keyframe_types,
        "question": resolved_question,
        "answer": answer,
        "parsed_answer": parse_structured_answer(answer),
        "model": model_name,
        "images": image_entries,
        "frame_count": frame_count,
        "inference_seconds": inference_seconds,
        "inference_seconds_per_frame": inference_seconds / frame_count if frame_count else None,
        "demonstrations": demonstration_entries,
    }


if __name__ == "__main__":
    args = parse_args()
    result = answer_question_about_keyframes(
        keyframe_dir=args.keyframe_dir,
        question=args.question,
        cameras=args.cameras,
        keyframe_types=args.keyframe_types,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        dtype_name=args.dtype,
        prompt_mode=args.prompt_mode,
        demonstration_entries=collect_demonstrations(args.prompt_mode, args.shot_mode, args.demo_upright, args.demo_non_upright),
    )

    print(f"[model] {result['model']}")
    print(f"[prompt_mode] {result['prompt_mode']}")
    if result["question"] is not None:
        print(f"[question] {result['question']}")
    print(f"[inference_seconds] {result['inference_seconds']:.3f}")
    print(f"[inference_seconds_per_frame] {result['inference_seconds_per_frame']:.3f}")
    for image in result["images"]:
        print(
            f"[image] keyframe={image['keyframe_type']} camera={image['camera']} path={image['path']}"
        )
    print("[answer]")
    print(result["answer"])

    if args.output_json is not None:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2))
        print(f"[json] {output_path}")