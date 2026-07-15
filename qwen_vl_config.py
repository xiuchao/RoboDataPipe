from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_DEMO_UPRIGHT = "/data/xiuchao/biArm/DEM/data_anno/upstraight_labeling/upstraight.jpg"
DEFAULT_DEMO_NON_UPRIGHT = "/data/xiuchao/biArm/DEM/data_anno/upstraight_labeling/lying"


@dataclass(frozen=True)
class PromptModeConfig:
    requires_question: bool
    instruction: str
    default_question: str | None = None
    default_keyframe_types: tuple[str, ...] | None = None


def build_cylinder_upright_prompt() -> str:
    return """
    Determine the orientation of the metallic cylindrical tube on top of the metallic rectangular base.

    Estimate the direction of the cylindrical tube's central axis:
    - "upright": the axis is approximately perpendicular to the tabletop. the tube's circular opening is facing upwards, usually cannot be seen.
    - "lying": the axis is parallel to the tabletop, either horizontal or slanted. The visible circular opening is facing sideways.
    - "unclear": the orientation cannot be reliably determined.

        Answer strictly in JSON with this schema:
        {
            "is_upright": true or false,
            "confidence": "high" or "medium" or "low"
        }

        Use "is_upright": true only when the tube is clearly upright.
        Use "is_upright": false when it is lying down or the orientation is unclear.
        Do not include any text outside the JSON.
        """.strip()


PROMPT_MODES: dict[str, PromptModeConfig] = {
    "qa": PromptModeConfig(
        requires_question=True,
        instruction=(
            "You are analyzing robot keyframes from a pick-and-place episode. "
            "Use the images in order and answer the question concisely and concretely."
        ),
    ),
    "success_judge": PromptModeConfig(
        requires_question=False,
        instruction=(
            "You are analyzing robot keyframes from a pick-and-place episode. "
            "Decide whether the overall task appears successful from these images. "
            "Reply with a JSON object only, using this schema: "
            '{"success": true or false, "confidence": 0.0 to 1.0, "reason": "short explanation"}. '
            "Base the answer only on visible evidence in the provided images."
        ),
        default_question="Is the pick-and-place task successful in this episode?",
    ),
    "cylinder_upright": PromptModeConfig(
        requires_question=False,
        instruction=build_cylinder_upright_prompt(),
        default_keyframe_types=("post_place", "episode_end"),
    ),
}


def resolve_default_demonstration_paths(
    prompt_mode: str,
    shot_mode: str,
    demo_upright: str | Path | None,
    demo_non_upright: str | Path | None,
) -> tuple[str | Path | None, str | Path | None]:
    if shot_mode == "zeroshot":
        return None, None

    if prompt_mode != "cylinder_upright":
        return demo_upright, demo_non_upright

    if demo_upright is None and Path(DEFAULT_DEMO_UPRIGHT).exists():
        demo_upright = DEFAULT_DEMO_UPRIGHT
    if demo_non_upright is None and Path(DEFAULT_DEMO_NON_UPRIGHT).exists():
        demo_non_upright = DEFAULT_DEMO_NON_UPRIGHT

    return demo_upright, demo_non_upright


def resolve_question(question: str | None, prompt_mode: str) -> str | None:
    mode_cfg = PROMPT_MODES[prompt_mode]
    if mode_cfg.requires_question:
        if not question:
            raise ValueError(f"--question is required for --prompt-mode {prompt_mode}")
        return question

    if question:
        return question

    return mode_cfg.default_question


def resolve_keyframe_types(
    records: list[dict[str, Any]],
    keyframe_types: list[str] | None,
    prompt_mode: str,
) -> list[str] | None:
    if keyframe_types is not None:
        return keyframe_types

    default_keyframe_types = PROMPT_MODES[prompt_mode].default_keyframe_types
    if default_keyframe_types is None:
        return None

    available = {record.get("keyframe_type") for record in records}
    preferred = [keyframe_type for keyframe_type in default_keyframe_types if keyframe_type in available]
    return preferred or None


def build_messages_for_mode(
    image_entries: list[dict[str, Any]],
    question: str | None,
    prompt_mode: str,
    demonstration_entries: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    mode_cfg = PROMPT_MODES[prompt_mode]
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": mode_cfg.instruction,
        }
    ]

    if demonstration_entries:
        content.append(
            {
                "type": "text",
                "text": "Labeled examples for calibration. Use them to understand the visual distinction before judging the query images.",
            }
        )
        for index, entry in enumerate(demonstration_entries, start=1):
            content.append(
                {
                    "type": "text",
                    "text": f"Example {index}: label={entry['label']}. {entry.get('note', '')}".strip(),
                }
            )
            content.append({"type": "image", "image": entry["path"]})

        content.append(
            {
                "type": "text",
                "text": "Now judge the query image or query images below. Do not copy the example labels unless the query evidence supports them.",
            }
        )

    for index, entry in enumerate(image_entries, start=1):
        content.append(
            {
                "type": "text",
                "text": f"Image {index}: keyframe={entry['keyframe_type']}, camera={entry['camera']}.",
            }
        )
        content.append({"type": "image", "image": entry["path"]})
    if question:
        content.append({"type": "text", "text": f"Question: {question}"})
    return [{"role": "user", "content": content}]