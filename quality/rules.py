from __future__ import annotations

import numpy as np


def assess_learning_quality(
    translation_smoothness: float | None,
    efficiency: float | None,
    hesitation: float | None,
    joint_smoothness: float | None,
) -> tuple[float | None, list[str]]:
    flags: list[str] = []

    if translation_smoothness is not None:
        if translation_smoothness < -25.0:
            flags.append("jerky")
    if efficiency is not None:
        if efficiency < 0.1:
            flags.append("inefficient")
    if hesitation is not None:
        if hesitation > 0.2:
            flags.append("hesitant")

    if any(
        value is None
        for value in (
            translation_smoothness,
            efficiency,
            hesitation,
            joint_smoothness,
        )
    ):
        return None, flags

    translation_score = _smoothness_score(translation_smoothness)
    joint_score = _smoothness_score(joint_smoothness)
    hesitation_score = float(np.clip(1.0 - hesitation, 0.0, 1.0))
    score = 10.0 * (
        0.35 * translation_score
        + 0.35 * float(np.clip(efficiency, 0.0, 1.0))
        + 0.20 * hesitation_score
        + 0.10 * joint_score
    )
    return float(score), flags


def _smoothness_score(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(value + 18.0) / 4.0)))