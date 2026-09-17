from __future__ import annotations

import numpy as np

from quality.models import TimestampQuality


def path_length(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def log_dimensionless_jerk(positions: np.ndarray, dt: float) -> float | None:
    if len(positions) < 4 or dt <= 0:
        return None
    velocity = np.gradient(positions, dt, axis=0)
    acceleration = np.gradient(velocity, dt, axis=0)
    jerk = np.gradient(acceleration, dt, axis=0)
    duration = (len(positions) - 1) * dt
    peak_velocity = float(np.max(np.linalg.norm(velocity, axis=1)))
    if peak_velocity < 1e-10 or duration < 1e-10:
        return None
    jerk_energy = float(np.sum(jerk**2) * dt)
    return float(-np.log(max(duration**3 / peak_velocity**2 * jerk_energy, 1e-10)))


def trajectory_efficiency(positions: np.ndarray) -> float | None:
    if len(positions) < 2:
        return None
    traveled = path_length(positions)
    if traveled < 1e-10:
        return None
    displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    return float(np.clip(displacement / traveled, 0.0, 1.0))


def hesitation_fraction(positions: np.ndarray, dt: float) -> float | None:
    if len(positions) < 3 or dt <= 0:
        return None
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / dt
    active_speed = float(np.percentile(speed, 90))
    if active_speed < 1e-10:
        return None
    paused = speed < 0.05 * active_speed
    active = ~paused
    active_indices = np.flatnonzero(active)
    if len(active_indices) < 2:
        return 0.0
    interior = paused[active_indices[0]:active_indices[-1] + 1]
    return float(np.mean(interior)) if len(interior) else 0.0


def gripper_chatter_rate(signal: np.ndarray, duration: float) -> float | None:
    if signal is None or len(signal) < 2 or duration <= 0:
        return None
    values = signal[:, 0] if signal.ndim == 2 else signal
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum - minimum < 1e-6:
        return 0.0
    threshold = (minimum + maximum) / 2.0
    binary = (values > threshold).astype(np.int32)
    transitions = int(np.abs(np.diff(binary)).sum())
    return float(transitions / duration)


def timestamp_quality(timestamps: np.ndarray) -> TimestampQuality | None:
    if timestamps is None or len(timestamps) < 2:
        return None
    dt = np.diff(timestamps)
    dt_mean = float(np.mean(dt))
    dt_std = float(np.std(dt))
    return TimestampQuality(
        dt_mean=dt_mean,
        dt_std=dt_std,
        jitter_ratio=dt_std / max(dt_mean, 1e-10),
    )


def quality_score(
    smoothness: float | None,
    efficiency: float | None,
    hesitation: float | None,
    chatter: float | None,
    jitter_ratio: float | None,
) -> tuple[float | None, list[str]]:
    parts: list[float] = []
    flags: list[str] = []

    if smoothness is not None:
        parts.append(1.0 / (1.0 + np.exp(-(smoothness + 18.0) / 4.0)))
        if smoothness < -25.0:
            flags.append("jerky")
    if efficiency is not None:
        parts.append(efficiency)
        if efficiency < 0.1:
            flags.append("inefficient")
    if hesitation is not None:
        parts.append(max(0.0, 1.0 - hesitation))
        if hesitation > 0.2:
            flags.append("hesitant")
    if chatter is not None:
        parts.append(max(0.0, 1.0 - chatter / 5.0))
        if chatter > 2.0:
            flags.append("gripper_chatter")
    if jitter_ratio is not None:
        parts.append(max(0.0, 1.0 - jitter_ratio))
        if jitter_ratio > 0.1:
            flags.append("timestamp_jitter")

    if not parts:
        return None, flags
    return float(np.mean(parts) * 10.0), flags