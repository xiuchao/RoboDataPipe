from __future__ import annotations

import numpy as np

from quality.results import TimestampQuality
from trajectory.signals import quaternion_angular_steps


def path_length(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def active_path_length(signal: np.ndarray, active_transitions: np.ndarray) -> float:
    if len(signal) < 2:
        return 0.0
    steps = np.linalg.norm(np.diff(signal, axis=0), axis=1)
    if active_transitions.shape != steps.shape:
        raise ValueError("active transition mask must have one value per signal transition")
    return float(steps[active_transitions].sum())


def rotation_metrics(
    quaternions: np.ndarray,
    dt: float,
    active_transitions: np.ndarray | None = None,
) -> tuple[float, float | None, float | None]:
    angular_steps = quaternion_angular_steps(quaternions)
    acceleration_parts: list[np.ndarray] = []
    if active_transitions is not None:
        if active_transitions.shape != angular_steps.shape:
            raise ValueError("active transition mask must have one value per rotation transition")
        for start, stop in _true_runs(active_transitions):
            run_speed = angular_steps[start:stop] / dt if dt > 0 else np.asarray([])
            if len(run_speed) >= 2:
                acceleration_parts.append(np.diff(run_speed) / dt)
        angular_steps = angular_steps[active_transitions]
    path = float(angular_steps.sum())
    if len(angular_steps) == 0 or dt <= 0:
        return path, None, None
    angular_speed = angular_steps / dt
    speed_mean = float(np.mean(angular_speed))
    if active_transitions is not None:
        if not acceleration_parts:
            return path, speed_mean, None
        acceleration = np.concatenate(acceleration_parts)
    elif len(angular_speed) < 2:
        return path, speed_mean, None
    else:
        acceleration = np.diff(angular_speed) / dt
    return path, speed_mean, float(np.sqrt(np.mean(acceleration**2)))


def segmented_log_dimensionless_jerk(
    signal: np.ndarray,
    dt: float,
    active_transitions: np.ndarray,
) -> float | None:
    values: list[tuple[float, int]] = []
    for start, stop in _true_runs(active_transitions):
        segment = signal[start:stop + 1]
        value = log_dimensionless_jerk(segment, dt)
        if value is not None:
            values.append((value, len(segment)))
    if not values:
        return None
    return float(np.average([value for value, _ in values], weights=[weight for _, weight in values]))


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist()))


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


def segmented_trajectory_efficiency(
    positions: np.ndarray,
    included_transitions: np.ndarray,
) -> float | None:
    values: list[tuple[float, int]] = []
    for start, stop in _true_runs(included_transitions):
        segment = positions[start:stop + 1]
        value = trajectory_efficiency(segment)
        if value is not None:
            values.append((value, stop - start))
    if not values:
        return None
    return float(
        np.average(
            [value for value, _ in values],
            weights=[weight for _, weight in values],
        )
    )


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


def segmented_hesitation_fraction(
    positions: np.ndarray,
    dt: float,
    included_transitions: np.ndarray,
) -> float | None:
    values: list[tuple[float, int]] = []
    for start, stop in _true_runs(included_transitions):
        segment = positions[start:stop + 1]
        value = hesitation_fraction(segment, dt)
        if value is not None:
            values.append((value, stop - start))
    if not values:
        return None
    return float(
        np.average(
            [value for value, _ in values],
            weights=[weight for _, weight in values],
        )
    )


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


def masked_gripper_chatter_rate(
    signal: np.ndarray,
    dt: float,
    included_transitions: np.ndarray,
) -> float | None:
    values = signal[:, 0] if signal.ndim == 2 else signal
    if len(values) < 2 or included_transitions.shape != (len(values) - 1,):
        return None
    if not np.any(included_transitions):
        return None
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if maximum - minimum < 1e-6:
        return 0.0
    threshold = (minimum + maximum) / 2.0
    binary = (values > threshold).astype(np.int32)
    transitions = np.abs(np.diff(binary))[included_transitions]
    duration = max(float(np.sum(included_transitions)) * dt, 1e-6)
    return float(np.sum(transitions) / duration)


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
