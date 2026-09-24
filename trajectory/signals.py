from __future__ import annotations

import numpy as np


def vector_step_norms(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.empty(0, dtype=np.float64)
    return np.linalg.norm(np.diff(values, axis=0), axis=1)


def quaternion_angular_steps(quaternions: np.ndarray) -> np.ndarray:
    if len(quaternions) < 2:
        return np.empty(0, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("quaternion norm must be non-zero")
    normalized = quaternions / norms
    dots = np.sum(normalized[:-1] * normalized[1:], axis=1)
    return 2.0 * np.arccos(np.clip(np.abs(dots), 0.0, 1.0))


def scalar_step_changes(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.empty(0, dtype=np.float64)
    flattened = values.reshape(len(values), -1)
    return np.max(np.abs(np.diff(flattened, axis=0)), axis=1)