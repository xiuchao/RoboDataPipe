from __future__ import annotations

from typing import Any

import numpy as np

from trajectory.contracts import ArmSignalContract, TrajectoryContract
from trajectory.models import CanonicalArmTrajectory, CanonicalTrajectory


def canonicalize_trajectory(
    items: list[dict[str, Any]],
    contract: TrajectoryContract,
) -> CanonicalTrajectory:
    if not items:
        return CanonicalTrajectory(contract=contract)

    actions = _stack_vector(items, contract.action.key, contract.action.dim)
    states = _stack_vector(items, contract.state.key, contract.state.dim)
    arms = {
        arm_name: _canonicalize_arm(
            actions,
            states,
            action_spec,
            contract.state.arms[arm_name],
        )
        for arm_name, action_spec in contract.action.arms.items()
    }
    timestamps = _timestamps(items, contract)
    return CanonicalTrajectory(arms=arms, timestamps=timestamps, contract=contract)


def _stack_vector(items: list[dict[str, Any]], key: str, expected_dim: int) -> np.ndarray:
    vectors = []
    for frame_index, item in enumerate(items):
        if key not in item:
            raise KeyError(f"frame {frame_index} is missing '{key}'")
        vector = _to_numpy(item[key]).astype(np.float32).reshape(-1)
        if vector.shape != (expected_dim,):
            raise ValueError(
                f"frame {frame_index} '{key}' has shape {vector.shape}; expected ({expected_dim},)"
            )
        vectors.append(vector)
    return np.stack(vectors, axis=0)


def _canonicalize_arm(
    actions: np.ndarray,
    states: np.ndarray,
    action_spec: ArmSignalContract,
    state_spec: ArmSignalContract,
) -> CanonicalArmTrajectory:
    return CanonicalArmTrajectory(
        action_position=_select(actions, action_spec.position),
        action_orientation=_select(actions, action_spec.orientation),
        action_gripper=_select(actions, action_spec.gripper),
        joint_position=_select(states, state_spec.joint_position),
        state_gripper=_select(states, state_spec.gripper),
    )


def _select(array: np.ndarray, indices: tuple[int, ...] | None) -> np.ndarray | None:
    return None if indices is None else array[:, indices]


def _timestamps(items: list[dict[str, Any]], contract: TrajectoryContract) -> np.ndarray:
    key = contract.sampling.timestamp_key
    frequency_hz = contract.sampling.frequency_hz
    values = [
        float(item[key]) if key in item and item[key] is not None else index / frequency_hz
        for index, item in enumerate(items)
    ]
    return np.asarray(values, dtype=np.float64)


def _to_numpy(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().numpy()
    return np.asarray(value)