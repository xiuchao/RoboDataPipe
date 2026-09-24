from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from project_paths import PROJECT_ROOT


ActionSpace = Literal["end_effector", "joint"]
ActionMode = Literal["absolute", "delta"]


def _indices(value: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list of indices")
    indices = tuple(int(index) for index in value)
    if len(set(indices)) != len(indices) or min(indices) < 0:
        raise ValueError(f"{name} must contain unique non-negative indices")
    return indices


@dataclass(frozen=True)
class RotationContract:
    representation: Literal["quaternion", "rotation_vector"]
    order: Literal["xyzw", "wxyz"] | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RotationContract":
        representation = data.get("representation")
        order = data.get("order")
        if representation not in {"quaternion", "rotation_vector"}:
            raise ValueError("rotation.representation must be 'quaternion' or 'rotation_vector'")
        if representation == "quaternion" and order not in {"xyzw", "wxyz"}:
            raise ValueError("rotation.order must be 'xyzw' or 'wxyz'")
        if representation == "rotation_vector":
            order = None
        return cls(representation=representation, order=order)


@dataclass(frozen=True)
class ArmSignalContract:
    position: tuple[int, ...] | None = None
    orientation: tuple[int, ...] | None = None
    gripper: tuple[int, ...] | None = None
    joint_position: tuple[int, ...] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, name: str) -> "ArmSignalContract":
        def optional_indices(field: str) -> tuple[int, ...] | None:
            value = data.get(field)
            return None if value is None else _indices(value, name=f"{name}.{field}")

        return cls(
            position=optional_indices("position"),
            orientation=optional_indices("orientation"),
            gripper=optional_indices("gripper"),
            joint_position=optional_indices("joint_position"),
        )

    def all_indices(self) -> tuple[int, ...]:
        fields = (self.position, self.orientation, self.gripper, self.joint_position)
        return tuple(index for indices in fields if indices is not None for index in indices)


@dataclass(frozen=True)
class VectorContract:
    key: str
    dim: int
    arms: dict[str, ArmSignalContract]

    def validate(self, *, name: str) -> None:
        if self.dim <= 0:
            raise ValueError(f"{name}.dim must be positive")
        claimed: dict[int, str] = {}
        for arm_name, arm in self.arms.items():
            for index in arm.all_indices():
                if index >= self.dim:
                    raise ValueError(f"{name}.arms.{arm_name} index {index} exceeds dim {self.dim}")
                if index in claimed:
                    raise ValueError(
                        f"{name} index {index} is assigned to both {claimed[index]} and {arm_name}"
                    )
                claimed[index] = arm_name


@dataclass(frozen=True)
class ActionContract(VectorContract):
    space: ActionSpace
    mode: ActionMode
    coordinate_frame: str
    position_unit: str | None
    rotation: RotationContract | None


@dataclass(frozen=True)
class StateContract(VectorContract):
    joint_position_unit: str | None


@dataclass(frozen=True)
class SamplingContract:
    frequency_hz: float
    timestamp_key: str


@dataclass(frozen=True)
class GripperSignalContract:
    representation: str
    open_value: float
    closed_value: float


@dataclass(frozen=True)
class ActivityThresholds:
    translation_speed: float
    rotation_speed: float
    joint_speed: float
    gripper_change: float


@dataclass(frozen=True)
class TrajectoryContract:
    version: int
    sampling: SamplingContract
    action: ActionContract
    state: StateContract
    gripper_action: GripperSignalContract
    gripper_state: GripperSignalContract
    activity_thresholds: ActivityThresholds

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrajectoryContract":
        sampling_data = data["sampling"]
        action_data = data["action"]
        state_data = data["state"]
        gripper_data = data["gripper"]
        activity_data = data.get("quality", {}).get("activity_thresholds", {})

        action_arms = {
            name: ArmSignalContract.from_dict(value, name=f"action.arms.{name}")
            for name, value in action_data["arms"].items()
        }
        state_arms = {
            name: ArmSignalContract.from_dict(value, name=f"state.arms.{name}")
            for name, value in state_data["arms"].items()
        }
        if set(action_arms) != set(state_arms):
            raise ValueError("action.arms and state.arms must define the same arm names")

        rotation_data = action_data.get("rotation")
        contract = cls(
            version=int(data["version"]),
            sampling=SamplingContract(
                frequency_hz=float(sampling_data["frequency_hz"]),
                timestamp_key=str(sampling_data.get("timestamp_key", "timestamp")),
            ),
            action=ActionContract(
                key=str(action_data.get("key", "action")),
                dim=int(action_data["dim"]),
                arms=action_arms,
                space=action_data["space"],
                mode=action_data["mode"],
                coordinate_frame=str(action_data["coordinate_frame"]),
                position_unit=action_data.get("position_unit"),
                rotation=(
                    RotationContract.from_dict(rotation_data)
                    if rotation_data is not None
                    else None
                ),
            ),
            state=StateContract(
                key=str(state_data.get("key", "observation.state")),
                dim=int(state_data["dim"]),
                arms=state_arms,
                joint_position_unit=state_data.get("joint_position_unit"),
            ),
            gripper_action=_gripper_contract(gripper_data["action"]),
            gripper_state=_gripper_contract(gripper_data["state"]),
            activity_thresholds=ActivityThresholds(
                translation_speed=float(activity_data.get("translation_speed", 0.005)),
                rotation_speed=float(activity_data.get("rotation_speed", 0.05)),
                joint_speed=float(activity_data.get("joint_speed", 0.05)),
                gripper_change=float(activity_data.get("gripper_change", 0.01)),
            ),
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        if self.version != 1:
            raise ValueError(f"unsupported trajectory contract version {self.version}")
        if self.sampling.frequency_hz <= 0:
            raise ValueError("sampling.frequency_hz must be positive")
        if min(
            self.activity_thresholds.translation_speed,
            self.activity_thresholds.rotation_speed,
            self.activity_thresholds.joint_speed,
            self.activity_thresholds.gripper_change,
        ) < 0:
            raise ValueError("quality activity thresholds must be non-negative")
        if self.action.space not in {"end_effector", "joint"}:
            raise ValueError(f"unsupported action space '{self.action.space}'")
        if self.action.mode not in {"absolute", "delta"}:
            raise ValueError(f"unsupported action mode '{self.action.mode}'")
        self.action.validate(name="action")
        self.state.validate(name="state")
        for arm_name, arm in self.action.arms.items():
            if arm.position is not None and len(arm.position) != 3:
                raise ValueError(f"action.arms.{arm_name}.position must have 3 indices")
            if arm.orientation is not None:
                expected_orientation_dim = (
                    4
                    if self.action.rotation is not None
                    and self.action.rotation.representation == "quaternion"
                    else 3
                )
                if len(arm.orientation) != expected_orientation_dim:
                    raise ValueError(
                        f"action.arms.{arm_name}.orientation must have "
                        f"{expected_orientation_dim} indices"
                    )


def _gripper_contract(data: dict[str, Any]) -> GripperSignalContract:
    return GripperSignalContract(
        representation=str(data["representation"]),
        open_value=float(data["open_value"]),
        closed_value=float(data["closed_value"]),
    )


def load_trajectory_contract(config: dict[str, Any]) -> TrajectoryContract:
    contract_path = config.get("contract")
    if contract_path is None:
        raise ValueError("dataset config must define a trajectory contract")
    path = Path(contract_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    with open(path) as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"trajectory contract at {path} must be a mapping")
    return TrajectoryContract.from_dict(data)