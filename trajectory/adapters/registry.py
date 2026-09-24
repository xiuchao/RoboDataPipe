from __future__ import annotations

from typing import Any

from trajectory.adapters.agibot2 import Agibot2Adapter
from trajectory.adapters.base import BaseRobotAdapter
from trajectory.adapters.ur5 import UR5Adapter


def build_robot_adapter(config: dict[str, Any]) -> BaseRobotAdapter:
    embodiment = config.get("embodiment", "ur5")
    if embodiment == "ur5":
        return UR5Adapter(config)
    if embodiment in {"aibot2", "agibot2"}:
        return Agibot2Adapter(config)
    raise KeyError(f"Unknown embodiment '{embodiment}'")