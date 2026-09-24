from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from project_paths import DATASET_REGISTRY_PATH


DATASET_REGISTRY = str(DATASET_REGISTRY_PATH)


def load_registry(registry_path: str | Path = DATASET_REGISTRY) -> dict[str, dict[str, Any]]:
    with open(registry_path) as handle:
        registry = yaml.safe_load(handle)
    if not isinstance(registry, dict):
        raise ValueError(f"dataset registry at {registry_path} must be a mapping")
    return registry