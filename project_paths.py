from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_REGISTRY_PATH = PROJECT_ROOT / "datasets.yaml"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
KEYFRAME_OUTPUT_DIR = OUTPUTS_DIR / "keyframes"
QWENVL_OUTPUT_DIR = OUTPUTS_DIR / "qwenvl"
RESULT_OUTPUT_DIR = OUTPUTS_DIR / "result"
QUALITY_OUTPUT_DIR = OUTPUTS_DIR / "quality"

UPSTRAIGHT_LABELING_DIR = PROJECT_ROOT / "data_anno" / "upstraight_labeling"
DEFAULT_DEMO_UPRIGHT_PATH = UPSTRAIGHT_LABELING_DIR / "upstraight.jpg"
DEFAULT_DEMO_NON_UPRIGHT_PATH = UPSTRAIGHT_LABELING_DIR / "lying"