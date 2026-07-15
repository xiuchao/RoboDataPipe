#!/usr/bin/env bash
set -euo pipefail

REGISTRY="${DEM_DATASETS_YAML:-/data/xiuchao/biArm/DEM/datasets.yaml}"

usage() {
    echo "Usage:"
    echo "  $0 list"
    echo "  $0 <dataset_name> <episode_index> [extra lerobot-dataset-viz args...]"
    echo
    echo "Examples:"
    echo "  $0 list"
    echo "  $0 ur5_easy 0"
    echo "  $0 ur5_medium 12"
    echo "  $0 ur5_easy 0 --save 1"
}

if [ $# -lt 1 ]; then
    usage
    exit 1
fi

if [ "$1" = "list" ]; then
    python - "$REGISTRY" <<'PY'
import sys
import yaml
from pathlib import Path

registry_path = Path(sys.argv[1])
cfg = yaml.safe_load(registry_path.read_text())

for name, item in cfg.items():
    print(f"{name:20s}  {item.get('root', '')}")
PY
    exit 0
fi

if [ $# -lt 2 ]; then
    usage
    exit 1
fi

DATASET="$1"
EPISODE="$2"
shift 2

mapfile -t CFG_LINES < <(python - "$REGISTRY" "$DATASET" <<'PY'
import sys
import yaml
from pathlib import Path

registry_path = Path(sys.argv[1])
dataset_name = sys.argv[2]

cfg = yaml.safe_load(registry_path.read_text())

if dataset_name not in cfg:
    print(f"Unknown dataset: {dataset_name}", file=sys.stderr)
    print("Available datasets:", file=sys.stderr)
    for name in cfg:
        print(f"  {name}", file=sys.stderr)
    sys.exit(2)

item = cfg[dataset_name]

repo_id = item["repo_id"]
root = item["root"]

print(repo_id)
print(root)
PY
)

REPO_ID="${CFG_LINES[0]}"
ROOT="${CFG_LINES[1]}"

echo "[viz] dataset:  $DATASET"
echo "[viz] repo_id:  $REPO_ID"
echo "[viz] root:     $ROOT"
echo "[viz] episode:  $EPISODE"
echo

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
lerobot-dataset-viz \
    --repo-id "$REPO_ID" \
    --root "$ROOT" \
    --mode local \
    --episode-index "$EPISODE" \
    "$@"