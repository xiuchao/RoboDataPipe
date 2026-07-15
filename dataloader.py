from pathlib import Path
import yaml
from huggingface_hub import snapshot_download
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DATASET_REGISTRY = "/data/xiuchao/biArm/DEM/datasets.yaml"
REQUIRED_DIRS = ["data", "meta", "videos"]

DATASET_ALIASES = {
    "ur5_easy": "DSRFM_easy",
}


def is_lerobot_dataset_downloaded(root: str | Path, require_videos: bool = True) -> bool:
    root = Path(root)

    if not root.exists():
        return False

    required = ["data", "meta"]
    if require_videos:
        required.append("videos")

    for name in required:
        if not (root / name).exists():
            return False

    # basic content check
    has_parquet = any((root / "data").glob("chunk-*/*.parquet"))
    has_info = (root / "meta" / "info.json").exists()

    if require_videos:
        has_video = any((root / "videos").glob("**/*.mp4"))
    else:
        has_video = True

    return has_parquet and has_info and has_video


def load_registry(registry_path: str | Path):
    with open(registry_path, "r") as f:
        return yaml.safe_load(f)


def ensure_lerobot_dataset_local(
    name: str,
    registry_path: str | Path = DATASET_REGISTRY,
    *,
    force_download: bool = False,
    require_videos: bool = True,
):
    registry = load_registry(registry_path)

    name = DATASET_ALIASES.get(name, name)

    if name not in registry:
        available = ", ".join(registry.keys())
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")

    cfg = registry[name]
    repo_id = cfg["repo_id"]
    root = Path(cfg["root"])

    downloaded = is_lerobot_dataset_downloaded(root, require_videos=require_videos)

    if force_download or not downloaded:
        root.mkdir(parents=True, exist_ok=True)

        print(f"[download] {name}")
        print(f"  repo_id: {repo_id}")
        print(f"  local_dir: {root}")

        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(root),
            local_dir_use_symlinks=False,  # older hub versions may warn, but OK
        )
    else:
        print(f"[local] {name}: {root}")

    return cfg


def load_lerobot_dataset(
    name: str,
    registry_path: str | Path = DATASET_REGISTRY,
    *,
    force_download: bool = False,
    require_videos: bool = True,
):
    name = DATASET_ALIASES.get(name, name)

    cfg = ensure_lerobot_dataset_local(
        name,
        registry_path,
        force_download=force_download,
        require_videos=require_videos,
    )

    ds = LeRobotDataset(
        repo_id=cfg["repo_id"],
        root=cfg["root"],
    )

    return ds, cfg

if __name__ == "__main__":

    ds, cfg = load_lerobot_dataset("DSRFM_v3") #"DEM_handposition"
    print(len(ds))
    item = ds[0]
    breakpoint()