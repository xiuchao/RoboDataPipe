# Robot Data Pipeline

Utilities for extracting robot episode keyframes and judging task outcomes with Qwen2.5-VL.

This repo is organized around a simple pipeline:

1. Load a local LeRobot dataset from `datasets.yaml`.
2. Extract keyframes such as `episode_start`, `pre_grasp`, `pre_place`, and `episode_end`.
3. Send selected frames to Qwen-VL.
4. Save JSON and TXT summaries for either one episode or an entire dataset.

## Main Files

- `datasets.yaml`: local dataset registry and dataset-specific signal settings.
- `dataloader.py`: dataset loading helpers.
- `keyframes.py`: episode indexing and keyframe extraction.
- `extract_keyframes.py`: CLI for extracting keyframes from one episode.
- `qwen_vl_config.py`: prompt templates and few-shot defaults.
- `qwen_vl_qa.py`: frame-based QA for one extracted episode.
- `robo_dataset_pipeline.py`: dataset-level extraction + QA pipeline.
- `user_experiments.sh`: saved example commands.

## Dataset Registry

Current dataset names in `datasets.yaml` include:

- `DSRFM_easy`
- `DSRFM_v3`
- `DEM_handposition`
- `DEM_pickplace`

The large local dataset folders are intentionally ignored by git:

- `data_DEM/`
- `data_DSRFM/`

Generated outputs are also ignored:

- `out_keyframes/`
- `out_qwenvl/`
- `out_result/`

## Few-Shot Annotations

Few-shot upright judgment uses examples under `data_anno/upstraight_labeling/`.

Current defaults are:

- upright example: `data_anno/upstraight_labeling/upstraight.jpg`
- non-upright examples: all images under `data_anno/upstraight_labeling/lying/`

The loader accepts either a single image path or a directory path for demo overrides.

## Quick Start

Extract one episode's keyframes:

```bash
python3.12 extract_keyframes.py \
  --dataset DSRFM_easy \
  --episode 0 \
  --keyframes episode_start \
  --camera observation.images.camera_1 \
  --camera observation.images.camera_2 \
  --out /data/xiuchao/biArm/DEM/out_keyframes
```

Run frame QA on one extracted episode:

```bash
python3.12 qwen_vl_qa.py \
  --keyframe-dir /data/xiuchao/biArm/DEM/out_keyframes/DSRFM_easy/ep_000 \
  --prompt-mode cylinder_upright \
  --shot-mode fewshot \
  --camera observation.images.camera_1 \
  --keyframes episode_start \
  --max-new-tokens 120
```

Run the dataset pipeline for one episode:

```bash
python3.12 robo_dataset_pipeline.py \
  --dataset DSRFM_v3 \
  --episode 0 \
  --keyframes episode_start \
  --prompt-mode cylinder_upright \
  --shot-mode fewshot \
  --camera observation.images.camera_1 \
  --max-new-tokens 120
```

Run the dataset pipeline for all episodes:

```bash
python3.12 robo_dataset_pipeline.py \
  --dataset DSRFM_easy \
  --keyframes episode_start \
  --prompt-mode cylinder_upright \
  --shot-mode fewshot \
  --camera observation.images.camera_1 \
  --max-new-tokens 120
```

## Output Files

`robo_dataset_pipeline.py` can auto-generate output names when `--output-json` and `--output-txt` are omitted.

The generated filenames include:

- dataset name
- keyframe type
- prompt mode
- shot mode
- episode scope such as `ep000` or `all`
- month, day, and time

Typical outputs look like:

- `out_qwenvl/DSRFM_v3/episode_start_cylinder_upright_fewshot_ep000_0715_170843.json`
- `out_result/DSRFM_v3_episode_start_cylinder_upright_fewshot_ep000_0715_170843_summary.txt`

## Prompt Modes

Supported prompt modes:

- `qa`: free-form question answering over selected keyframes.
- `success_judge`: task-level success judgment.
- `cylinder_upright`: structured JSON judgment for upright vs non-upright cylinder placement.

For `cylinder_upright`, the prompt is instruction-driven and returns JSON like:

```json
{
  "is_upright": true,
  "confidence": "high"
}
```

## Notes

- Runtime commands in this repo use `python3.12`.
- The dataset pipeline loads the model once and reuses it across episodes.
- Summary TXT output includes run time and per-episode inference timing.
- Saved example commands are in `user_experiments.sh`.
