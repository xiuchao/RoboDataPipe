# Robot Data Pipeline

Utilities for extracting robot episode keyframes and judging task outcomes with Qwen2.5-VL.

This repo is organized around a simple pipeline:

1. Load a local LeRobot dataset from `datasets.yaml`.
2. Extract keyframes such as `episode_start`, `pre_grasp`, `pre_place`, and `episode_end`.
3. Send selected frames to Qwen-VL.
4. Save JSON and TXT summaries for either one episode or an entire dataset.

DEM can also score proprioceptive demonstration quality with a DEM-owned
adapter and analyzer stack that was distilled from reusable ideas in
`Tools/forge`.

## Main Files

- `datasets.yaml`: local dataset registry and dataset-specific signal settings.
- `dataloader.py`: dataset loading helpers.
- `keyframes.py`: episode indexing and keyframe extraction.
- `extract_keyframes.py`: CLI for extracting keyframes from one episode.
- `qwen_vl_config.py`: prompt templates and few-shot defaults.
- `qwen_vl_qa.py`: frame-based QA for one extracted episode.
- `robo_dataset_pipeline.py`: dataset-level extraction + QA pipeline.
- `online_failure_monitor.py`: online-style failure monitor from streamed state/image records.
- `analyze_quality.py`: DEM-owned episode and dataset quality scoring.
- `export_online_stream.py`: export one offline episode into JSONL plus images for pseudo-online testing.
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

- `outputs/keyframes/`
- `outputs/qwenvl/`
- `outputs/result/`

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
  --out /data/xiuchao/biArm/DEM/outputs/keyframes
```

Run frame QA on one extracted episode:

```bash
python3.12 qwen_vl_qa.py \
  --keyframe-dir /data/xiuchao/biArm/DEM/outputs/keyframes/DSRFM_easy/ep_000 \
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

Analyze trajectory quality for a quick sample:

```bash
python3.12 scripts/analyze_quality.py \
  --dataset DEM_pickplace \
  --sample 10
```

The command writes a timestamped report under `outputs/quality/`. Metrics
include per-arm smoothness, Cartesian efficiency, hesitation, path length,
gripper chatter, timestamp regularity, saturation when bounds are available,
and an overall score from 0 to 10. AgiBot data first passes through DEM's own
robot adapter and canonical trajectory model in `robot_events/adapters/`, so
generic metrics do not depend on raw 16-D indices. Omit `--sample` to analyze
all episodes, or add `--min-score 6` for a nonzero exit status when the
dataset falls below a required score.

Run the online-style failure monitor on a JSONL stream:

```bash
python3.12 scripts/online_failure_monitor.py \
  --dataset DEM_pickplace \
  --input-jsonl /path/to/teleop_stream.jsonl \
  --prompt-mode shelf_placement_after_release \
  --camera observation.images.camera_top \
  --camera observation.images.camera_left \
  --output-json outputs/result/online_alerts.json \
  --output-txt outputs/result/online_events.txt
```

Each JSONL line should be one synchronized sample with action/state arrays and camera image paths, for example:

```json
{
  "episode_index": 0,
  "frame_index": 105,
  "timestamp": 10.5,
  "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
  "observation.state": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8, 0.8],
  "images": {
    "observation.images.camera_top": "/abs/path/top_000105.jpg",
    "observation.images.camera_left": "/abs/path/left_000105.jpg"
  }
}
```

The online monitor keeps a short rolling buffer and emits a compact event log containing:

- `right_arm_start_to_place`
- `gripper_open`
- `gripper_fully_open`
- `object_in_shelf_status`
- `release_retreat_start`

It also emits alerts when `object_in_shelf_status` is `still_held`, `dropped_outside`, `missed_compartment`, or `uncertain`.

To generate a pseudo-online stream from an existing episode for testing:

```bash
python3.12 scripts/export_online_stream.py \
  --dataset DEM_pickplace \
  --episode 0 \
  --camera observation.images.camera_top \
  --camera observation.images.camera_left \
  --output-dir outputs/online_stream_demo \
  --overwrite
```

To replay that stream in a local browser GUI with images plus live event/alert descriptions:

```bash
python3.12 scripts/online_monitor_gui.py \
  --dataset DEM_pickplace \
  --input-jsonl outputs/online_stream_demo/ep_000.jsonl \
  --camera observation.images.camera_top \
  --camera observation.images.camera_left \
  --port 8765
```

Then open `http://127.0.0.1:8765` in a browser. The page starts paused, and `Play` or `Step` will advance the pseudo-online stream while accumulating emitted events and alerts.

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
- `shelf_placement_after_release`: structured JSON judgment for whether the released object is inside the shelf compartment.

For `cylinder_upright`, the prompt is instruction-driven and returns JSON like:

```json
{
  "is_upright": true,
  "confidence": "high"
}
```

For `shelf_placement_after_release`, the default keyframes are `gripper_open` and `gripper_fully_open`, so with top and left cameras Qwen-VL receives 4 images per episode. The prompt returns JSON like:

```json
{
  "placement_status": "inside",
  "confidence": 0.95
}
```

Example dataset run:

```bash
python3.12 robo_dataset_pipeline.py \
  --dataset DEM_pickplace \
  --prompt-mode shelf_placement_after_release \
  --shot-mode zeroshot \
  --keyframe-out outputs/keyframes_all_eps \
  --camera observation.images.camera_top \
  --camera observation.images.camera_left
```

## Notes

- Runtime commands in this repo use `python3.12`.
- The dataset pipeline loads the model once and reuses it across episodes.
- Summary TXT output includes run time and per-episode inference timing.
- Saved example commands are in `user_experiments.sh`.
