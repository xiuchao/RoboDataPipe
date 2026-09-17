python3.12 extract_keyframes.py \
  --dataset DSRFM_easy \
  --episode 0 \
  --keyframes episode_start \
  --camera observation.images.camera_1 \
  --camera observation.images.camera_2 \
  --out /data/xiuchao/biArm/DEM/outputs/keyframes


python3.12 qwen_vl_qa.py \
  --keyframe-dir /data/xiuchao/biArm/DEM/outputs/keyframes/DSRFM_easy/ep_000 \
  --camera observation.images.camera_1 \
  --question "is the cylindrical object upstraight? Answer with yes or no." 


# for selected episodes
python3.12 robo_dataset_pipeline.py \
  --dataset DSRFM_easy \
  --episode 0 \
  --keyframes episode_start \
  --prompt-mode cylinder_upright \
  --shot-mode zeroshot \
  --camera observation.images.camera_1 \
  --max-new-tokens 120


# fewshot with default demos from data_anno
python3.12 robo_dataset_pipeline.py \
  --dataset DSRFM_easy \
  --keyframes episode_start \
  --prompt-mode cylinder_upright \
  --shot-mode fewshot \
  --camera observation.images.camera_1 \
  --max-new-tokens 120

python3.12 robo_dataset_pipeline.py \
  --dataset DSRFM_v3 \
  --keyframes episode_start \
  --prompt-mode cylinder_upright \
  --shot-mode fewshot \
  --camera observation.images.camera_1 \
  --max-new-tokens 120


# = W2 July  ====================================================
# for DEM_pickplace, after resolving the embodiment issue
python3.12 scripts/extract_keyframes.py \
  --dataset DEM_pickplace \
  --episode 0 \
  --keyframes gripper_close \
  --camera observation.images.camera_top \
  --camera observation.images.camera_left \
  --out /data/xiuchao/biArm/DEM/outputs/keyframes

# should use action to extract keyframes
python3.12 scripts/extract_keyframes.py \
  --dataset DEM_pickplace \
  --episode 0 \
  --keyframes gripper_close \
  --camera observation.images.camera_top \
  --camera observation.images.camera_left \
  --signal-source action \
  --out /data/xiuchao/biArm/DEM/outputs/keyframes_action_compare



# = W3 July ==================================================
# fully open status
python3.12 scripts/extract_keyframes.py \
  --dataset DEM_pickplace \
  --keyframes gripper_open gripper_fully_open \
  --camera observation.images.camera_top \
  --camera observation.images.camera_left \
  --signal-source action \
  --release-gripper-source observation.state \
  --out outputs/keyframes_all_eps

# qwen-vl for all episodes
python3.12 scripts/robo_dataset_pipeline.py \
  --dataset DEM_pickplace \
  --prompt-mode shelf_placement_after_release \
  --shot-mode zeroshot \
  --keyframe-out outputs/keyframes_all_eps \
  --camera observation.images.camera_top \
  --camera observation.images.camera_left

# online stream export for all episodes
python3.12 scripts/export_online_stream.py \
  --dataset DEM_pickplace \
  --camera observation.images.camera_top \
  --camera observation.images.camera_left \
  --output-dir outputs/online_stream_demo \
  --overwrite

python3.12 scripts/online_failure_monitor.py \
  --dataset DEM_pickplace \
  --input-jsonl outputs/online_stream_demo/ep_000.jsonl \
  --camera observation.images.camera_top \
  --camera observation.images.camera_left \
  --output-json outputs/result/online_alerts_ep000.json


# = W4 July ============================================

# online event log generation
python3.12 scripts/online_failure_monitor.py \
  --dataset DEM_pickplace \
  --input-jsonl outputs/online_stream_demo/ep_000.jsonl \
  --camera observation.images.camera_top \
  --camera observation.images.camera_left \
  --output-json outputs/result/online_events_ep000.json \
  --output-txt outputs/result/online_events_ep000.txt

# online event log generation with refined failure categories
python3.12 scripts/online_failure_monitor.py \
  --dataset DEM_pickplace \
  --input-jsonl outputs/online_stream_demo/ep_000.jsonl \
  --camera observation.images.camera_top \
  --camera observation.images.camera_left \
  --output-json outputs/result/online_events_ep000_refined.json \
  --output-txt outputs/result/online_events_ep000_refined.txt

# online replay GUI on 8765
python3.12 scripts/online_monitor_gui.py \
  --dataset DEM_pickplace \
  --input-jsonl outputs/online_stream_demo/ep_000.jsonl \
  --camera observation.images.camera_top \
  --camera observation.images.camera_left \
  --port 8765

# online replay GUI on 8765 for all exported episodes
python3.12 scripts/online_monitor_gui.py \
  --dataset DEM_pickplace \
  --input-dir outputs/online_stream_demo \
  --camera observation.images.camera_top \
  --camera observation.images.camera_left \
  --port 8768

