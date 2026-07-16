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


# ======================================================
# for DEM_pickplace, after resolving the embodiment issue
python3.12 scripts/extract_keyframes.py \
  --dataset DEM_pickplace \
  --episode 0 \
  --keyframes gripper_close \
  --camera observation.images.camera_top \
  --camera observation.images.camera_left \
  --out /data/xiuchao/biArm/DEM/outputs/keyframes

python3.12 scripts/extract_keyframes.py \
  --dataset DEM_pickplace \
  --episode 0 \
  --keyframes gripper_open \
  --camera observation.images.camera_top \
  --camera observation.images.camera_left \
  --camera observation.images.camera_right \
  --out /data/xiuchao/biArm/DEM/outputs/keyframes