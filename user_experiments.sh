python3.12 extract_keyframes.py \
  --dataset DSRFM_easy \
  --episode 0 \
  --keyframes episode_start \
  --camera observation.images.camera_1 \
  --camera observation.images.camera_2 \
  --out /data/xiuchao/biArm/DEM/out_keyframes


python3.12 qwen_vl_qa.py \
  --keyframe-dir /data/xiuchao/biArm/DEM/out_keyframes/DSRFM_easy/ep_000 \
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

python3.12 robo_dataset_pipeline.py \
  --dataset DSRFM_v3 \
  --keyframes episode_start \
  --episode 0 \
  --prompt-mode cylinder_upright \
  --shot-mode fewshot \
  --camera observation.images.camera_1 \
  --max-new-tokens 120




