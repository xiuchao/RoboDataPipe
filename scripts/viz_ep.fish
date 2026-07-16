function viz_ep 
    lerobot-dataset-viz \ 
    --repo-id ases200q2/UR5_RTDE_SpaceMouse_EE_pick_and_place_object_Easy_filtered \ 
    --root /data/xiuchao/biArm/DEM/data_DSRFM/ur5_easy_filtered \ 
    --mode local \ 
    --episode-index $argv[1] 
end

# RS-DFM 
lerobot-dataset-viz \
    --repo-id ases200q2/UR5_RTDE_SpaceMouse_EE_pick_and_place_object_Easy_filtered \
    --root /data/xiuchao/biArm/DEM/data_DSRFM/ur5_easy_filtered \
    --mode local \
    --episode-index 0

# RS-DFM 
lerobot-dataset-viz \
    --repo-id ases200q2/UR5_RTDE_SpaceMouse_EE_v3_20260625-135224_filtered \
    --root /data/xiuchao/biArm/DEM/data_DSRFM/ur5_v3_filtered \
    --mode local \
    --episode-index 0
    
# DEM
lerobot-dataset-viz \
    --repo-id alphabot2/aibot2_2026-07-07_hand_position_pick_and_place \
    --root /data/xiuchao/biArm/DEM/data_DEM/hand_position_pick_and_place/ \
    --mode local \
    --episode-index 0

    