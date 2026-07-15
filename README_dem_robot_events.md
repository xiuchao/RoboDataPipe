# dem_robot_events

Small utilities for extracting event-centric frames from local LeRobot datasets.

## Example

```bash
python -m dem_robot_events.cli \
  --root /data/xiuchao/biArm/DEM/ur5_easy_filtered \
  --repo-id ases200q2/UR5_RTDE_SpaceMouse_EE_pick_and_place_object_Easy_filtered \
  --event gripper_close \
  --signal-source observation.state \
  --gripper-dim -1 \
  --direction decrease \
  --camera observation.images.camera_1 \
  --camera observation.images.camera_2 \
  --out /data/xiuchao/biArm/DEM/gripper_close_events
```

If close is represented by an increasing gripper value, change:

```bash
--direction increase
```

To test one episode first:

```bash
python -m dem_robot_events.cli \
  --root /data/xiuchao/biArm/DEM/ur5_easy_filtered \
  --repo-id ases200q2/UR5_RTDE_SpaceMouse_EE_pick_and_place_object_Easy_filtered \
  --episode 0 \
  --out ./debug_events
```

## Python API

```python
from dem_robot_events import load_lerobot_dataset, detect_gripper_close, save_event_frames
from dem_robot_events.dataset import build_episode_index
from dem_robot_events.signals import gripper_signal

root = "/data/xiuchao/biArm/DEM/ur5_easy_filtered"
repo_id = "ases200q2/UR5_RTDE_SpaceMouse_EE_pick_and_place_object_Easy_filtered"

ds = load_lerobot_dataset(repo_id=repo_id, root=root)
episodes = build_episode_index(ds)

ep0 = episodes[0]
items = [ds[i] for i in ep0.global_indices]
signal = gripper_signal(items, source="observation.state", dim=-1)
event = detect_gripper_close(signal, direction="decrease")

save_event_frames(
    ep0.episode_index,
    items,
    event,
    cameras=["observation.images.camera_1", "observation.images.camera_2"],
    out_dir="./debug_events",
)
```

## Notes

LeRobot handles aligned image/state/action loading. This package only adds
robot-specific event semantics such as gripper close/open.
