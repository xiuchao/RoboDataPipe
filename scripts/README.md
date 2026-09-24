# Scripts

Stable user-facing commands stay at the top level:

- `analyze_quality.py`: score trajectory quality.
- `evaluate_task_outcomes.py`: extract keyframes and evaluate task outcomes with Qwen-VL.
- `visualize_task_outcomes.py`: build an HTML dashboard from a saved task-outcome summary.
- `extract_keyframes.py`: extract behavior keyframes.
- `analyze_monitor_stream.py`: process an online-style JSONL stream and save event results.
- `replay_monitor_stream.py`: replay a monitor stream in a browser UI.

Supporting commands live under `tools/`:

- `export_online_stream.py`: convert offline episodes to replayable JSONL streams.
- `inspect_episode_camera.py`: inspect automatic camera and moving-arm selection.
- `review_episodes.py`: rank and manually review suspicious episodes.
- `inspect_agibot_action_state.py`: inspect raw AgiBot parquet action/state behavior.
- `viz_lerobot.sh`: launch the LeRobot dataset visualizer from the registry.

Saved ad hoc commands and historical experiments live under `experiments/`.
Reusable behavior belongs in `dataset_io/`, `trajectory/`, `robehavior/`,
`quality/`, or `workflows/`; Python code must not import from `scripts/`.