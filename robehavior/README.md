# Robot Behavior Analysis

`robehavior` infers observable behavior from a canonical robot trajectory. It
supports offline activity, phase, role, and keyframe analysis, plus a separate
incremental online monitor.

## Modules

| Module | Responsibility |
| --- | --- |
| `activity.py` | Measures per-arm activity from canonical action/state signals, labels quiet/passive/active intensity, builds active intervals, and infers the moving arm. It does not assign task phases or arm roles. |
| `events.py` | Defines the lightweight `Event` and `EventTimeline` data structures used to represent detected behavior events. It does not detect events itself. |
| `phases.py` | Segments each arm into phases such as `grasp`, `transport`, `release`, `retreat`, `manipulate`, and `idle`, then converts phase boundaries into event timelines. |
| `roles.py` | Infers per-arm roles (`actor`, `support`, `passive`, or `idle`) from activity and phase timelines. |
| `profile.py` | Composes activity analysis, phase segmentation, and role inference into one offline `BehaviorProfile`. |
| `keyframes.py` | Consumes phase-boundary events, resolves keyframe specifications, applies image-selection policies, and saves keyframe images and JSON metadata. It does not independently detect phases. |
| `online_monitor.py` | Runs the independent streaming state machine, including incremental command, fully-open, retreat, image-buffer, and alert handling. |
| `registry.py` | Resolves episode cameras from the inferred moving arm and the embodiment adapter. |
| `__init__.py` | Exposes the supported package-level activity, phase, role, and behavior-profile APIs. |

## Analysis Paths

Offline analysis follows this path:

```text
CanonicalTrajectory -> activity -> phases -> phase events -> keyframes
                                  \-> roles
```

Streaming analysis is intentionally separate:

```text
stream samples -> online monitor -> online events and alerts
```

Offline phases prefer measured gripper state over action commands. For example,
`near_target` means that the measured state is within the contract-defined
tolerance of the calibrated open or closed value; it is not a visual judgment.
Visual confirmation remains an optional VLM or image-evaluation concern.
