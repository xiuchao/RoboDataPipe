# Demonstration Quality Analysis

`quality` evaluates canonical robot trajectories after embodiment adapters and
trajectory contracts have normalized their signals. It separates numerical
measurements, generic scoring rules, task-specific behavior expectations, and
reporting.

## Modules

| Module | Responsibility |
| --- | --- |
| `metrics.py` | Implements pure numerical trajectory metrics, including path length, rotation statistics, log dimensionless jerk, trajectory efficiency, hesitation, gripper chatter, and timestamp quality. It does not assign flags or scores. |
| `rules.py` | Converts learning-quality metrics into the overall score and generic flags such as `jerky`, `inefficient`, and `hesitant`. |
| `contextual_rules.py` | Parses dataset-specific behavior expectations and compares observed phase and role timelines with those expectations to produce structured findings. |
| `evaluator.py` | Runs behavior analysis for one canonical trajectory, selects task-active movement, computes per-arm metrics, applies rules, and aggregates episode and dataset results. |
| `pipeline.py` | Provides dataset-level entry points: loads episodes, builds canonical trajectories through the embodiment adapter, invokes the evaluator, and optionally writes JSON. |
| `results.py` | Defines `TimestampQuality`, `QualityFinding`, `EpisodeQuality`, and `QualityReport`, including JSON serialization and compact output formatting. |
| `review.py` | Converts reports to Markdown and pandas tables, summarizes metric distributions, and ranks suspicious episodes for manual review. |
| `dashboard.py` | Builds the self-contained Plotly HTML dashboard, including score distributions, diagnostics, cross-metric plots, labels, and the review queue. |
| `__init__.py` | Exposes the supported quality analysis, result, review, Markdown, and HTML APIs. |

## Analysis Path

```text
dataset episode -> trajectory adapter -> CanonicalTrajectory
                                         |-> robehavior profile
                                         |-> numerical metrics
behavior context + metrics -> rules and contextual findings
                           -> EpisodeQuality -> QualityReport
                           -> JSON / Markdown / HTML / review table
```

`metrics.py` only measures signals. Generic score thresholds belong to
`rules.py`, while task-specific expected behavior belongs to
`contextual_rules.py`. Diagnostic metrics such as path length and timestamp
jitter are reported for review but do not automatically contribute to the
overall score.

## Applicability

Metric applicability is determined by the trajectory contract. For example,
Cartesian path and smoothness metrics require absolute end-effector actions,
while joint metrics require observed joint positions. Delta actions are not
treated as absolute poses, and binary gripper commands are not treated as
measured actuator chatter.

Only transitions whose role is `actor` or `support`, and whose phase represents
task movement, contribute to learning-quality metrics. Passive or idle arms do
not receive an artificial penalty. Dataset-specific behavior expectations are
evaluated separately from the generic score and may produce findings such as
`unexpected_static` or `unexpected_role`.

<h2 id="data-quality-evaluation"><font color="#2563eb">Data Quality Evaluation</font></h2>

Demonstration quality is evaluated in two stages. The stages answer different
questions and should not be collapsed into one score.

<h3><font color="#2563eb">1. Validity and Task Success</font></h3>

An episode must first be usable and successful:

- Validity checks whether required signals, timestamps, dimensions, cameras,
    and trajectory-contract fields are present and internally consistent.
- Task success checks whether the demonstrated task actually reached its
    intended outcome. Proprioceptive signals can establish behavior boundaries,
    but they cannot always prove scene-level outcomes such as whether an object
    is upright or inside a shelf compartment.
- VLM question answering over selected keyframes provides the optional visual
    task-success judgment. This result is a gate, not a motion-quality metric.

An invalid or unsuccessful episode should be rejected or reviewed before its
motion-quality score is used for training selection. A smooth failed
demonstration is still a failed demonstration.

<h3><font color="#2563eb">2. Non-Visual Demonstration Quality</font></h3>

Episodes that pass the first stage are evaluated from action, state, and
timestamp signals. The embodiment adapter and trajectory contract define arm
layout, action space, action mode, rotation representation, gripper semantics,
units, and sampling rate before any metric is computed. This prevents, for
example, treating delta actions as absolute poses or binary gripper commands as
measured actuator chatter.

Behavior analysis then identifies task-active arms and phases. Only transitions
whose role is `actor` or `support`, and whose phase represents task movement,
contribute to learning-quality metrics. Passive or idle arms do not receive an
artificially poor score.

| Metric | Signal and scope | Purpose | In overall score |
| --- | --- | --- | --- |
| `translation_smoothness` | Absolute end-effector action position over task-movement segments | Negative log dimensionless jerk; larger values indicate smoother commanded Cartesian motion | 35% |
| `trajectory_efficiency` | Absolute end-effector action position, computed per movement segment | Compares direct displacement with traveled path | 35% |
| `hesitation_fraction` | Absolute end-effector action position over task movement | Measures the fraction of low-speed movement transitions | 20% |
| `joint_smoothness` | Observed joint state over task-movement segments | Negative log dimensionless jerk of executed joint motion | 10% |
| `gripper_chatter_rate` | Non-binary gripper action over task-active intervals | Detects repeated gripper reversals; binary commands are excluded | No |
| `translation_path_length` | Absolute end-effector action position | Describes commanded Cartesian travel | No |
| `joint_path_length` | Observed joint state | Describes executed joint travel | No |
| `rotation_path_length` | Absolute end-effector action orientation | Describes accumulated angular travel | No |
| `angular_speed_mean` | Absolute end-effector action orientation | Describes average angular speed | No |
| `angular_acceleration_rms` | Absolute end-effector action orientation | Describes rotational acceleration variation | No |
| `jitter_ratio` | Episode timestamps | Describes sampling regularity | No |

The four weighted learning metrics produce an arm score from 0 to 10. All four
must be available; weights are not silently redistributed when a metric is not
applicable. Episode scores average the scored task-active arms, and dataset
scores average scored episodes. Diagnostic metrics remain available for
distribution review and outlier ranking without changing the score.

Dataset-specific behavior expectations are evaluated separately from the
generic score. They compare observed phase and role timelines with explicit
rules in `datasets.yaml` and produce findings such as `unexpected_static` or
`unexpected_role`.

## Behavior Summary Interpretation

The activity classes use these current thresholds:

```text
quiet:   activity_score < 0.5
active:  activity_score >= 1.0 and relative_activity >= 0.5
passive: everything between those cases
```

The 0.5-second smoothing window is converted to frames using episode timestamps.
If timestamps are missing, the adapter constructs them from
`sampling.frequency_hz` in the trajectory contract. For a 10 Hz dataset, the
window is 5 frames; for a 20 Hz dataset, it is 10 frames.

The summary should be read together with `phases` and `roles`. Continuous
activity alone does not prove semantic progress such as approaching an object.
Task phases such as `grasp`, `transport`, `release`, and `retreat` provide that
additional evidence. In the example above, the left arm moves, but its low
relative activity and lack of task phases classify it as passive; the right arm
is the task actor.

Quality assessment is split into generic measurements and task-aware
interpretation. The behavior layer first derives per-arm activity, phases, and
observed roles from the canonical trajectory. A static arm is descriptive, not
a failure, unless the dataset entry declares an explicit expectation. For
example:

```yaml
quality: {behavior_expectations: [{when: {arm: left, phase: transport}, expect: {arm: right, role: support}, severity: error}]}
```

This rule emits an `unexpected_static` or `unexpected_role` finding only during
the matching left-arm transport intervals. Supported phases are `idle`,
`unclassified_motion`, `grasp`, `transport`, `release`, `retreat`, and
`manipulate`; supported roles are `actor`, `support`, `passive`, and `idle`.

### Reference Percentile Calibration

Generic quality flags use fixed engineering heuristics. A known-good reference
set can additionally initialize dataset-relative thresholds: P5 for metrics
where lower values are worse (`translation_smoothness`, `joint_smoothness`,
and `trajectory_efficiency`) and P95 for `hesitation_fraction`, where higher
values are worse. Calibration is computed per arm and requires 20 valid,
task-active values per metric by default.

The reference episode list must contain episodes already accepted by validity
and task-success review. Calibration does not infer success, change the 0-10
score, or replace generic flags. Its `reference_*` flags are added only when a
calibration file is explicitly loaded, so they mean outside this dataset's
known-good distribution rather than universally bad.

To initialize P5/P95 thresholds, put accepted episode IDs in a JSON list or a
newline-delimited text file, then generate a calibration from the analyzed
reference episodes:

```bash
python3.12 scripts/analyze_quality.py \
    --dataset DEM_pickplace \
    --reference-episodes good_episodes.json \
    --write-calibration outputs/quality/DEM_pickplace_calibration.json
```

Apply those dataset-relative thresholds in a later run:

```bash
python3.12 scripts/analyze_quality.py \
    --dataset DEM_pickplace \
    --calibration outputs/quality/DEM_pickplace_calibration.json
```

Use `--lower-quantile`, `--upper-quantile`, and `--min-reference-count` to
override the P5/P95 defaults and minimum sample size.

## Metric Definitions and Applicability

In each `per_arm` result, metrics are grouped by purpose. `learning_quality`
contains metrics useful for demonstration filtering or weighting, while
`diagnostics` contains descriptive motion and sampling measurements. The arm's
`overall_score`, applicability metadata, and role status remain at the arm
level.

| Output | Source | Requirement and interpretation |
| --- | --- | --- |
| `translation_smoothness`, `translation_path_length` | end-effector action position | Requires `action.space: end_effector` and `action.mode: absolute`. These describe command-space motion, not measured execution. |
| `rotation_path_length`, `angular_speed_mean`, `angular_acceleration_rms` | end-effector action orientation | Also requires an absolute action and a declared rotation representation. Quaternion values use the configured component order. |
| `joint_smoothness`, `joint_path_length` | observed joint state | Requires joint positions in `state`; these describe executed joint motion. |
| `trajectory_efficiency`, `hesitation_fraction` | end-effector action position | Computed separately over task movement phases, then aggregated. Phase boundaries prevent approach, transport, and retreat from being treated as one point-to-point path. |
| `gripper_chatter_rate` | gripper action | Not reported for `binary_command`: normal grasp and release commands are intentional state changes, not actuator chatter. |
| `jitter_ratio` | timestamps | Describes sampling regularity and is independent of arm kinematics. |

`translation_smoothness` is the negative logarithm of dimensionless jerk for
the end-effector position signal:

```text
velocity     = gradient(position, dt)
acceleration = gradient(velocity, dt)
jerk         = gradient(acceleration, dt)
jerk_energy  = sum(jerk^2) * dt

translation_smoothness = -log(T^3 / peak_velocity^2 * jerk_energy)
```

Here, `T` is the segment duration and `peak_velocity` is the maximum
translation-speed norm in that segment. Negative values are expected: the
quantity inside the logarithm is commonly greater than one. A larger value
(closer to zero, or positive) indicates smoother motion, while a more negative
value indicates stronger jerk. For example, `-12.96` is smoother than `-25`.
The current generic rule flags values below `-25` as `jerky`.

Smoothness is computed independently for each included movement segment and
then averaged using segment frame counts as weights. At least four position
samples are required; stationary or zero-duration segments do not produce a
value. Because numerical third derivatives amplify sampling noise, smoothness
should be compared only across data with consistent sampling and preprocessing.
`joint_smoothness` uses the same definition on observed joint positions.

Only transitions whose role is `actor` or `support` contribute to task-quality
metrics. Within those intervals, motion metrics use the
`unclassified_motion`, `transport`, `retreat`, and `manipulate` phases;
single-transition `grasp` and `release` events split adjacent motion segments.
An arm containing only `passive` or `idle` roles has no arm score and does not
contribute to the episode score.

`overall_score` is the arm's fixed-weight learning-quality score:

```text
overall_score = 10 * (0.35 * S_translation
                                        + 0.35 * trajectory_efficiency
                                        + 0.20 * (1 - hesitation_fraction)
                                        + 0.10 * S_joint)
```

`S_translation` and `S_joint` map log dimensionless jerk to `[0, 1]` with
`1 / (1 + exp(-(smoothness + 18) / 4))`; larger values indicate smoother
motion. All four components are required. If any component is unavailable, the
arm has no `overall_score` rather than silently redistributing its weight.
Timestamp jitter, path lengths, angular diagnostics, and gripper chatter do not
affect this score.

Scores produced from different action spaces or different available signals
should not be compared unless all four components are available under the same
metric definitions. For contracts with `coordinate_frame: unknown`, geometric
metrics remain useful within one episode if the frame is stable, but absolute
paths should not be compared across robots or datasets.
