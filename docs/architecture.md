# Architecture

The toolkit is centered on a canonical trajectory shared by behavior and
quality analysis. Raw dataset episodes are normalized through an
embodiment-specific adapter and an explicit trajectory contract before any
behavior interpretation or quality metric is computed.

```mermaid
flowchart TD
  CLI["scripts / CLI"] --> REG["datasets.yaml"]
  CLI --> IO["dataset_io<br/>episode loading"]
  REG --> IO
  REG --> ADAPTER["<b>trajectory adapter</b><br/>selected by embodiment"]
  REG --> CONTRACT["trajectory contract<br/>signal semantics and thresholds"]
  IO --> RAW["raw episode frames"]
  RAW --> ADAPTER
  CONTRACT --> ADAPTER
  ADAPTER --> CANON["<b>CanonicalTrajectory</b><br/>per-arm action, state, timestamps"]

  CANON --> BEHAVIOR["robehavior<br/>activity, events, phases, roles"]
  CANON --> QUALITY["<b>quality metrics</b><br/>rules and scoring"]
  BEHAVIOR --> KEYFRAMES["keyframes and online monitoring<br/>services"]
  BEHAVIOR --> QUALITY
  QUALITY --> REPORTS["JSON / Markdown / HTML reports"]

  KEYFRAMES --> TASK["<b>task-outcome evaluation</b>"]
  VLM["optional VLM service"] --> TASK
  TASK --> OUTCOMES["task outcome results"]

  classDef representation fill:#dceaf7,stroke:#3975a8,color:#183b59,stroke-width:2px;
  classDef analysis fill:#dcefeb,stroke:#287271,color:#173f3e,stroke-width:2px;
  classDef service fill:#e8e2f3,stroke:#75609a,color:#382b50,stroke-width:2px;
  classDef output fill:#fff0d6,stroke:#c77b20,color:#5f390d,stroke-width:2px;
  class ADAPTER,CONTRACT,CANON representation;
  class BEHAVIOR,QUALITY analysis;
  class KEYFRAMES,VLM service;
  class REPORTS,TASK,OUTCOMES output;
```

Blue nodes are trajectory representations. Teal nodes are analysis stages.
Purple nodes are services. Amber nodes are user-facing outputs or
result-producing stages.

## Module Boundaries

- `trajectory/` owns embodiment selection, contract parsing, and conversion of
  raw arrays into `CanonicalTrajectory`.
- `robehavior/` owns behavioral inference, including moving-arm activity,
  phases, roles, keyframe selection, and online monitoring.
- `quality/` consumes the canonical trajectory plus behavior context to produce
  learning metrics, diagnostics, scores, and reports.
- VLM integration is optional and consumes selected keyframes; it is not part
  of trajectory parsing or quality scoring.

Files under `scripts/` are command-line entry points and experiment runners.
Library code does not import from `scripts/`. `robehavior` accepts injected
judges instead of importing a concrete VLM. LeRobot, Hugging Face, and model
imports are delayed until their services are used, so contract, event, and
metric modules can be imported without those optional runtimes installed.

## Related Documentation

- [`robehavior/README.md`](../robehavior/README.md): behavior activity,
  phases, roles, keyframes, and online monitoring.
- [`quality/README.md`](../quality/README.md): quality metrics, scoring rules,
  contextual findings, reports, and metric applicability.
