# DEM_pickplace Dataset Quality Report

- Dataset: `/data/xiuchao/biArm/DEM/data_DEM/hand_position_pick_and_place`
- Scope: All 68 episodes
- Generated: `2026-09-23T06:02:32.996667+00:00`
- Valid episodes: `68/68`
- Scored episodes: `68/68`
- Dataset score: `8.03/10`

## Metric Summary

| Metric | Count | Mean | Median | Worst episode | Best episode |
| --- | ---: | ---: | ---: | --- | --- |
| Overall score | 68 | 8.03 | 8.10 | `episode_005` (6.71) | `episode_017` (8.90) |
| Translation smoothness | 68 | -9.89 | -9.82 | `episode_000` (-13.47) | `episode_038` (-7.66) |
| Joint smoothness | 68 | -8.48 | -8.37 | `episode_000` (-11.27) | `episode_033` (-6.36) |
| Trajectory efficiency | 68 | 0.65 | 0.66 | `episode_005` (0.39) | `episode_017` (0.85) |
| Hesitation fraction | 68 | 0.11 | 0.11 | `episode_005` (0.19) | `episode_050` (0.03) |

## Diagnostic Distribution

Diagnostics describe motion and sampling distributions and do not contribute to `overall_score`. Low or high values are context for review, not automatically better or worse.

| Metric | Count | Missing | Median | P10 | P90 | Minimum | Maximum |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Joint path length | 68 | 0 | 3.75 | 2.89 | 4.74 | `episode_025` (2.27) | `episode_000` (5.85) |
| Translation path length | 68 | 0 | 0.65 | 0.51 | 1.00 | `episode_038` (0.43) | `episode_000` (1.23) |
| Rotation path length | 68 | 0 | 2.46 | 1.69 | 4.12 | `episode_027` (1.25) | `episode_000` (5.52) |
| Angular speed mean | 68 | 0 | 0.37 | 0.28 | 0.46 | `episode_027` (0.22) | `episode_066` (0.56) |
| Angular acceleration RMS | 68 | 0 | 4.77 | 3.22 | 6.83 | `episode_046` (2.19) | `episode_040` (9.35) |
| Timestamp jitter ratio | 68 | 0 | 0.00 | 0.00 | 0.00 | `episode_033` (0.00) | `episode_000` (0.00) |

### Diagnostic Distribution Tails

Episodes outside P10-P90 are listed for comparison; they are not automatic quality failures.

- Joint path length: low `episode_016, episode_024, episode_025, episode_027, episode_030, episode_031`; high `episode_000, episode_004, episode_005, episode_008, episode_051, episode_063, episode_066`
- Translation path length: low `episode_020, episode_025, episode_027, episode_034, episode_038, episode_053, episode_055`; high `episode_000, episode_004, episode_005, episode_008, episode_011, episode_013, episode_066`
- Rotation path length: low `episode_024, episode_025, episode_027, episode_030, episode_034, episode_035, episode_038`; high `episode_000, episode_004, episode_005, episode_008, episode_011, episode_019, episode_066`
- Angular speed mean: low `episode_012, episode_024, episode_025, episode_027, episode_030, episode_035, episode_038`; high `episode_004, episode_008, episode_040, episode_062, episode_063, episode_066`
- Angular acceleration RMS: low `episode_012, episode_027, episode_035, episode_038, episode_046, episode_050, episode_064`; high `episode_004, episode_005, episode_006, episode_008, episode_015, episode_040, episode_062`
- Timestamp jitter ratio: low `none`; high `none`

## Automatic Flags

No episodes crossed the configured absolute flag thresholds.

## Manual Review Priority

1. `episode_005`: score `6.71`
2. `episode_000`: score `6.78`
3. `episode_013`: score `7.26`
