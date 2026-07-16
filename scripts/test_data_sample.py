from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = Path("/data/xiuchao/biArm/DEM/data_DEM/hand_position_pick_and_place")
PARQUET_PATTERN = "data/chunk-*/file-*.parquet"

# AgiBot state/action layout:
# [left arm 7, right arm 7, left gripper 1, right gripper 1]
LEFT_GRIPPER_INDEX = 14
RIGHT_GRIPPER_INDEX = 15

# Detect meaningful action changes
ACTION_CHANGE_THRESHOLD = 0.1

# Check whether state follows action after 0–10 frames
MAX_LAG = 10

# Number of values shown in summaries
MAX_UNIQUE_VALUES_TO_SHOW = 30
MAX_CHANGE_EVENTS_TO_SHOW = 10
CONTEXT_FRAMES_BEFORE = 3
CONTEXT_FRAMES_AFTER = 8


# ============================================================
# Data loading
# ============================================================

def load_agibot_parquet_files(
    dataset_root: Path,
    parquet_pattern: str,
) -> pd.DataFrame:
    """Load all AgiBot parquet files into one DataFrame."""

    parquet_files = sorted(dataset_root.glob(parquet_pattern))

    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found under:\n"
            f"  root={dataset_root}\n"
            f"  pattern={parquet_pattern}"
        )

    print(f"Found {len(parquet_files)} parquet file(s).")

    dataframes = []

    for parquet_path in parquet_files:
        print(f"Loading: {parquet_path}")

        frame = pd.read_parquet(
            parquet_path,
            columns=[
                "observation.state",
                "action",
                "episode_index",
                "frame_index",
            ],
        )

        frame = frame.copy()
        frame["source_file"] = str(parquet_path)

        dataframes.append(frame)

    dataset = pd.concat(dataframes, ignore_index=True)

    print(f"\nLoaded {len(dataset)} total frames.")
    return dataset


# ============================================================
# Feature extraction
# ============================================================

def stack_vector_column(
    dataframe: pd.DataFrame,
    column_name: str,
) -> np.ndarray:
    """Convert a DataFrame column of vectors into a 2D NumPy array."""

    try:
        array = np.stack(dataframe[column_name].to_numpy())
    except Exception as exc:
        raise ValueError(
            f"Could not stack column '{column_name}'. "
            "Check whether every row contains a vector of equal length."
        ) from exc

    if array.ndim != 2:
        raise ValueError(
            f"Expected '{column_name}' to produce a 2D array, "
            f"but got shape {array.shape}."
        )

    return array


def extract_gripper_signals(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Extract left/right gripper states and actions."""

    state_array = stack_vector_column(
        dataframe,
        "observation.state",
    )
    action_array = stack_vector_column(
        dataframe,
        "action",
    )

    required_dimension = max(
        LEFT_GRIPPER_INDEX,
        RIGHT_GRIPPER_INDEX,
    ) + 1

    if state_array.shape[1] < required_dimension:
        raise ValueError(
            f"observation.state dimension is {state_array.shape[1]}, "
            f"but index {required_dimension - 1} is required."
        )

    if action_array.shape[1] < required_dimension:
        raise ValueError(
            f"action dimension is {action_array.shape[1]}, "
            f"but index {required_dimension - 1} is required."
        )

    signals = dataframe[
        [
            "episode_index",
            "frame_index",
            "source_file",
        ]
    ].copy()

    signals["left_gripper_state"] = (
        state_array[:, LEFT_GRIPPER_INDEX]
    )
    signals["right_gripper_state"] = (
        state_array[:, RIGHT_GRIPPER_INDEX]
    )

    signals["left_gripper_action"] = (
        action_array[:, LEFT_GRIPPER_INDEX]
    )
    signals["right_gripper_action"] = (
        action_array[:, RIGHT_GRIPPER_INDEX]
    )

    return signals


# ============================================================
# Signal summaries
# ============================================================

def print_signal_summary(
    signal_name: str,
    values: pd.Series,
) -> None:
    """Print basic statistics for one gripper signal."""

    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    ).dropna()

    if numeric_values.empty:
        print(f"\n{signal_name}: no valid values")
        return

    rounded_unique = np.unique(
        np.round(numeric_values.to_numpy(), 4)
    )

    print(f"\n{signal_name}")
    print("-" * len(signal_name))
    print(f"count:  {len(numeric_values)}")
    print(f"min:    {numeric_values.min():.6f}")
    print(f"max:    {numeric_values.max():.6f}")
    print(f"mean:   {numeric_values.mean():.6f}")
    print(f"std:    {numeric_values.std():.6f}")

    quantiles = numeric_values.quantile(
        [0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0]
    )

    print("quantiles:")
    for quantile, value in quantiles.items():
        print(f"  q={quantile:>4.2f}: {value:.6f}")

    print(
        f"unique rounded values "
        f"(first {MAX_UNIQUE_VALUES_TO_SHOW}):"
    )
    print(
        rounded_unique[
            :MAX_UNIQUE_VALUES_TO_SHOW
        ]
    )

    if len(rounded_unique) > MAX_UNIQUE_VALUES_TO_SHOW:
        print(
            f"... total rounded unique values: "
            f"{len(rounded_unique)}"
        )


def summarize_all_signals(
    signals: pd.DataFrame,
) -> None:
    """Print summaries for all gripper state/action signals."""

    signal_names = [
        "left_gripper_state",
        "right_gripper_state",
        "left_gripper_action",
        "right_gripper_action",
    ]

    for signal_name in signal_names:
        print_signal_summary(
            signal_name,
            signals[signal_name],
        )


# ============================================================
# Per-episode analysis
# ============================================================

def get_episode_signals(
    signals: pd.DataFrame,
    episode_index: int,
) -> pd.DataFrame:
    """Return one episode ordered by frame index."""

    episode = signals[
        signals["episode_index"] == episode_index
    ].copy()

    if episode.empty:
        available = sorted(
            signals["episode_index"].unique().tolist()
        )
        raise ValueError(
            f"Episode {episode_index} was not found. "
            f"Available episodes include: {available[:20]}"
        )

    episode = episode.sort_values(
        "frame_index"
    ).reset_index(drop=True)

    return episode


def find_action_change_positions(
    episode: pd.DataFrame,
    action_column: str,
    threshold: float = ACTION_CHANGE_THRESHOLD,
) -> np.ndarray:
    """
    Return row positions where consecutive action values
    differ by more than the threshold.
    """

    action_values = episode[action_column].to_numpy(
        dtype=np.float64
    )

    if len(action_values) < 2:
        return np.array([], dtype=np.int64)

    action_difference = np.abs(
        np.diff(action_values)
    )

    # +1 maps diff index to the later frame
    return np.where(
        action_difference > threshold
    )[0] + 1


def print_action_change_context(
    episode: pd.DataFrame,
    side: str,
) -> None:
    """Print state/action values around action transition frames."""

    state_column = f"{side}_gripper_state"
    action_column = f"{side}_gripper_action"

    change_positions = find_action_change_positions(
        episode,
        action_column,
    )

    print(
        f"\n{side.upper()} gripper action changes: "
        f"{len(change_positions)}"
    )

    if len(change_positions) == 0:
        print(
            f"No changes larger than "
            f"{ACTION_CHANGE_THRESHOLD} detected."
        )
        return

    for event_number, position in enumerate(
        change_positions[:MAX_CHANGE_EVENTS_TO_SHOW],
        start=1,
    ):
        start_position = max(
            0,
            position - CONTEXT_FRAMES_BEFORE,
        )
        end_position = min(
            len(episode),
            position + CONTEXT_FRAMES_AFTER,
        )

        changed_frame_index = episode.iloc[
            position
        ]["frame_index"]

        print(
            f"\nChange event {event_number}: "
            f"around frame_index={changed_frame_index}"
        )
        print(
            "row_pos  frame_index      action       state"
        )

        for row_position in range(
            start_position,
            end_position,
        ):
            row = episode.iloc[row_position]

            marker = (
                " <-- change"
                if row_position == position
                else ""
            )

            print(
                f"{row_position:7d}  "
                f"{int(row['frame_index']):11d}  "
                f"{row[action_column]:10.6f}  "
                f"{row[state_column]:10.6f}"
                f"{marker}"
            )


# ============================================================
# Lagged action-state correlation
# ============================================================

def calculate_lagged_correlations(
    episode: pd.DataFrame,
    side: str,
    max_lag: int = MAX_LAG,
) -> pd.DataFrame:
    """
    Calculate correlations between action[t] and state[t + lag].

    Positive lag means:
        action is issued first,
        state is observed several frames later.
    """

    action_column = f"{side}_gripper_action"
    state_column = f"{side}_gripper_state"

    action = episode[action_column].astype(float)
    state = episode[state_column].astype(float)

    results = []

    for lag in range(max_lag + 1):
        # action[t] compared with state[t + lag]
        shifted_state = state.shift(-lag)

        valid_mask = (
            action.notna()
            & shifted_state.notna()
        )

        valid_action = action[valid_mask]
        valid_state = shifted_state[valid_mask]

        if (
            len(valid_action) < 3
            or valid_action.nunique() < 2
            or valid_state.nunique() < 2
        ):
            correlation = np.nan
        else:
            correlation = valid_action.corr(
                valid_state
            )

        results.append(
            {
                "side": side,
                "lag_frames": lag,
                "correlation": correlation,
                "num_samples": int(
                    valid_mask.sum()
                ),
            }
        )

    return pd.DataFrame(results)


def print_lagged_correlations(
    episode: pd.DataFrame,
    side: str,
) -> None:
    """Print action-to-future-state lag correlation."""

    correlations = calculate_lagged_correlations(
        episode,
        side,
    )

    print(
        f"\n{side.upper()} action[t] vs "
        f"state[t + lag]"
    )
    print(correlations.to_string(index=False))

    valid_results = correlations.dropna(
        subset=["correlation"]
    )

    if valid_results.empty:
        print(
            "No meaningful correlation could be calculated. "
            "The signal may be constant."
        )
        return

    best_row = valid_results.iloc[
        valid_results["correlation"].abs().argmax()
    ]

    print(
        "\nBest absolute correlation:"
        f" lag={int(best_row['lag_frames'])} frames,"
        f" corr={best_row['correlation']:.4f}"
    )


# ============================================================
# Optional approximate scale check
# ============================================================

def summarize_state_by_action(
    episode: pd.DataFrame,
    side: str,
    delay_frames: int = 9,
) -> None:
    """Compare action[t] with state[t + delay_frames]."""

    action_column = f"{side}_gripper_action"
    state_column = f"{side}_gripper_state"

    aligned = pd.DataFrame(
        {
            "action": episode[action_column],
            "future_state": episode[state_column].shift(-delay_frames),
        }
    ).dropna()

    if aligned["action"].nunique() < 2:
        print(f"\n{side.upper()}: action is constant.")
        return

    print(
        f"\n{side.upper()} state grouped by action "
        f"with delay={delay_frames} frames"
    )

    summary = aligned.groupby("action")["future_state"].agg(
        count="count",
        mean="mean",
        median="median",
        min="min",
        max="max",
        std="std",
    )
    print(summary)

# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    
    dataset = load_agibot_parquet_files(
        dataset_root=DATASET_ROOT,
        parquet_pattern=PARQUET_PATTERN,
    )

    gripper_signals = extract_gripper_signals(
        dataset
    )

    # Overall dataset statistics
    print("\n" + "=" * 70)
    print("OVERALL GRIPPER STATISTICS")
    print("=" * 70)

    summarize_all_signals(
        gripper_signals
    )

    # Change this to the episode you want to inspect
    episode_index = 0

    episode = get_episode_signals(
        gripper_signals,
        episode_index=episode_index,
    )

    print("\n" + "=" * 70)
    print(f"EPISODE {episode_index} ANALYSIS")
    print("=" * 70)
    print(f"Number of frames: {len(episode)}")

    # Print summaries for this episode
    summarize_all_signals(
        episode
    )

    # Detect open/close transitions
    print_action_change_context(
        episode,
        side="left",
    )
    print_action_change_context(
        episode,
        side="right",
    )

    # Check control delay
    print_lagged_correlations(
        episode,
        side="left",
    )
    print_lagged_correlations(
        episode,
        side="right",
    )

    # Approximate possible state/action scaling
    summarize_state_by_action(episode,side="right", delay_frames=9)

    # Save extracted signals for easier manual inspection
    output_path = (
        DATASET_ROOT
        / f"gripper_signals_episode_{episode_index}.csv"
    )

    episode.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved episode gripper signals to:\n"
        f"{output_path}"
    )