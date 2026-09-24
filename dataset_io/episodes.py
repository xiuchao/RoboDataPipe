from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EpisodeSlice:
    episode_index: int
    global_indices: list[int]


def build_episode_index(dataset: Any) -> list[EpisodeSlice]:
    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is not None and "episode_index" in hf_dataset.column_names:
        episode_values = [int(_scalar(value)) for value in hf_dataset["episode_index"]]
        if not episode_values:
            return []

        episode_slices: list[EpisodeSlice] = []
        start = 0
        current_episode = episode_values[0]
        for index, episode_index in enumerate(episode_values[1:], start=1):
            if episode_index != current_episode:
                episode_slices.append(EpisodeSlice(current_episode, list(range(start, index))))
                start = index
                current_episode = episode_index
        episode_slices.append(
            EpisodeSlice(current_episode, list(range(start, len(episode_values))))
        )
        return episode_slices

    groups: dict[int, list[int]] = {}
    for index in range(len(dataset)):
        item = dataset[index]
        episode_index = int(_scalar(item["episode_index"]))
        groups.setdefault(episode_index, []).append(index)
    return [EpisodeSlice(episode, indices) for episode, indices in sorted(groups.items())]


def load_episode_items(dataset: Any, episode_index: int) -> list[dict[str, Any]]:
    episode_slice = next(
        (
            episode_slice
            for episode_slice in build_episode_index(dataset)
            if episode_slice.episode_index == episode_index
        ),
        None,
    )
    if episode_slice is None:
        raise ValueError(f"episode {episode_index} not found")

    hf_dataset = getattr(dataset, "hf_dataset", None)
    if hf_dataset is None:
        return [dataset[index] for index in episode_slice.global_indices]

    batch = hf_dataset.select(episode_slice.global_indices)
    return [
        {column: batch[column][index] for column in batch.column_names}
        for index in range(len(episode_slice.global_indices))
    ]


def _scalar(value: Any) -> Any:
    item = getattr(value, "item", None)
    return item() if callable(item) else value