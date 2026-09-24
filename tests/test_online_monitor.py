from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from robehavior.online_monitor import (
    LoggedEvent,
    OnlineFailureMonitor,
    OnlineMonitorConfig,
    get_dataset_camera_candidates,
    select_available_cameras,
)


class OnlineFailureMonitorTest(unittest.TestCase):
    def test_default_cameras_are_resolved_from_dataset_registry(self) -> None:
        cameras = get_dataset_camera_candidates("DEM_pickplace", "datasets.yaml")

        self.assertEqual(
            cameras[:2],
            ["observation.images.camera_top", "observation.images.camera_front"],
        )

    def test_requested_cameras_are_filtered_to_available_sample_images(self) -> None:
        cameras = select_available_cameras(
            ["observation.images.camera_top", "observation.images.camera_left"],
            [{"images": {"observation.images.camera_left": "left.jpg"}}],
        )

        self.assertEqual(cameras, ["observation.images.camera_left"])

    def test_injected_object_status_judge_records_result_once(self) -> None:
        judge_calls: list[list[dict]] = []

        def judge(image_entries: list[dict]) -> dict:
            judge_calls.append(image_entries)
            return {
                "parsed_answer": {
                    "placement_status": "placed_correctly",
                    "confidence": 0.9,
                }
            }

        monitor = OnlineFailureMonitor(
            OnlineMonitorConfig(dataset_name="DEM_pickplace", cameras=("camera",)),
            object_status_judge=judge,
        )
        monitor.open_event = LoggedEvent("gripper_open", 3, 10, 1.0, 1.0, "test")
        monitor.fully_open_event = LoggedEvent("gripper_fully_open", 3, 11, 1.1, 1.0, "test")

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame.jpg"
            image_path.touch()
            items = [
                {"episode_index": 3, "frame_index": 10, "camera": str(image_path)},
                {"episode_index": 3, "frame_index": 11, "camera": str(image_path)},
            ]

            status_event = monitor._maybe_capture_object_status(items)

            self.assertIsNotNone(status_event)
            assert status_event is not None
            self.assertEqual(status_event["placement_status"], "placed_correctly")
            self.assertEqual(status_event["confidence"], 0.9)
            self.assertEqual([entry["keyframe_type"] for entry in judge_calls[0]], ["gripper_open", "gripper_fully_open"])
            self.assertIsNone(monitor._maybe_capture_object_status(items))
            self.assertEqual(len(judge_calls), 1)


if __name__ == "__main__":
    unittest.main()