from __future__ import annotations

import unittest
from unittest.mock import patch

from workflows import dataset_task_evaluation as evaluation


class DatasetTaskEvaluationTest(unittest.TestCase):
    def test_resolved_question_is_passed_to_episode_qa(self) -> None:
        episode_result = {
            "answer": "ok",
            "parsed_answer": {"is_upright": True},
            "inference_seconds": 2.0,
            "inference_seconds_per_frame": 1.0,
            "frame_count": 2,
        }

        with (
            patch.object(evaluation, "load_lerobot_dataset", return_value=(object(), {"root": "/tmp/data"})),
            patch.object(evaluation, "resolve_episode_indices", return_value=[4]),
            patch.object(evaluation, "resolve_question", return_value="resolved question"),
            patch.object(evaluation, "collect_demonstrations", return_value=[]),
            patch.object(evaluation, "load_qwen_model", return_value=("processor", "model")),
            patch.object(evaluation, "resolve_episode_cameras", return_value=(["camera"], None)),
            patch.object(evaluation, "extracted_keyframes_ready", return_value=True),
            patch.object(evaluation, "answer_question_about_keyframes", return_value=episode_result) as answer,
        ):
            result = evaluation.evaluate_dataset_task_outcomes(
                evaluation.DatasetTaskEvaluationConfig(
                    dataset_name="DEM_pickplace",
                    prompt_mode="cylinder_upright",
                    keyframe_types=["episode_start"],
                )
            )

        self.assertEqual(answer.call_args.kwargs["question"], "resolved question")
        self.assertEqual(answer.call_args.kwargs["processor"], "processor")
        self.assertEqual(answer.call_args.kwargs["model"], "model")
        self.assertEqual(result["question"], "resolved question")
        self.assertEqual(result["total_episodes"], 1)
        self.assertEqual(result["episodes"]["ep004"]["selected_cameras"], ["camera"])


if __name__ == "__main__":
    unittest.main()