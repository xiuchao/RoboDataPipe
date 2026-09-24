from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from vlm.qwen_vl_qa import generate_answer, select_records


class _MovableList(list):
    def to(self, _device: str) -> _MovableList:
        return self


class _Processor:
    def apply_chat_template(self, *_args, **_kwargs) -> str:
        return "prompt"

    def __call__(self, **_kwargs) -> dict[str, _MovableList]:
        return {"input_ids": _MovableList([[1, 2]])}

    def batch_decode(self, token_ids, **_kwargs) -> list[str]:
        self.decoded_token_ids = token_ids
        return ["answer"]


class _Model:
    def parameters(self):
        yield types.SimpleNamespace(device="cpu")

    def generate(self, **_kwargs) -> list[list[int]]:
        return [[1, 2, 3]]


class QwenVlQaTest(unittest.TestCase):
    def test_select_records_ignores_missing_types_in_error_message(self) -> None:
        records = [{"keyframe_type": "episode_start"}, {}]

        with self.assertRaisesRegex(ValueError, r"Available: \['episode_start'\]"):
            select_records(records, ["episode_end"])

    def test_generate_answer_accepts_two_or_three_vision_results(self) -> None:
        for vision_result in [(["image"], None), (["image"], None, {"fps": []})]:
            with self.subTest(result_size=len(vision_result)):
                qwen_vl_utils = types.ModuleType("qwen_vl_utils")
                qwen_vl_utils.process_vision_info = lambda _messages, result=vision_result: result
                processor = _Processor()

                with patch.dict(sys.modules, {"qwen_vl_utils": qwen_vl_utils}):
                    answer = generate_answer(
                        [{"role": "user", "content": []}],
                        processor=processor,
                        model=_Model(),
                        max_new_tokens=8,
                        temperature=0.0,
                    )

                self.assertEqual(answer, "answer")
                self.assertEqual(processor.decoded_token_ids, [[3]])


if __name__ == "__main__":
    unittest.main()