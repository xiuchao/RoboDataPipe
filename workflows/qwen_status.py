from __future__ import annotations

import threading
from typing import Any

from robehavior.online_monitor import OnlineMonitorConfig
from vlm.qwen_vl_qa import answer_question_about_image_entries, load_qwen_model


_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


class QwenObjectStatusJudge:
    def __init__(self, config: OnlineMonitorConfig):
        self.config = config
        self.processor: Any | None = None
        self.model: Any | None = None

    def __call__(self, image_entries: list[dict[str, Any]]) -> dict[str, Any]:
        self._ensure_model_loaded()
        return answer_question_about_image_entries(
            image_entries,
            self.config.question,
            keyframe_types=["gripper_open", "gripper_fully_open"],
            model_name=self.config.model_name,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            dtype_name=self.config.dtype_name,
            prompt_mode=self.config.prompt_mode,
            processor=self.processor,
            model=self.model,
        )

    def _ensure_model_loaded(self) -> None:
        if self.processor is not None and self.model is not None:
            return
        cache_key = (self.config.model_name, self.config.dtype_name)
        cached = _MODEL_CACHE.get(cache_key)
        if cached is None:
            with _MODEL_CACHE_LOCK:
                cached = _MODEL_CACHE.get(cache_key)
                if cached is None:
                    cached = load_qwen_model(self.config.model_name, self.config.dtype_name)
                    _MODEL_CACHE[cache_key] = cached
        self.processor, self.model = cached