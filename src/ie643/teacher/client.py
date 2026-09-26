"""Call a vLLM OpenAI-compatible server for Qwen3-VL extraction."""

from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    if path.suffix.lower() == ".png":
        mime = "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


class VLLMClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "Qwen/Qwen3-VL-32B-Instruct",
        timeout: int = 180,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.repetition_penalty = repetition_penalty
        self.frequency_penalty = frequency_penalty

    def extract(self, image_path: Path, prompt: str, *, max_tokens: int = 4096) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "frequency_penalty": self.frequency_penalty,
            "response_format": {"type": "json_object"},
            "repetition_detection": {
                "min_pattern_size": 1,
                "max_pattern_size": 8,
                "min_count": 16,
            },
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_url(image_path)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"vLLM request failed: {exc}") from exc
        try:
            return payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected vLLM payload: {payload!r}") from exc
