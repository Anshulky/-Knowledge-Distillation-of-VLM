"""Prompt, image preprocessing and JSON parsing for the Gradio Space.

Everything comes from space/vendor/ (copied from the training repo by sync_vendor.py), so the Space
uses the same hint-free prompt, the same Qwen smart_resize and the same parser as training and evaluation.
"""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

from PIL import Image

VENDOR = Path(__file__).resolve().parent / "vendor"
if not (VENDOR / "ie643").is_dir():
    raise RuntimeError("space/vendor is missing: run `python space/sync_vendor.py` before launching or uploading")
sys.path.insert(0, str(VENDOR))

from ie643.preprocess.image import JPEG_QUALITY, canonicalize_image  # noqa: E402
from ie643.schema import teacher_prompt  # noqa: E402
from ie643.teacher.parse import parse_teacher_output  # noqa: E402

__all__ = ["prepare_image", "teacher_prompt", "parse_output"]


def prepare_image(image: Image.Image) -> tuple[Image.Image, tuple[int, int], tuple[int, int]]:
    """Same pixels the student saw in training: EXIF fix, RGB, smart_resize, then the pack's JPEG/PNG round trip.

    Returns (image, original (w, h), model-side (w, h)).
    """
    canon = canonicalize_image(image)
    buf = BytesIO()
    if canon.format_out == "PNG":
        canon.image.save(buf, "PNG")
    else:
        canon.image.save(buf, "JPEG", quality=JPEG_QUALITY)
    buf.seek(0)
    return Image.open(buf).convert("RGB"), canon.orig_size, canon.size


def parse_output(text: str) -> dict:
    parsed = parse_teacher_output(text, fallback=None)
    if parsed is None:
        raise ValueError("model output is not a JSON object")
    return parsed
