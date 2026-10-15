"""Qwen-VL aligned image canonicalization.

Qwen2.5-VL and Qwen3-VL both resize with factor 28 and a min/max pixel budget.
We apply that resize once, save RGB JPEG/PNG, and freeze the file so teacher
logits and student distillation see the same pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, ImageOps

# Matches qwen_vl_utils.vision_process defaults used by Qwen2.5-VL.
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28  # 3,136
# 1280 visual tokens is the usual inference budget (~1.0 MP).
MAX_PIXELS = 1280 * 28 * 28  # 1,003,520
MAX_RATIO = 200
JPEG_QUALITY = 92


@dataclass(frozen=True)
class CanonicalImage:
    image: Image.Image
    orig_size: tuple[int, int]
    size: tuple[int, int]
    mode_in: str
    format_out: str  # "JPEG" or "PNG"


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Same contract as Qwen2.5-VL `smart_resize`."""
    if height < 1 or width < 1:
        raise ValueError(f"invalid size {width}x{height}")
    ratio = max(height, width) / min(height, width)
    if ratio > MAX_RATIO:
        raise ValueError(f"aspect ratio {ratio:.1f} exceeds {MAX_RATIO}")

    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _floor_by_factor(int(height / beta), factor))
        w_bar = max(factor, _floor_by_factor(int(width / beta), factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(int(height * beta), factor)
        w_bar = _ceil_by_factor(int(width * beta), factor)
    return h_bar, w_bar


def _to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in {"RGBA", "LA", "PA"}:
        bg = Image.new("RGB", image.size, (255, 255, 255))
        rgba = image.convert("RGBA")
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    if image.mode == "P":
        return image.convert("RGBA").convert("RGB") if "transparency" in image.info else image.convert("RGB")
    if image.mode in {"L", "1", "I", "F", "CMYK", "YCbCr"}:
        return image.convert("RGB")
    return image.convert("RGB")


def load_pil(source: Image.Image | bytes | BinaryIO | str | Path) -> Image.Image:
    if isinstance(source, Image.Image):
        im = source
    elif isinstance(source, (bytes, bytearray)):
        im = Image.open(BytesIO(source))
    elif hasattr(source, "read"):
        im = Image.open(source)  # type: ignore[arg-type]
    else:
        im = Image.open(source)
    try:
        im = ImageOps.exif_transpose(im)
    except Exception:
        pass
    return im


def looks_binary(image: Image.Image, sample: int = 64) -> bool:
    gray = image.convert("L")
    small = np.asarray(gray.resize((sample, sample), Image.Resampling.NEAREST))
    return int(np.unique(small).size) <= 8


def canonicalize_image(
    source: Image.Image | bytes | BinaryIO | str | Path,
    *,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> CanonicalImage:
    """Decode → EXIF-fix → RGB → Qwen smart_resize. Same pixels for teacher and student."""
    raw = load_pil(source)
    mode_in = raw.mode
    orig = raw.size
    rgb = _to_rgb(raw)
    new_h, new_w = smart_resize(
        rgb.height, rgb.width, min_pixels=min_pixels, max_pixels=max_pixels
    )
    if (new_w, new_h) != rgb.size:
        rgb = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
    # PNG for bilevel / gray thermal scans so JPEG does not ring the ink.
    fmt = "PNG" if mode_in in {"1", "L"} or looks_binary(rgb) else "JPEG"
    return CanonicalImage(
        image=rgb,
        orig_size=orig,
        size=rgb.size,
        mode_in=mode_in,
        format_out=fmt,
    )


def save_canonical(canon: CanonicalImage, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if canon.format_out == "PNG":
        out = path.with_suffix(".png")
        canon.image.save(out, "PNG", optimize=True)
    else:
        out = path.with_suffix(".jpg")
        canon.image.save(out, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return out
