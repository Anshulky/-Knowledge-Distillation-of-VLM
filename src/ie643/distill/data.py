"""Train examples: packed image + teacher token ids + soft labels.

Targets are the teacher's ``token_ids`` (already ending in ``<|im_end|>``), never a
re-tokenisation of ``teacher_raw``. The chat template is the student's, so the prompt
includes its default system prompt. See docs/distillation_plan.md C.1.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ie643.preprocess.image import MAX_PIXELS, MIN_PIXELS
from ie643.rollout.spans import json_leaves
from ie643.schema import teacher_prompt
from ie643.teacher.soft import load_soft

PATCH = 14
PACK_FACTOR = 28


def teacher_pixel_size(width: int, height: int) -> tuple[int, int]:
    """Option B: packed multiples of 28 become multiples of 32 (scale 8/7)."""
    if width % PACK_FACTOR or height % PACK_FACTOR:
        raise ValueError(f"packed size {width}x{height} is not a multiple of {PACK_FACTOR}")
    return width // PACK_FACTOR * 32, height // PACK_FACTOR * 32


def resize_packed_for_teacher(image: Image.Image) -> Image.Image:
    width, height = image.size
    new_w, new_h = teacher_pixel_size(width, height)
    if (new_w, new_h) == (width, height):
        return image
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def configure_processor(processor):
    """Keep the processor from resizing images that packing already sized."""
    image_processor = processor.image_processor
    if hasattr(image_processor, "min_pixels"):
        image_processor.min_pixels = MIN_PIXELS
    if hasattr(image_processor, "max_pixels"):
        image_processor.max_pixels = MAX_PIXELS
    return processor


def image_token_id(processor) -> int:
    tokenizer = processor.tokenizer
    for name in ("<|image_pad|>", "<|vision_pad|>"):
        idx = tokenizer.convert_tokens_to_ids(name)
        unk = getattr(tokenizer, "unk_token_id", None)
        if isinstance(idx, int) and idx != unk:
            return idx
    raise RuntimeError("could not find the image-pad token on this processor")


def chat_text(processor) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": teacher_prompt()}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prompt_overhead(processor, image_id: int) -> int:
    """Non-image tokens in the chat template.

    The probe is the smallest packed size (56x56, exactly ``min_pixels``) so the
    processor does not upscale it. That image is 2x2 visual tokens.
    """
    side = PACK_FACTOR * 2
    probe = Image.new("RGB", (side, side), "white")
    enc = processor(text=[chat_text(processor)], images=[probe], return_tensors="pt")
    n_img = int((enc["input_ids"][0] == image_id).sum())
    expected = (side // PACK_FACTOR) ** 2
    if n_img != expected:
        raise RuntimeError(f"{side}x{side} probe produced {n_img} image tokens, expected {expected}")
    return int(enc["input_ids"].shape[1] - n_img)


def assert_student_grid(grid_thw: torch.Tensor, width: int, height: int) -> None:
    _, gh, gw = (int(x) for x in grid_thw.view(-1)[:3])
    if gh * PATCH != height or gw * PATCH != width:
        raise RuntimeError(
            f"processor resized the packed image: grid {gh}x{gw} x {PATCH}px != {width}x{height}"
        )


def _decode(tokenizer, ids: list[int]) -> str:
    try:
        return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(ids, skip_special_tokens=False)


def token_char_offsets(tokenizer, ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    """Character span of each token inside ``tokenizer.decode(ids)``."""
    if not ids:
        return "", []
    singles = [_decode(tokenizer, [i]) for i in ids]
    joined = "".join(singles)
    full = _decode(tokenizer, ids)
    if joined == full:
        offsets = []
        pos = 0
        for piece in singles:
            offsets.append((pos, pos + len(piece)))
            pos += len(piece)
        return joined, offsets
    ends = [len(_decode(tokenizer, ids[: i + 1])) for i in range(len(ids))]
    offsets = []
    prev = 0
    for end in ends:
        offsets.append((prev, end))
        prev = end
    return full, offsets


def value_mask_for_output(tokenizer, token_ids: list[int], eos_id: int) -> list[int]:
    """1 on output tokens that overlap a non-null JSON leaf. The final EOS stays 0."""
    mask = [0] * len(token_ids)
    body = token_ids[:-1] if token_ids and token_ids[-1] == eos_id else list(token_ids)
    if not body:
        return mask
    text, offsets = token_char_offsets(tokenizer, body)
    for leaf in json_leaves(text):
        if leaf.is_null or leaf.end <= leaf.start:
            continue
        for k, (start, end) in enumerate(offsets):
            if end > leaf.start and start < leaf.end:
                mask[k] = 1
    return mask


def row_probabilities(datasets: list[str], power: float, weighting: bool) -> np.ndarray:
    """P(row) ∝ n_dataset^(power - 1). That is P(dataset) ∝ n^power, then uniform inside it."""
    n = len(datasets)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    if not weighting:
        return np.full(n, 1.0 / n)
    counts = Counter(datasets)
    weights = np.array([counts[d] ** (power - 1.0) for d in datasets], dtype=np.float64)
    total = weights.sum()
    if total <= 0:
        return np.full(n, 1.0 / n)
    return weights / total


def extend_with_output(enc: dict, out_ids: torch.Tensor) -> tuple[dict, int]:
    """Append teacher token ids. Per-token side tensors (mm_token_type_ids, ...) get zeros."""
    n_prompt = int(enc["input_ids"].shape[1])
    out = out_ids.view(1, -1).to(dtype=enc["input_ids"].dtype)
    for key, val in list(enc.items()):
        if key == "input_ids":
            continue
        if torch.is_tensor(val) and val.ndim >= 2 and tuple(val.shape[:2]) == (1, n_prompt):
            enc[key] = torch.cat([val, torch.zeros(1, out.shape[1], dtype=val.dtype)], dim=1)
    enc["input_ids"] = torch.cat([enc["input_ids"], out], dim=1)
    enc["attention_mask"] = torch.ones_like(enc["input_ids"])
    return enc, n_prompt


def encode_image(processor, image: Image.Image, out_ids: torch.Tensor) -> tuple[dict, int]:
    enc = processor(text=[chat_text(processor)], images=[image], return_tensors="pt")
    enc = {k: v for k, v in enc.items()}
    return extend_with_output(enc, out_ids)


@dataclass
class ExampleIndex:
    rec: dict
    soft_path: Path
    n_out: int
    length: int


class DistillDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        manifest: Path,
        label_filter: Path,
        processor,
        *,
        soft_key: str = "teacher_soft",
        max_length: int = 4096,
        drop_flagged: bool = False,
        eos_id: int = 151645,
        teacher_processor=None,
        teacher_resize: str = "B",
        limit: int | None = None,
    ) -> None:
        self.processor = configure_processor(processor)
        self.teacher_processor = configure_processor(teacher_processor) if teacher_processor is not None else None
        self.teacher_resize = teacher_resize
        self.eos_id = eos_id
        self.max_length = max_length
        self.root = manifest.parent
        self.image_id = image_token_id(self.processor)
        self.overhead = prompt_overhead(self.processor, self.image_id)
        self.teacher_image_id = image_token_id(self.teacher_processor) if self.teacher_processor is not None else None

        filt = json.loads(label_filter.read_text(encoding="utf-8"))
        keep = set(filt["keep_uids"])
        flagged = set(filt.get("flagged") or {})
        if drop_flagged:
            keep -= flagged
        rows = []
        with manifest.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec["uid"] in keep:
                    rows.append(rec)
        if limit is not None:
            rows = sorted(rows, key=lambda r: r["uid"])[:limit]
        self.dropped_long: list[str] = []
        self.rows: list[ExampleIndex] = []
        for rec in rows:
            rel = rec.get(soft_key)
            if not rel:
                continue
            path = self.root / rel
            with np.load(path) as soft:
                n_out = int(soft["token_ids"].shape[0])
            n_img = (int(rec["height"]) // PACK_FACTOR) * (int(rec["width"]) // PACK_FACTOR)
            length = self.overhead + n_img + n_out
            if length > max_length:
                self.dropped_long.append(rec["uid"])
                continue
            self.rows.append(ExampleIndex(rec, path, n_out, length))
        self.datasets = [row.rec["dataset"] for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        rec = row.rec
        soft = load_soft(row.soft_path)
        token_ids = [int(x) for x in soft["token_ids"].tolist()]
        if token_ids[-1] != self.eos_id:
            raise RuntimeError(f"{rec['uid']} soft labels do not end in eos {self.eos_id}")
        image = Image.open(self.root / rec["image"]).convert("RGB")
        if image.size != (int(rec["width"]), int(rec["height"])):
            raise RuntimeError(f"{rec['uid']} image is {image.size}, manifest says {rec['width']}x{rec['height']}")
        out_ids = torch.tensor(token_ids, dtype=torch.long)
        enc, n_prompt = encode_image(self.processor, image, out_ids)
        assert_student_grid(enc["image_grid_thw"], image.size[0], image.size[1])
        length = int(enc["input_ids"].shape[1])
        if length > self.max_length:
            raise RuntimeError(f"{rec['uid']} encoded to {length} tokens (max {self.max_length})")
        ids = enc["input_ids"][0]
        labels = ids.clone()
        labels[:n_prompt] = -100
        kd = torch.zeros(length, dtype=torch.float32)
        kd[n_prompt:] = 1
        values = torch.tensor(value_mask_for_output(self.processor.tokenizer, token_ids, self.eos_id), dtype=torch.float32)
        value_mask = torch.zeros(length, dtype=torch.float32)
        value_mask[n_prompt:] = values
        item = {
            "uid": rec["uid"],
            "input_ids": ids,
            "attention_mask": enc["attention_mask"][0],
            "labels": labels,
            "kd_mask": kd,
            "value_mask": value_mask,
            "pixel_values": enc["pixel_values"],
            "image_grid_thw": enc["image_grid_thw"].view(-1),
            "topk_ids": torch.tensor(soft["topk_ids"].astype(np.int64)),
            "topk_logprobs": torch.tensor(soft["topk_logprobs"].astype(np.float32)),
            "n_prompt": n_prompt,
            "n_out": len(token_ids),
        }
        for key, val in enc.items():
            if key in item or key in {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}:
                continue
            if torch.is_tensor(val) and val.ndim >= 1 and int(val.shape[0]) == 1 and int(val.shape[-1] if val.ndim > 1 else val.shape[0]) == length:
                item[key] = val.view(-1) if val.ndim <= 2 else val[0]
        if self.teacher_processor is not None:
            teacher_image = resize_packed_for_teacher(image) if self.teacher_resize == "B" else image
            t_enc, t_prompt = encode_image(self.teacher_processor, teacher_image, out_ids)
            tw, th = teacher_image.size
            _, gh, gw = (int(x) for x in t_enc["image_grid_thw"].view(-1)[:3])
            patch = int(getattr(self.teacher_processor.image_processor, "patch_size", 16))
            merge = int(getattr(self.teacher_processor.image_processor, "merge_size", 2))
            if gh * patch != th or gw * patch != tw:
                raise RuntimeError(f"{rec['uid']} teacher resized the image: grid {gh}x{gw} x {patch}px != {tw}x{th}")
            s_img = int((enc["input_ids"][0] == self.image_id).sum())
            t_img = int((t_enc["input_ids"][0] == self.teacher_image_id).sum())
            if self.teacher_resize == "B" and s_img != t_img:
                raise RuntimeError(f"{rec['uid']} image tokens student {s_img} != teacher {t_img}")
            item["teacher_input_ids"] = t_enc["input_ids"][0]
            item["teacher_attention_mask"] = t_enc["attention_mask"][0]
            item["teacher_pixel_values"] = t_enc["pixel_values"]
            item["teacher_image_grid_thw"] = t_enc["image_grid_thw"].view(-1)
            item["teacher_n_prompt"] = t_prompt
            item["teacher_grid_hw"] = torch.tensor([gh // merge, gw // merge])
        item["student_grid_hw"] = torch.tensor([int(rec["height"]) // PACK_FACTOR, int(rec["width"]) // PACK_FACTOR])
        return item


def collate(batch: list[dict], pad_id: int = 151643) -> dict:
    def pad_stack(key: str, fill: int | float, dtype=None) -> torch.Tensor:
        length = max(int(x[key].shape[0]) for x in batch)
        rows = []
        for item in batch:
            src = item[key]
            out = src.new_full((length, *src.shape[1:]), fill)
            out[: src.shape[0]] = src
            rows.append(out)
        stacked = torch.stack(rows)
        return stacked if dtype is None else stacked.to(dtype)

    out = {
        "uid": [b["uid"] for b in batch],
        "input_ids": pad_stack("input_ids", pad_id),
        "attention_mask": pad_stack("attention_mask", 0),
        "labels": pad_stack("labels", -100),
        "kd_mask": pad_stack("kd_mask", 0),
        "value_mask": pad_stack("value_mask", 0),
        "pixel_values": torch.cat([b["pixel_values"] for b in batch], dim=0),
        "image_grid_thw": torch.stack([b["image_grid_thw"] for b in batch]),
        "topk_ids": pad_stack("topk_ids", -1),
        "topk_logprobs": pad_stack("topk_logprobs", 0),
        "n_prompt": torch.tensor([b["n_prompt"] for b in batch]),
        "n_out": torch.tensor([b["n_out"] for b in batch]),
        "student_grid_hw": torch.stack([b["student_grid_hw"] for b in batch]),
    }
    side_keys = [k for k in batch[0] if k.startswith("mm_") or k.endswith("_token_type_ids")]
    for key in side_keys:
        out[key] = pad_stack(key, 0)
    if "teacher_input_ids" in batch[0]:
        out["teacher_input_ids"] = pad_stack("teacher_input_ids", pad_id)
        out["teacher_attention_mask"] = pad_stack("teacher_attention_mask", 0)
        out["teacher_pixel_values"] = torch.cat([b["teacher_pixel_values"] for b in batch], dim=0)
        out["teacher_image_grid_thw"] = torch.stack([b["teacher_image_grid_thw"] for b in batch])
        out["teacher_n_prompt"] = torch.tensor([b["teacher_n_prompt"] for b in batch])
        out["teacher_grid_hw"] = torch.stack([b["teacher_grid_hw"] for b in batch])
    return out


class LengthGroupedBatchSampler(torch.utils.data.Sampler):
    """Draw ``epoch_size`` rows (dataset weight ∝ n^power), then yield length-sorted micro-batches."""

    def __init__(
        self,
        lengths: list[int],
        datasets: list[str],
        micro_batch: int,
        epoch_size: int,
        *,
        power: float = 0.5,
        weighting: bool = True,
        seed: int = 643,
    ) -> None:
        self.lengths = lengths
        self.probs = row_probabilities(datasets, power, weighting)
        self.micro_batch = micro_batch
        self.epoch_size = epoch_size
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return math_ceil(self.epoch_size, self.micro_batch)

    def __iter__(self):
        import numpy as np

        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        n = len(self.lengths)
        replace = self.epoch_size > n or self.probs is not None and not np.allclose(self.probs, 1.0 / n)
        picks = rng.choice(n, size=self.epoch_size, replace=replace or self.epoch_size > n, p=self.probs)
        order = sorted(picks.tolist(), key=lambda i: (self.lengths[i], i))
        batches = [order[i : i + self.micro_batch] for i in range(0, len(order), self.micro_batch)]
        rng.shuffle(batches)
        return iter(batches)


def math_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d
