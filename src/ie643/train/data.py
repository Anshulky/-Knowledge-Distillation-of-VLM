"""CE-only training examples for the student: packed image + extraction prompt -> source_gt JSON.

The prompt side is built exactly like inference (same chat template and prompt as
ie643.teacher through vLLM), so the fine-tuned model is served and scored with the
unchanged extraction client. Only answer tokens carry labels; with a labeled-field
map, the value tokens of fields the dataset does not annotate are ignored too.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from ie643.schema import FIELD_ORDER, coerce_canonical, teacher_prompt

IGNORE_INDEX = -100
# Same limits as ie643.pack (Qwen smart_resize): packed images pass through unchanged.
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28


def target_json(source_gt: dict[str, Any] | None) -> tuple[str, dict[str, tuple[int, int]]]:
    """Canonical JSON (identical to json.dumps(..., ensure_ascii=False)) and each value's char span."""
    canon = coerce_canonical(source_gt or {})
    parts: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    pos = 1  # after "{"
    for i, key in enumerate(FIELD_ORDER):
        head = ("" if i == 0 else ", ") + json.dumps(key, ensure_ascii=False) + ": "
        value = json.dumps(canon[key], ensure_ascii=False)
        start = pos + len(head)
        spans[key] = (start, start + len(value))
        parts.append(head + value)
        pos = start + len(value)
    return "{" + "".join(parts) + "}", spans


def is_val_uid(uid: str, frac: float) -> bool:
    """Deterministic held-out slice of train for eval loss (test is never touched)."""
    return int(hashlib.sha1(uid.encode("utf-8")).hexdigest()[:8], 16) / 2**32 < frac


class SFTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        root: Path,
        processor: Any,
        *,
        labeled: dict[str, list[str]] | None,
        max_len: int,
    ) -> None:
        self.rows = rows
        self.root = root
        self.processor = processor
        self.tok = processor.tokenizer
        self.labeled = {ds: set(keys) for ds, keys in labeled.items()} if labeled else None
        self.max_len = max_len
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.newline = self.tok("\n", add_special_tokens=False)["input_ids"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.rows[idx]
        image = Image.open(self.root / rec["image"]).convert("RGB")
        prompt = teacher_prompt()
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.processor(text=[text], images=[image], return_tensors="pt")
        prompt_ids = enc["input_ids"][0].tolist()

        answer, spans = target_json(rec.get("source_gt"))
        ans = self.tok(answer, add_special_tokens=False, return_offsets_mapping=True)
        ans_ids = list(ans["input_ids"])
        ans_labels = list(ans_ids)
        if self.labeled is not None and rec["dataset"] in self.labeled:
            keep = self.labeled[rec["dataset"]]
            skip = [spans[k] for k in FIELD_ORDER if k not in keep]
            for j, (a, b) in enumerate(ans["offset_mapping"]):
                if any(a < e and b > s for s, e in skip):
                    ans_labels[j] = IGNORE_INDEX

        # "<|im_end|>\n" closes the assistant turn; learning <|im_end|> teaches the model to stop.
        input_ids = prompt_ids + ans_ids + [self.im_end] + self.newline
        labels = [IGNORE_INDEX] * len(prompt_ids) + ans_labels + [self.im_end] + [IGNORE_INDEX] * len(self.newline)
        if len(input_ids) > self.max_len:
            input_ids, labels = input_ids[: self.max_len], labels[: self.max_len]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": enc["pixel_values"],
            "image_grid_thw": enc["image_grid_thw"],
        }


def collate(batch: list[dict[str, torch.Tensor]], *, pad_id: int) -> dict[str, torch.Tensor]:
    """Right-pad text; Qwen-VL takes all images' patches concatenated with one grid row per image."""
    n = max(len(b["input_ids"]) for b in batch)
    input_ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), n), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), n), dtype=torch.long)
    for i, b in enumerate(batch):
        length = len(b["input_ids"])
        input_ids[i, :length] = b["input_ids"]
        labels[i, :length] = b["labels"]
        attention_mask[i, :length] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": torch.cat([b["pixel_values"] for b in batch]),
        "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batch]),
    }
