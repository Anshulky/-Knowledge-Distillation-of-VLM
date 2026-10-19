"""CE-only training examples for the student: packed image + extraction prompt -> source_gt JSON.

The prompt side is built exactly like inference (same chat template and prompt as
ie643.teacher through vLLM), so the fine-tuned model is served and scored with the
unchanged extraction client. Only answer tokens carry labels.

Masking (dataset labels are incomplete): a field with a value is always trained. An empty
field is trained as null only where the dataset annotates that field at all (``null_fields``,
filled in >= 1% of its rows, see ie643.labeled); a field the dataset never labels is masked.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from ie643.labeled import is_empty
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


def masked_spans(source_gt: dict[str, Any] | None, spans: dict[str, tuple[int, int]], null_fields: set[str]) -> list[tuple[int, int]]:
    """Value spans left out of the loss: empty fields the dataset does not fill reliably."""
    canon = coerce_canonical(source_gt or {})
    return [spans[k] for k in FIELD_ORDER if is_empty(canon[k]) and k not in null_fields]


def balance_weights(datasets: list[str], power: float) -> list[float]:
    """Per-row sampling weight so that P(dataset) ∝ n_dataset^power (power 1 = no balancing)."""
    counts = Counter(datasets)
    return [counts[ds] ** (power - 1.0) for ds in datasets]


class SFTDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        root: Path,
        processor: Any,
        *,
        null_fields: dict[str, list[str]] | None,
        max_len: int,
    ) -> None:
        """null_fields: per dataset, fields whose empty label is trained as null; None trains every null."""
        self.root = root
        self.processor = processor
        self.tok = processor.tokenizer
        self.null_fields = {ds: set(keys) for ds, keys in null_fields.items()} if null_fields is not None else None
        self.max_len = max_len
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.newline = self.tok("\n", add_special_tokens=False)["input_ids"]
        # Too-long examples are dropped (never truncated: a cut-off target teaches a broken JSON).
        overhead = self._prompt_overhead()
        self.rows, self.dropped_long = [], []
        for rec in rows:
            (self.rows if self.length(rec, overhead) <= max_len else self.dropped_long).append(rec)

    def _prompt_text(self) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": teacher_prompt()}]}]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _prompt_overhead(self) -> int:
        """Prompt tokens other than the image; the processor expands the one image pad per 28x28 patch."""
        return len(self.tok(self._prompt_text(), add_special_tokens=False)["input_ids"]) - 1

    def length(self, rec: dict[str, Any], overhead: int) -> int:
        n_image = (int(rec["width"]) // 28) * (int(rec["height"]) // 28)
        n_answer = len(self.tok(target_json(rec.get("source_gt"))[0], add_special_tokens=False)["input_ids"])
        return overhead + n_image + n_answer + 1 + len(self.newline)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rec = self.rows[idx]
        image = Image.open(self.root / rec["image"]).convert("RGB")
        enc = self.processor(text=[self._prompt_text()], images=[image], return_tensors="pt")
        prompt_ids = enc["input_ids"][0].tolist()

        answer, spans = target_json(rec.get("source_gt"))
        ans = self.tok(answer, add_special_tokens=False, return_offsets_mapping=True)
        ans_ids = list(ans["input_ids"])
        ans_labels = list(ans_ids)
        if self.null_fields is not None:
            skip = masked_spans(rec.get("source_gt"), spans, self.null_fields.get(rec["dataset"], set()))
            for j, (a, b) in enumerate(ans["offset_mapping"]):
                if any(a < e and b > s for s, e in skip):
                    ans_labels[j] = IGNORE_INDEX

        # "<|im_end|>\n" closes the assistant turn; learning <|im_end|> teaches the model to stop.
        input_ids = prompt_ids + ans_ids + [self.im_end] + self.newline
        labels = [IGNORE_INDEX] * len(prompt_ids) + ans_labels + [self.im_end] + [IGNORE_INDEX] * len(self.newline)
        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": enc["pixel_values"],
            "image_grid_thw": enc["image_grid_thw"],
        }
        # transformers 5.x builds the 3D (M-RoPE) image positions only when it gets mm_token_type_ids
        # (0 text, 1 image); without it image tokens silently get 1D positions, unlike vLLM at inference.
        if "mm_token_type_ids" in enc:
            types = enc["mm_token_type_ids"][0]
            item["mm_token_type_ids"] = torch.cat([types, types.new_zeros(len(input_ids) - len(prompt_ids))])
        return item


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
    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": torch.cat([b["pixel_values"] for b in batch]),
        "image_grid_thw": torch.cat([b["image_grid_thw"] for b in batch]),
    }
    if "mm_token_type_ids" in batch[0]:
        types = torch.zeros((len(batch), n), dtype=batch[0]["mm_token_type_ids"].dtype)
        for i, b in enumerate(batch):
            types[i, : len(b["mm_token_type_ids"])] = b["mm_token_type_ids"]
        out["mm_token_type_ids"] = types
    return out
