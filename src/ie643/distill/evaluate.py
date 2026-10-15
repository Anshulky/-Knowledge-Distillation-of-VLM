"""Sanity generation on a fixed set of train uids. Not used to pick checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from ie643.distill.data import chat_text
from ie643.teacher.parse import extract_json_object


def sanity_uids(uids: list[str], n: int) -> list[str]:
    return sorted(uids)[:n]


@torch.no_grad()
def generate_one(model, processor, image: Image.Image, max_new_tokens: int, device: str) -> str:
    enc = processor(text=[chat_text(processor)], images=[image], return_tensors="pt")
    enc = {k: v.to(device) if torch.is_tensor(v) else v for k, v in enc.items()}
    prompt_len = int(enc["input_ids"].shape[1])
    was_training = model.training
    model.eval()
    prev_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True
    try:
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    finally:
        model.config.use_cache = prev_cache
        model.train(was_training)
    new_ids = out[0, prompt_len:]
    return processor.tokenizer.decode(new_ids, skip_special_tokens=True)


def _parses(text: str) -> bool:
    try:
        extract_json_object(text)
    except (ValueError, json.JSONDecodeError):
        return False
    return True


def sanity_report(
    model,
    processor,
    rows: list[dict],
    root: Path,
    *,
    raw_key: str,
    max_new_tokens: int,
    device: str,
) -> dict:
    """JSON parse rate and exact match against the teacher string, on train rows only."""
    parsed = 0
    exact = 0
    details = []
    for rec in rows:
        text = generate_one(model, processor, Image.open(root / rec["image"]).convert("RGB"), max_new_tokens, device)
        gold = rec.get(raw_key) or ""
        ok_parse = _parses(text)
        ok_exact = text.strip() == str(gold).strip()
        parsed += int(ok_parse)
        exact += int(ok_exact)
        details.append({"uid": rec["uid"], "parse": ok_parse, "exact": ok_exact})
    n = len(rows)
    return {
        "n": n,
        "parse_rate": parsed / n if n else 0.0,
        "exact_match": exact / n if n else 0.0,
        "rows": details,
    }
