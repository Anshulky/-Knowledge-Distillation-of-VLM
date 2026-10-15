"""Soft labels: per-token top-k logprobs from a vLLM chat completion, one .npz per uid.

Layout (docs/distillation_plan.md, B.4):
  token_ids      int32   [T]     tokens the model emitted (hard targets)
  topk_ids       int32   [T, K]  top-K token ids per position, logprob descending
  topk_logprobs  float16 [T, K]
  chosen_logprob float32 [T]     logprob of the emitted token (can sit outside the top-K under penalties)
  meta           str             JSON: model, server version, logprobs mode, K, sampling, hashes
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

_PREFIX = "token_id:"


def safe_name(uid: str) -> str:
    return uid.replace("::", "__").replace(":", "_").replace("/", "_")


def soft_path(soft_dir: Path, rec: dict) -> Path:
    return soft_dir / rec["dataset"] / f"{safe_name(rec['uid'])}.npz"


def _token_id(token: str) -> int:
    if not token.startswith(_PREFIX):
        raise ValueError(f"expected '{_PREFIX}<id>' (serve with return_tokens_as_token_ids), got {token!r}")
    return int(token[len(_PREFIX):])


def sha1_file(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def to_arrays(content: list[dict[str, Any]], k: int) -> dict[str, np.ndarray]:
    """vLLM ``choices[0].logprobs.content`` -> fixed-shape arrays (short rows padded with id -1 / -inf)."""
    n = len(content)
    token_ids = np.empty(n, dtype=np.int32)
    chosen = np.empty(n, dtype=np.float32)
    topk_ids = np.full((n, k), -1, dtype=np.int32)
    topk_lp = np.full((n, k), -np.inf, dtype=np.float32)
    for t, pos in enumerate(content):
        token_ids[t] = _token_id(pos["token"])
        chosen[t] = pos["logprob"]
        top = sorted(pos.get("top_logprobs") or [], key=lambda e: -e["logprob"])[:k]
        for j, entry in enumerate(top):
            topk_ids[t, j] = _token_id(entry["token"])
            topk_lp[t, j] = entry["logprob"]
    return {
        "token_ids": token_ids,
        "topk_ids": topk_ids,
        "topk_logprobs": topk_lp.astype(np.float16),
        "chosen_logprob": chosen,
    }


def save_soft(path: Path, content: list[dict[str, Any]], *, k: int, meta: dict[str, Any]) -> int:
    arrays = to_arrays(content, k)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, meta=np.asarray(json.dumps(meta, sort_keys=True)), **arrays)
    tmp.replace(path)
    return len(arrays["token_ids"])


def load_soft(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        out = {key: z[key] for key in z.files}
    out["meta"] = json.loads(str(out["meta"]))
    return out
