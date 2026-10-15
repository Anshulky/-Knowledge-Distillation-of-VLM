"""MiniLMv2 Q/K/V relation distillation (docs/distillation_plan.md, E5).

Relations are the softmax of scaled dot products inside R slices of the concatenated
head vectors, so the student (2048-d) and the teacher (8192-d) need no projector.
Rows are output tokens. Columns are image tokens then output tokens (or outputs only).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


class StopForward(Exception):
    """Raised from the teacher layer hook so the rest of the 32B stack does not run."""


@dataclass
class QKVCapture:
    q: torch.Tensor | None = None
    k: torch.Tensor | None = None
    v: torch.Tensor | None = None
    acts: list[torch.Tensor] = field(default_factory=list)
    rope_on: list[bool] = field(default_factory=lambda: [False])

    def clear(self) -> None:
        self.q = self.k = self.v = None
        self.acts.clear()


def decoder_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    seen: set[int] = set()
    layers = []
    for module in model.modules():
        if type(module).__name__.endswith("DecoderLayer") and hasattr(module, "self_attn") and id(module) not in seen:
            seen.add(id(module))
            layers.append(module)
    if not layers:
        raise RuntimeError("no text decoder layers found")
    return layers


def layer_index(n_layers: int, number: int) -> int:
    """1-based layer number (36 = last of 36) to a module index."""
    if not 1 <= number <= n_layers:
        raise ValueError(f"layer {number} is outside 1..{n_layers}")
    return number - 1


def attach_qkv(attn: torch.nn.Module, cap: QKVCapture, *, pre_rope: bool) -> list:
    """Hooks on q/k/v projections. Pre-RoPE is the projection output; V is always that."""
    handles = []

    def save(name: str):
        def hook(_module, _inputs, output):
            setattr(cap, name, output)
        return hook

    if pre_rope:
        handles.append(attn.q_proj.register_forward_hook(save("q")))
        handles.append(attn.k_proj.register_forward_hook(save("k")))
    handles.append(attn.v_proj.register_forward_hook(save("v")))
    return handles


def attach_acts(layers: list[torch.nn.Module], cap: QKVCapture) -> list:
    """Max-|hidden| entering each layer, for visual-sink detection. Detached."""

    def hook(_module, args, kwargs):
        hidden = kwargs.get("hidden_states") if kwargs else None
        if hidden is None:
            hidden = args[0]
        cap.acts.append(hidden.detach().abs().amax(dim=-1))

    return [layer.register_forward_pre_hook(hook, with_kwargs=True) for layer in layers]


def attach_stop(layer: torch.nn.Module) -> list:
    def hook(_module, _inputs, _output):
        raise StopForward()

    return [layer.register_forward_hook(hook)]


def install_post_rope(layer: torch.nn.Module) -> None:
    """Capture rotated Q/K from the modeling module's ``apply_rotary_pos_emb``.

    The attention forward looks that name up as a global. The training loop puts the
    live ``QKVCapture`` in ``module._ie643_rope_holder[0]`` and sets ``rope_on`` only
    while the distillation layer runs, so earlier layers do not overwrite it.
    """
    import sys

    mod = sys.modules[type(layer).__module__]
    if not hasattr(mod, "apply_rotary_pos_emb"):
        raise RuntimeError(
            f"{type(layer).__module__} has no apply_rotary_pos_emb; post-RoPE capture is unavailable"
        )
    if getattr(mod, "_ie643_rope_patched", False):
        return
    holder: list = [None]
    orig = mod.apply_rotary_pos_emb

    def wrapped(q, k, *args, **kwargs):
        out = orig(q, k, *args, **kwargs)
        cap = holder[0]
        if cap is not None and cap.rope_on[0]:
            cap.q, cap.k = _as_hidden(out[0]), _as_hidden(out[1])
        return out

    mod.apply_rotary_pos_emb = wrapped
    mod._ie643_rope_patched = True
    mod._ie643_rope_holder = holder


def _as_hidden(t: torch.Tensor) -> torch.Tensor:
    """[B, heads, N, d] -> [B, N, heads*d]. Projection outputs are already [B, N, H]."""
    if t.ndim == 4:
        return t.permute(0, 2, 1, 3).reshape(t.shape[0], t.shape[2], -1)
    return t


def expand_kv(kv: torch.Tensor, n_kv: int, n_q: int) -> torch.Tensor:
    """Repeat KV heads so K/V match the query-head width. [B, N, n_kv*d] -> [B, N, n_q*d]."""
    if n_q % n_kv != 0:
        raise ValueError(f"cannot expand {n_kv} kv heads to {n_q} query heads")
    b, n, hidden = kv.shape
    d = hidden // n_kv
    x = kv.view(b, n, n_kv, d).repeat_interleave(n_q // n_kv, dim=2)
    return x.reshape(b, n, n_q * d)


def _relation(rows: torch.Tensor, cols: torch.Tensor, col_mask: torch.Tensor) -> torch.Tensor:
    """rows/cols [R, n, d] -> softmax relations [R, n_rows, n_cols]."""
    d = rows.shape[-1]
    logits = torch.matmul(rows, cols.transpose(-1, -2)) / math.sqrt(d)
    logits = logits + col_mask
    return torch.softmax(logits, dim=-1)


def _kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """KL(p || q) per row, mean over rows and relation heads. Zeros do not contribute."""
    log_p = torch.where(p > 0, p.log(), torch.zeros_like(p))
    log_q = torch.where(q > 0, q.log(), torch.zeros_like(q))
    return (p * (log_p - log_q)).sum(-1).mean()


def relation_heads(x: torch.Tensor, n_rel: int) -> torch.Tensor:
    """[N, H] -> [R, N, H/R]."""
    if x.shape[-1] % n_rel != 0:
        raise ValueError(f"hidden {x.shape[-1]} is not divisible by R={n_rel}")
    n, hidden = x.shape
    return x.view(n, n_rel, hidden // n_rel).permute(1, 0, 2)


def column_mask(
    row_pos: torch.Tensor,
    col_pos: torch.Tensor,
    sink: torch.Tensor,
    *,
    causal: bool,
) -> torch.Tensor:
    """[n_rows, n_cols] additive mask. Sink columns, and later positions when causal."""
    mask = torch.zeros(row_pos.shape[0], col_pos.shape[0], device=row_pos.device)
    if causal:
        mask = mask.masked_fill(col_pos.unsqueeze(0) > row_pos.unsqueeze(1), float("-inf"))
    if sink.any():
        mask = mask.masked_fill(sink.unsqueeze(0), float("-inf"))
    return mask


def pairwise_kl(
    student: torch.Tensor,
    teacher: torch.Tensor,
    row_pos: torch.Tensor,
    col_pos: torch.Tensor,
    sink: torch.Tensor,
    *,
    n_rel: int,
    causal: bool,
    resample_hw: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> torch.Tensor:
    """KL between one pair of [N, H] vectors, restricted to the given row/column positions.

    ``student`` and ``teacher`` are already indexed down to the aligned column set S,
    and ``row_pos`` indexes rows inside that set (output tokens). Positions are the
    original sequence positions, used only for the causal test.
    """
    s = relation_heads(student.float(), n_rel)
    t = relation_heads(teacher.float(), n_rel)
    # row indices: outputs are a subset of columns. Match by position value.
    col_index = {int(p): i for i, p in enumerate(col_pos.tolist())}
    row_idx = torch.tensor([col_index[int(p)] for p in row_pos.tolist()], device=student.device)
    mask = column_mask(row_pos, col_pos, sink, causal=causal).to(dtype=s.dtype)
    rel_s = _relation(s[:, row_idx], s, mask)
    rel_t = _relation(t[:, row_idx], t, mask)
    if resample_hw is not None:
        src_hw, dst_hw = resample_hw
        flat = resample_image_columns(rel_t.reshape(-1, rel_t.shape[-1]), src_hw, dst_hw)
        rel_t = flat.view(rel_t.shape[0], rel_t.shape[1], -1)
    return _kl(rel_t, rel_s)


def sink_columns(
    acts: list[torch.Tensor],
    img_pos: torch.Tensor,
    ratio: float,
) -> torch.Tensor:
    """Boolean over image tokens. True where max-over-layers |h| exceeds ``ratio`` times the median."""
    if not acts or img_pos.numel() == 0:
        return torch.zeros(img_pos.shape[0], dtype=torch.bool, device=img_pos.device)
    stacked = torch.stack([a[img_pos] for a in acts])  # [L, n_img]
    med = stacked.median(dim=-1, keepdim=True).values.clamp(min=1e-8)
    score = (stacked / med).amax(0)
    return score > ratio


def resample_image_columns(values: torch.Tensor, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> torch.Tensor:
    """Bilinear resample of the leading image columns of ``values`` [n_rows, n_src + n_rest].

    Rows are renormalised. Used by resize option A, where the teacher grid is 32 px
    and the student grid is 28 px.
    """
    sh, sw = src_hw
    dh, dw = dst_hw
    n_src = sh * sw
    img, rest = values[:, :n_src], values[:, n_src:]
    img = img.reshape(values.shape[0], 1, sh, sw)
    img = F.interpolate(img, size=(dh, dw), mode="bilinear", align_corners=False)
    out = torch.cat([img.reshape(values.shape[0], dh * dw), rest], dim=-1)
    denom = out.sum(-1, keepdim=True).clamp(min=1e-8)
    return out / denom


def _select_qkv(qkv, cols: torch.Tensor, n_q: int, n_kv: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = qkv
    k = expand_kv(k.unsqueeze(0), n_kv, n_q).squeeze(0)
    v = expand_kv(v.unsqueeze(0), n_kv, n_q).squeeze(0)
    return q.index_select(0, cols), k.index_select(0, cols), v.index_select(0, cols)


def _kl_selected(
    student: torch.Tensor,
    teacher: torch.Tensor,
    n_img_s: int,
    n_img_t: int,
    n_out: int,
    sink_s: torch.Tensor,
    sink_t: torch.Tensor,
    *,
    n_rel: int,
    causal: bool,
    resample_hw: tuple[tuple[int, int], tuple[int, int]] | None,
) -> torch.Tensor:
    """KL for vectors already restricted to S = image columns then output columns."""

    def logical(n_img: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cols = torch.arange(n_img + n_out, device=student.device)
        rows = cols[n_img:]
        return rows, cols, cols

    s_rows, s_cols, _ = logical(n_img_s)
    t_rows, t_cols, _ = logical(n_img_t)
    rel_s = _masked_rel(student, s_rows, s_cols, sink_s, n_rel=n_rel, causal=causal)
    rel_t = _masked_rel(teacher, t_rows, t_cols, sink_t, n_rel=n_rel, causal=causal)
    if resample_hw is not None:
        src_hw, dst_hw = resample_hw
        flat = resample_image_columns(rel_t.reshape(-1, rel_t.shape[-1]), src_hw, dst_hw)
        rel_t = flat.view(rel_t.shape[0], rel_t.shape[1], -1)
    return _kl(rel_t, rel_s)


def _masked_rel(x, row_pos, col_pos, sink, *, n_rel: int, causal: bool) -> torch.Tensor:
    heads = relation_heads(x.float(), n_rel)
    row_idx = row_pos  # logical positions are also the indices into S
    mask = column_mask(row_pos, col_pos, sink, causal=causal).to(dtype=heads.dtype)
    return _relation(heads[:, row_idx], heads, mask)


def relation_loss_from_qkv(
    student_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    teacher_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    student_cols: torch.Tensor,
    teacher_cols: torch.Tensor,
    n_out: int,
    sink_student: torch.Tensor,
    sink_teacher: torch.Tensor,
    *,
    student_heads: int,
    student_kv_heads: int,
    teacher_heads: int,
    teacher_kv_heads: int,
    n_rel: int,
    causal: bool,
    resample_hw: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> torch.Tensor:
    """Sum of Q-Q, K-K and V-V relation KLs for one sample.

    QKV tensors are ``[seq, hidden]`` (batch item 0). ``student_cols`` / ``teacher_cols``
    are sequence indices ordered as image tokens (raster) then output tokens. Sink masks
    have one bool per column. When the image grids differ (resize option A), ``resample_hw``
    is ``(teacher_hw, student_hw)`` and the teacher image columns are resampled onto the
    student grid before the KL.
    """
    s = _select_qkv(student_qkv, student_cols, student_heads, student_kv_heads)
    t = _select_qkv(teacher_qkv, teacher_cols, teacher_heads, teacher_kv_heads)
    n_img_s = int(student_cols.shape[0] - n_out)
    n_img_t = int(teacher_cols.shape[0] - n_out)
    total = student_qkv[0].new_zeros(())
    for xs, xt in zip(s, t, strict=True):
        total = total + _kl_selected(
            xs, xt, n_img_s, n_img_t, n_out, sink_student, sink_teacher,
            n_rel=n_rel, causal=causal, resample_hw=resample_hw,
        )
    return total
