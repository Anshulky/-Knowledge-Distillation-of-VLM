"""CE and top-(k+1) bucket KD (docs/distillation_plan.md, E3 and E4).

Teacher buckets are the top-K logprobs plus one tail bucket ``log(1 - Σ p_topk)``.
Temperature is applied to both bucket vectors the same way, so a matched pair stays
matched. CE always covers every output token; ``weight: value`` restricts only KD.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LOG_FLOOR = -30.0
# Never take log(1 - mass) at mass = 1: that gradient is infinite and shows up as NaN
# once the student (or a one-hot teacher bucket) puts essentially all mass in the top-K.
TAIL_EPS = 1e-6
MAX_LOG_MASS = math.log1p(-TAIL_EPS)


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """log(1 - exp(x)) for x <= 0. Stable across the x = log(1/2) boundary."""
    threshold = -math.log(2)
    return torch.where(x > threshold, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def kd_buckets(
    topk_ids: torch.Tensor,
    topk_logprobs: torch.Tensor,
    student_logits: torch.Tensor,
    tau: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tempered teacher and student bucket log-probs, shape [..., K+1].

    ``topk_ids`` / ``topk_logprobs`` are [..., K], ``student_logits`` is [..., V].
    Ids < 0 (padding) are ignored. Logprobs are clamped at -30 so a -inf from the
    teacher file cannot NaN the backward.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    tk_lp = topk_logprobs.float().clamp(min=LOG_FLOOR)
    valid = topk_ids >= 0
    tk_lp = tk_lp.masked_fill(~valid, LOG_FLOOR)
    log_mass = torch.logsumexp(tk_lp, dim=-1).clamp(max=MAX_LOG_MASS)
    tail = log1mexp(log_mass).clamp(min=LOG_FLOOR)
    lp_t = torch.cat([tk_lp, tail.unsqueeze(-1)], dim=-1)

    lsm = F.log_softmax(student_logits.float(), dim=-1)
    gathered = lsm.gather(-1, topk_ids.clamp(min=0))
    gathered = gathered.masked_fill(~valid, LOG_FLOOR)
    s_mass = torch.logsumexp(gathered, dim=-1).clamp(max=MAX_LOG_MASS)
    s_tail = log1mexp(s_mass).clamp(min=LOG_FLOOR)
    lp_s = torch.cat([gathered, s_tail.unsqueeze(-1)], dim=-1)

    scale = float(tau)
    return F.log_softmax(lp_t / scale, dim=-1), F.log_softmax(lp_s / scale, dim=-1)


def bucket_divergence(lp_t: torch.Tensor, lp_s: torch.Tensor, div: str, skew: float = 0.1) -> torch.Tensor:
    """Per-token divergence of the K+1 buckets. ``lp_*`` are log-probabilities."""
    if div == "fkl":
        p = lp_t.exp()
        return (p * (lp_t - lp_s)).sum(-1)
    if div == "rkl":
        q = lp_s.exp()
        return (q * (lp_s - lp_t)).sum(-1)
    if div == "skl":
        if not 0.0 < skew < 1.0:
            raise ValueError(f"skew must be in (0, 1), got {skew}")
        log_m = torch.logaddexp(math.log(skew) + lp_t, math.log(1.0 - skew) + lp_s)
        p = lp_t.exp()
        return (p * (lp_t - log_m)).sum(-1)
    raise ValueError(f"unknown kd.div {div!r} (expected fkl, rkl, skl)")


def token_weights(lp_t_unit: torch.Tensor, value_mask: torch.Tensor, mode: str) -> torch.Tensor:
    """Per-token KD weights. ``lp_t_unit`` is the teacher buckets at tau = 1.

    ``entropy`` normalises by the mean entropy of this group and clips to [0, 5].
    ``value`` is the JSON-leaf mask (0/1). ``none`` is 1.
    """
    if mode == "none":
        return torch.ones(lp_t_unit.shape[:-1], device=lp_t_unit.device, dtype=lp_t_unit.dtype)
    if mode == "value":
        w = value_mask.to(dtype=lp_t_unit.dtype)
        if float(w.sum()) <= 0:
            return torch.ones_like(w)
        return w
    if mode == "entropy":
        p = lp_t_unit.exp()
        log_p = torch.where(p > 0, lp_t_unit, torch.zeros_like(lp_t_unit))
        entropy = -(p * log_p).sum(-1)
        mean = entropy.mean().clamp(min=1e-8)
        return (entropy / mean).clamp(0, 5)
    raise ValueError(f"unknown kd.weight {mode!r} (expected none, entropy, value)")


def sequence_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    value_mask: torch.Tensor,
    n_prompt: torch.Tensor,
    n_out: torch.Tensor,
    topk_ids: torch.Tensor | None,
    topk_logprobs: torch.Tensor | None,
    *,
    kd_enabled: bool,
    alpha: float,
    tau: float,
    div: str,
    skew: float,
    weight: str,
) -> dict[str, torch.Tensor]:
    """Mean CE over output tokens, plus optional bucket KD.

    Logit at index ``n_prompt - 1 + j`` predicts output token ``j``. Labels store the
    token id at each position (prompt positions are -100).
    """
    ce_parts: list[torch.Tensor] = []
    kd_parts: list[torch.Tensor] = []
    w_parts: list[torch.Tensor] = []
    for b in range(logits.shape[0]):
        start = int(n_prompt[b])
        length = int(n_out[b])
        if start < 1 or length < 1:
            raise RuntimeError(f"empty output span at batch {b}: n_prompt={start} n_out={length}")
        pred = logits[b, start - 1 : start - 1 + length]
        target = labels[b, start : start + length]
        ce_parts.append(F.cross_entropy(pred.float(), target, reduction="none"))
        if not kd_enabled:
            continue
        assert topk_ids is not None and topk_logprobs is not None
        ids = topk_ids[b, :length].unsqueeze(0)
        lp = topk_logprobs[b, :length].unsqueeze(0)
        lp_t, lp_s = kd_buckets(ids, lp, pred.unsqueeze(0), tau)
        if tau == 1.0:
            lp_unit = lp_t
        else:
            lp_unit, _ = kd_buckets(ids, lp, pred.unsqueeze(0), 1.0)
        kd_parts.append(bucket_divergence(lp_t[0], lp_s[0], div, skew))
        w_parts.append(token_weights(lp_unit[0], value_mask[b, start : start + length], weight))

    ce = torch.cat(ce_parts).mean()
    out = {"ce": ce, "kd": logits.new_zeros(()), "loss": ce}
    if kd_enabled:
        dist = torch.cat(kd_parts)
        weights = torch.cat(w_parts)
        kd = (weights * (tau ** 2) * dist).sum() / weights.sum().clamp(min=1e-8)
        out["kd"] = kd
        out["loss"] = alpha * ce + (1.0 - alpha) * kd
    return out
