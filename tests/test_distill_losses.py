"""E4 bucket KD and E5 relation checks from docs/distillation_plan.md."""

from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from ie643.distill.losses import bucket_divergence, kd_buckets, log1mexp, sequence_loss
from ie643.distill.relations import (
    QKVCapture,
    StopForward,
    _relation,
    attach_qkv,
    column_mask,
    relation_heads,
    relation_loss_from_qkv,
)


def _kl(p, q) -> float:
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))


class BucketKD(unittest.TestCase):
    def test_log1mexp(self) -> None:
        x = torch.tensor([math.log(0.5), math.log(0.1), -1e-8])
        got = log1mexp(x)
        self.assertAlmostEqual(float(got[0]), math.log(0.5), places=5)
        self.assertAlmostEqual(float(got[1]), math.log(0.9), places=5)
        self.assertTrue(torch.isfinite(got[2]) or got[2] < -20)

    def test_worked_example_is_full_bucket_kl(self) -> None:
        teacher = [0.62, 0.30, 0.07, 0.01]
        student = [0.40, 0.10, 0.05, 0.45]
        logits = torch.tensor([0.40, 0.10, 0.05, 0.30, 0.15]).log().view(1, 1, -1)
        ids = torch.tensor([[[0, 1, 2]]])
        lp = torch.tensor(teacher[:3]).log().view(1, 1, 3)
        lp_t, lp_s = kd_buckets(ids, lp, logits, tau=1.0)
        got = float(bucket_divergence(lp_t, lp_s, "fkl"))
        self.assertAlmostEqual(got, _kl(teacher, student), places=4)
        self.assertAlmostEqual(got, 0.587, places=2)
        top_p = [v / 0.99 for v in teacher[:3]]
        top_q = [v / 0.55 for v in student[:3]]
        self.assertGreater(abs(got - _kl(top_p, top_q)), 0.2)

    def test_matched_distributions_are_zero(self) -> None:
        probs = torch.tensor([0.5, 0.2, 0.1, 0.1, 0.1])
        logits = probs.log().view(1, 1, -1)
        ids = torch.tensor([[[0, 1, 2]]])
        lp = probs[:3].log().view(1, 1, 3)
        for tau in (1.0, 2.0):
            lp_t, lp_s = kd_buckets(ids, lp, logits, tau=tau)
            for div in ("fkl", "rkl", "skl"):
                got = float(bucket_divergence(lp_t, lp_s, div, skew=0.1))
                self.assertAlmostEqual(got, 0.0, places=5, msg=f"{div} tau={tau}")

    def test_infinite_teacher_logprob_has_finite_grad(self) -> None:
        logits = torch.randn(1, 1, 5, requires_grad=True)
        ids = torch.tensor([[[0, 1, 2]]])
        lp = torch.tensor([[[math.log(0.7), float("-inf"), math.log(0.1)]]])
        lp_t, lp_s = kd_buckets(ids, lp, logits, tau=1.0)
        bucket_divergence(lp_t, lp_s, "fkl").sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_saturated_student_bucket_has_finite_grad(self) -> None:
        """A student that puts all mass on one top-k id used to NaN the backward."""
        logits = torch.full((1, 2, 8), -20.0, requires_grad=True)
        with torch.no_grad():
            logits[0, :, 0] = 50.0
        ids = torch.tensor([[[0, 1, 2], [0, 1, 2]]])
        lp = torch.tensor([[[math.log(0.9), math.log(0.05), math.log(0.04)], [0.0, -40.0, -40.0]]])
        lp_t, lp_s = kd_buckets(ids, lp, logits, tau=1.0)
        loss = bucket_divergence(lp_t, lp_s, "fkl").sum()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_ce_covers_every_output_token(self) -> None:
        logits = torch.zeros(1, 3, 4)
        logits[0, 0, 1] = 4
        logits[0, 1, 2] = 4
        labels = torch.tensor([[-100, 1, 2]])
        value = torch.tensor([[0.0, 0.0, 1.0]])
        out = sequence_loss(
            logits, labels, value, torch.tensor([1]), torch.tensor([2]),
            topk_ids=None, topk_logprobs=None,
            kd_enabled=False, alpha=1.0, tau=1.0, div="fkl", skew=0.1, weight="value",
        )
        manual = torch.nn.functional.cross_entropy(logits[0, :2].float(), labels[0, 1:])
        self.assertAlmostEqual(float(out["ce"]), float(manual), places=5)
        self.assertAlmostEqual(float(out["loss"]), float(manual), places=5)


class Relations(unittest.TestCase):
    def test_self_distillation_is_zero_and_rows_are_stochastic(self) -> None:
        torch.manual_seed(0)
        q = torch.randn(8, 32)
        k = torch.randn(8, 16)
        v = torch.randn(8, 16)
        cols = torch.arange(8)
        sink = torch.zeros(8, dtype=torch.bool)
        sink[1] = True
        loss = relation_loss_from_qkv(
            (q, k, v), (q, k, v), cols, cols, 3, sink, sink,
            student_heads=4, student_kv_heads=2, teacher_heads=4, teacher_kv_heads=2,
            n_rel=4, causal=True,
        )
        self.assertLess(float(loss.abs()), 1e-5)
        heads = relation_heads(q, 4)
        mask = column_mask(torch.arange(5, 8), torch.arange(8), sink, causal=True)
        rel = _relation(heads[:, 5:], heads, mask)
        self.assertTrue(torch.allclose(rel.sum(-1), torch.ones(rel.shape[:-1]), atol=1e-5))
        self.assertTrue(torch.all(rel[:, :, 1] == 0))

    def test_pre_rope_hook_matches_recomputed_projection(self) -> None:
        class Attn(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.q_proj = nn.Linear(4, 4, bias=False)
                self.k_proj = nn.Linear(4, 4, bias=False)
                self.v_proj = nn.Linear(4, 4, bias=False)

            def forward(self, hidden):
                return self.q_proj(hidden), self.k_proj(hidden), self.v_proj(hidden)

        attn = Attn()
        cap = QKVCapture()
        handles = attach_qkv(attn, cap, pre_rope=True)
        hidden = torch.randn(1, 3, 4)
        try:
            q, _k, _v = attn(hidden)
        finally:
            for handle in handles:
                handle.remove()
        self.assertTrue(torch.equal(cap.q, q))
        self.assertTrue(torch.equal(cap.q, attn.q_proj(hidden)))

    def test_stop_forward_aborts_the_rest_of_the_stack(self) -> None:
        class Block(nn.Module):
            def __init__(self, seen: list[int], index: int) -> None:
                super().__init__()
                self.seen = seen
                self.index = index

            def forward(self, x):
                self.seen.append(self.index)
                return x

        seen: list[int] = []
        stack = nn.Sequential(Block(seen, 0), Block(seen, 1))
        handle = stack[0].register_forward_hook(lambda *_a: (_ for _ in ()).throw(StopForward()))
        try:
            with self.assertRaises(StopForward):
                stack(torch.zeros(1))
        finally:
            handle.remove()
        self.assertEqual(seen, [0])


if __name__ == "__main__":
    unittest.main()
