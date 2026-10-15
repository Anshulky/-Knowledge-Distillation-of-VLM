"""Config, sampling weights, collate, LoRA targets, and the 8/7 teacher resize."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ie643.distill.config import load_config
from ie643.distill.data import collate, row_probabilities, teacher_pixel_size, value_mask_for_output
from ie643.distill.model import lora_target_names
from ie643.eval.bootstrap import bootstrap_diff


class Tok:
    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return "".join(chr(int(i)) for i in ids)


class Data(unittest.TestCase):
    def test_option_b_grid_matches_on_random_packed_sizes(self) -> None:
        self.assertEqual(teacher_pixel_size(812, 1204), (928, 1376))
        rng = np.random.default_rng(643)
        for _ in range(20):
            width = int(rng.integers(1, 40)) * 28
            height = int(rng.integers(1, 40)) * 28
            new_w, new_h = teacher_pixel_size(width, height)
            self.assertEqual(new_w % 32, 0)
            self.assertEqual(new_h % 32, 0)
            self.assertEqual((height // 28, width // 28), (new_h // 32, new_w // 32))

    def test_dataset_weights_follow_sqrt_counts(self) -> None:
        datasets = ["fatura2"] * 100 + ["sroie"] * 25
        probs = row_probabilities(datasets, power=0.5, weighting=True)
        # P(dataset) ∝ sqrt(n), so a fatura row is less likely than a sroie row.
        self.assertAlmostEqual(probs[:100].sum() / probs[100:].sum(), (100 ** 0.5) / (25 ** 0.5), places=5)
        self.assertGreater(probs[100], probs[0])
        uniform = row_probabilities(datasets, power=0.5, weighting=False)
        self.assertTrue(np.allclose(uniform, 1 / len(datasets)))

    def test_value_mask_skips_null_and_eos(self) -> None:
        text = '{"a":1,"b":null}'
        ids = [ord(ch) for ch in text] + [151645]
        mask = value_mask_for_output(Tok(), ids, 151645)
        self.assertEqual(mask[text.index("1")], 1)
        self.assertEqual(mask[text.index("null") : text.index("null") + 4], [0, 0, 0, 0])
        self.assertEqual(mask[-1], 0)

    def test_collate_pads_labels_and_ids(self) -> None:
        def item(length: int) -> dict:
            return {
                "uid": f"u{length}",
                "input_ids": torch.arange(length),
                "attention_mask": torch.ones(length),
                "labels": torch.arange(length),
                "kd_mask": torch.ones(length),
                "value_mask": torch.zeros(length),
                "pixel_values": torch.zeros(2, 3),
                "image_grid_thw": torch.tensor([1, 2, 2]),
                "topk_ids": torch.zeros(length, 2, dtype=torch.long),
                "topk_logprobs": torch.zeros(length, 2),
                "n_prompt": 1,
                "n_out": length - 1,
                "student_grid_hw": torch.tensor([2, 2]),
            }

        batch = collate([item(3), item(5)], pad_id=99)
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 5))
        self.assertEqual(int(batch["input_ids"][0, -1]), 99)
        self.assertEqual(int(batch["labels"][0, -1]), -100)
        self.assertEqual(tuple(batch["pixel_values"].shape), (4, 3))

    def test_configs_load_headline_defaults(self) -> None:
        root = Path(__file__).resolve().parents[1]
        e3 = load_config(root / "configs/e3_sft.yaml")
        e4 = load_config(root / "configs/e4_kd.yaml")
        e5 = load_config(root / "configs/e5_minilm.yaml")
        self.assertFalse(e3.kd.enabled)
        self.assertFalse(e3.rel.enabled)
        self.assertEqual(e3.optim.micro_batch, 4)
        self.assertEqual(e3.lora.r, 16)
        self.assertTrue(e3.data.dataset_weighting)
        self.assertTrue(e4.kd.enabled)
        self.assertEqual(e4.kd.div, "fkl")
        self.assertEqual(e4.kd.weight, "entropy")
        self.assertEqual(e4.kd.alpha, 0.5)
        self.assertTrue(e5.rel.enabled)
        self.assertEqual(e5.rel.student_layer, 36)
        self.assertEqual(e5.rel.teacher_layer, 48)
        self.assertEqual(e5.rel.resize, "B")
        self.assertEqual(e5.optim.micro_batch * e5.optim.grad_accum, 32)
        with self.assertRaises(ValueError):
            load_config_bad = root / "configs/e3_sft.yaml"
            text = load_config_bad.read_text(encoding="utf-8") + "\nnot_a_field: 1\n"
            tmp = root / "configs/_bad.yaml"
            tmp.write_text(text, encoding="utf-8")
            try:
                load_config(tmp)
            finally:
                tmp.unlink()

    def test_lora_targets_skip_vision(self) -> None:
        class Tree(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.visual = nn.Module()
                self.visual.q_proj = nn.Linear(2, 2)
                self.visual.merger = nn.Module()
                self.visual.merger.mlp = nn.Module()
                self.visual.merger.mlp.gate_proj = nn.Linear(2, 2)
                self.language_model = nn.Module()
                self.language_model.layers = nn.ModuleList([nn.Module()])
                layer = self.language_model.layers[0]
                layer.self_attn = nn.Module()
                layer.self_attn.q_proj = nn.Linear(2, 2)
                layer.mlp = nn.Module()
                layer.mlp.gate_proj = nn.Linear(2, 2)

        model = Tree()
        names = lora_target_names(model, ("q_proj", "gate_proj"), merger=False)
        self.assertIn("language_model.layers.0.self_attn.q_proj", names)
        self.assertIn("language_model.layers.0.mlp.gate_proj", names)
        self.assertNotIn("visual.q_proj", names)
        self.assertNotIn("visual.merger.mlp.gate_proj", names)
        with_merger = lora_target_names(model, ("q_proj", "gate_proj"), merger=True)
        self.assertIn("visual.merger.mlp.gate_proj", with_merger)

    def test_bootstrap_ci_contains_the_point_difference(self) -> None:
        left = [("sroie", 1, 0, 0), ("sroie", 0, 1, 1)]
        right = [("sroie", 1, 0, 0), ("sroie", 1, 0, 0)]
        ci = bootstrap_diff(left, right, n=200, seed=643)
        self.assertGreater(ci["mean"], 0)
        self.assertLessEqual(ci["lo"], ci["mean"])
        self.assertGreaterEqual(ci["hi"], ci["mean"])


if __name__ == "__main__":
    unittest.main()
