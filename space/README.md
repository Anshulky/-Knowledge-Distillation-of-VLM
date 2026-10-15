---
title: IE643 Receipt Extractor
emoji: 🧾
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.29.0
app_file: app.py
pinned: false
license: other
license_name: qwen-research
license_link: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/LICENSE
models:
  - Qwen/Qwen2.5-VL-3B-Instruct
  - Anshulky/ie643-qwen2.5vl-3b-kd-wnone
suggested_hardware: t4-small
short_description: Receipt/invoice to JSON with a distilled 3B Qwen2.5-VL
---

# IE643 receipt and invoice extractor

Upload a receipt or invoice photo. A **3B Qwen2.5-VL student**, distilled from **Qwen3-VL-32B**, returns one JSON
object with the IE643 schema: vendor, address, tax id, dates, currency, line items, taxes, totals, payment, …
A field is filled only when it is printed on the image.

## Model

- Base: `Qwen/Qwen2.5-VL-3B-Instruct` + LoRA adapter `Anshulky/ie643-qwen2.5vl-3b-kd-wnone` (r 16, α 32; merged at start-up).
- Training: 18,770 receipts and invoices (SROIE, CORD-v2, FATURA, invoices-and-receipts v2, synthetic receipts) labelled by
  the 32B teacher; cross-entropy on the teacher's tokens + top-20 logit KD (α 0.5, τ 1, forward KL, uniform token weights);
  2 epochs, seed 643.
- Frozen test set (4,240 images, cleaned GT): micro F1 **96.4** (teacher 96.5, same 3B without training 79.0).

## Inference (same as evaluation)

- Prompt: the hint-free IE643 instruction (`vendor/ie643/schema.py`), image first, then text.
- Image: EXIF fix, RGB, Qwen `smart_resize` to ≤ 1,003,520 px with sides a multiple of 28 (LANCZOS), as in training.
- Decoding: greedy, repetition penalty 1.15, up to 2,048 new tokens. Unlike the vLLM evaluation, there is no
  JSON-constrained decoding; on test images the fields match the vLLM outputs almost exactly.

## Setup notes

- **Secret `HF_TOKEN`** (Settings → Variables and secrets): a read token from an account with access to the gated adapter.
- Variables (optional): `ADAPTER` (another adapter repo, e.g. `Anshulky/ie643-qwen2.5vl-3b-sft-e3a`), `MAX_NEW_TOKENS`,
  `MERGE_ADAPTER=0` to keep the adapter unmerged.
- Hardware: a GPU (T4 small or better; bf16 on Ampere+, fp16 on T4). ZeroGPU also works (`spaces.GPU` is applied).
  On CPU a single receipt takes minutes.
- `vendor/` is copied from the training repo by `sync_vendor.py`; do not edit it here.

## Limitations

Research demo. It can misread digits on blurry photos, miss line items on long receipts, and return wrong fields on
document types far from the training data. Do not use the output for accounting without checking it. Use is subject
to the Qwen Research License (non-commercial) of the base model and the licences of the training datasets.
