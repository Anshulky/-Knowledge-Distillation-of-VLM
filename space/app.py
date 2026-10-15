"""Gradio demo: upload a receipt or invoice, get the IE643 extraction JSON from the distilled 3B student.

Model = BASE_MODEL + the LoRA adapter ADAPTER (KD w=none headline by default), merged once at start-up.
The adapter repo is gated: the Space needs a read token with access in the HF_TOKEN secret.

  python space/sync_vendor.py           # once, from the training repo
  HF_TOKEN=... python space/app.py      # local run
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract import parse_output, prepare_image, teacher_prompt  # noqa: E402

BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
ADAPTER = os.environ.get("ADAPTER", "Anshulky/ie643-qwen2.5vl-3b-kd-wnone")
MERGE = os.environ.get("MERGE_ADAPTER", "1") == "1"
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "2048"))
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", "1.15"))  # same as the test-set evaluation
TOKEN = os.environ.get("HF_TOKEN")  # Space secret; needed because the adapter repo is gated

SUMMARY_FIELDS = (
    ("Vendor", "vendor"),
    ("Address", "vendor_address"),
    ("Phone", "vendor_phone"),
    ("Tax id", "vendor_tax_id"),
    ("Buyer", "buyer"),
    ("Date", "date"),
    ("Time", "time"),
    ("Currency", "currency"),
    ("Receipt no.", "receipt_number"),
    ("Invoice no.", "invoice_number"),
    ("Subtotal", "subtotal"),
    ("Tax", "tax"),
    ("Discount", "discount"),
    ("Total", "total"),
    ("Change", "change"),
    ("Payment", "payment_method"),
)

_model = None
_processor = None


def _dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16  # T4 has no bf16


def get_model():
    global _model, _processor
    if _model is None:
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        if ADAPTER and not TOKEN and os.environ.get("SPACE_ID"):  # locally the cached `hf auth login` token is used
            print(f"WARNING: HF_TOKEN is not set; the gated adapter {ADAPTER} will fail to download. "
                  "Add a read token as the Space secret HF_TOKEN.", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(BASE_MODEL, dtype=_dtype(), token=TOKEN)
        if ADAPTER:
            # torch_device="cpu": PEFT would otherwise load the adapter tensors straight onto CUDA because
            # torch.cuda.is_available() is True on ZeroGPU, but no real GPU exists at start-up
            model = PeftModel.from_pretrained(model, ADAPTER, token=TOKEN, torch_device="cpu")
            if MERGE:
                model = model.merge_and_unload()  # plain model: no LoRA overhead per token
        if torch.cuda.is_available():
            model = model.to("cuda")  # ZeroGPU: allowed at import time; the GPU is attached per call
        _model = model.eval()
        _processor = AutoProcessor.from_pretrained(BASE_MODEL, token=TOKEN)
    return _model, _processor


def _summary(parsed: dict[str, Any], info: str) -> str:
    lines = [info]
    doc, lang = parsed.get("document_type"), parsed.get("language")
    if doc or lang:
        lines.append(f"**Document:** {doc or '—'} · **Language:** {lang or '—'}")
    for label, key in SUMMARY_FIELDS:
        value = parsed.get(key)
        if value not in (None, "", [], {}):
            lines.append(f"**{label}:** {value}")
    return "\n\n".join(lines)


def _items(parsed: dict[str, Any]) -> list[list[Any]]:
    rows = []
    for item in parsed.get("line_items") or []:
        if isinstance(item, dict):
            rows.append([item.get(k) if item.get(k) is not None else "" for k in ("name", "quantity", "unit_price", "amount")])
    return rows


def _generate(image: Image.Image) -> tuple[str, tuple[int, int], tuple[int, int], float, int]:
    model, processor = get_model()
    seen, orig, size = prepare_image(image)
    # identical to training (ie643.distill.data): student chat template, image first, then the hint-free prompt
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": teacher_prompt()}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[chat], images=[seen], return_tensors="pt")
    device = next(model.parameters()).device
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(**batch, do_sample=False, max_new_tokens=MAX_NEW_TOKENS,
                             repetition_penalty=REPETITION_PENALTY)
    elapsed = time.time() - t0
    new_tokens = out[:, batch["input_ids"].shape[1]:]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text, orig, size, elapsed, int(new_tokens.shape[1])


try:  # ZeroGPU: borrow a GPU only while generating
    import spaces

    _generate = spaces.GPU(duration=120)(_generate)
except ImportError:
    pass

if os.environ.get("EAGER_LOAD", "1") == "1":
    # Download + merge once at start-up, outside the GPU-timed call (first request would otherwise
    # spend its ZeroGPU time budget on a 7.5 GB download and the merge).
    get_model()


def extract(image: Image.Image | None):
    if image is None:
        raise gr.Error("Upload a receipt or invoice image.")
    text, orig, size, elapsed, n_tok = _generate(image)
    info = (f"**Model:** `{ADAPTER or BASE_MODEL}` · image {orig[0]}×{orig[1]} → {size[0]}×{size[1]} px · "
            f"{n_tok} tokens in {elapsed:.1f} s")
    try:
        parsed = parse_output(text)
    except ValueError as exc:
        return f"{info}\n\nCould not parse JSON ({exc}). The raw text is below.", "", [], text
    return _summary(parsed, info), json.dumps(parsed, ensure_ascii=False, indent=2), _items(parsed), text


def build() -> gr.Blocks:
    with gr.Blocks(title="IE643 receipt extractor") as demo:
        gr.Markdown(
            "# Receipt and invoice extraction\n\n"
            "A 3B Qwen2.5-VL student distilled from Qwen3-VL-32B (LoRA, top-20 logit KD) reads the image into the "
            "IE643 JSON schema. Fields are filled only when printed. Research demo; outputs can be wrong."
        )
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Receipt or invoice", sources=["upload", "clipboard"])
                button = gr.Button("Extract", variant="primary")
                examples = sorted((Path(__file__).resolve().parent / "examples").glob("*"))
                if examples:
                    gr.Examples([[str(p)] for p in examples], inputs=[image])
            with gr.Column():
                summary = gr.Markdown()
                items = gr.Dataframe(headers=["name", "quantity", "unit_price", "amount"],
                                     datatype=["str"] * 4, label="Line items", interactive=False)
        with gr.Accordion("JSON", open=True):
            payload = gr.Code(language="json", label="Canonical JSON")
        with gr.Accordion("Raw model text", open=False):
            raw = gr.Textbox(label="Raw output", lines=8)
        button.click(extract, [image], [summary, payload, items, raw])
    return demo


if __name__ == "__main__":
    build().queue(max_size=16).launch()
