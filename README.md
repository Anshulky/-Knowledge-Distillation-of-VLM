# IE643 — receipt / invoice information extraction

A **Qwen3-VL-32B** teacher reads receipt and invoice images into one shared JSON schema (26 fields, `ie643.schema.FIELD_ORDER`). A **Qwen2.5-VL-3B** student is then trained with LoRA, either on the teacher's labels (SFT / KD) or on the dataset labels (E3). Attention rollout shows which pixels each field came from.

```
splits  →  pack (pre-process)  →  teacher labels  →  train  →  eval  →  rollout  →  demo
```

## Setup

```bash
git clone https://github.com/Anshulky/-Knowledge-Distillation-of-VLM.git ie643   # folder name avoids the leading "-"
cd ie643
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh                 # install uv if missing
uv sync                                  # creates .venv (Python 3.12) and installs everything in uv.lock
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0     # FlashInfer cannot read SM 12.0 on this box
hf auth login                            # only for the gated adapters (demo)
```

`uv sync` installs the Python packages and the `ie643` package itself in editable mode. It cannot install the system parts: the NVIDIA driver / CUDA, and the `python3.12-dev` headers that vLLM needs to compile. vLLM is Linux-only, so run this on the GPU server, not on Windows.

Hardware: two GPUs for the 32B teacher (tensor parallel 2), and one GPU for 3B training, serving, and rollout. All commands are run from the repo root with `.venv` active.

## Data

Raw datasets are not in the repo (about 17 GB). The splits only need the **images**, because the labels are already stored as `source_gt` in `splits/*.json`. Put the datasets under `Data/` (in-domain) and `external_dataset/` (OOD) at exactly these paths, since the splits refer to files by relative path:

| Dataset | Source | Expected path |
|---|---|---|
| CORD-v2 | HF `naver-clova-ix/cord-v2` | `Data/naver-clova-ix/cord-v2/data/*.parquet` |
| invoices-and-receipts v2 | HF `mychen76/invoices-and-receipts_ocr_v2` | `Data/mychen76/invoices-and-receipts_ocr_v2/data/*.parquet` |
| synthetic-receipts-ocr | HF `albertobarnabo/synthetic-receipts-ocr` | `Data/albertobarnabo/synthetic-receipts-ocr/data/*.parquet` |
| FATURA | [Zenodo 10371464](https://zenodo.org/records/10371464) (`FATURA2.zip`) | `Data/FATURA2/invoices_dataset_final/images/` |
| SROIE 2019 | [ICDAR 2019 SROIE](https://rrc.cvc.uab.es/?ch=13) | `Data/SROIE/SROIE2019-20260919T110512Z-1-001/SROIE2019/0325updated.task2train(626p)/` (train), `Data/SROIE/SROIE_test_images_task_3/` (test) |
| DocILE | [docile.rossum.ai](https://docile.rossum.ai) (registration) | `external_dataset/docile/pdfs/` (+ `annotations/`, `trainval.json`) |
| MC-OCR 2021 | [CodaLab 27798](https://competitions.codalab.org/competitions/27798) | `external_dataset/mcocr_train_data/train_images/`, `mcocr_train_df.csv` |
| ReceiptBench | [github.com/wwwT0ri/ReceiptBench](https://github.com/wwwT0ri/ReceiptBench) | `external_dataset/ReceiptBench/ReceiptBench_image_v1/ReceiptBench_image_v1/` (+ `ReceiptBench_json_v1/`) |

```bash
for r in naver-clova-ix/cord-v2 mychen76/invoices-and-receipts_ocr_v2 albertobarnabo/synthetic-receipts-ocr; do
  hf download $r --repo-type dataset --local-dir Data/$r
done
```

Parquet records point at a specific file name and `row_index`. If a Hugging Face dataset has been re-uploaded since then, its file names change, and `ie643.pack` reports the missing file. GT files (SROIE labels, FATURA JSON, DocILE annotations, and so on) are needed only to rebuild the splits, not to pack or evaluate.

## Quick check

The full pipeline needs about 17 GB of raw data, two large GPUs for the 32B teacher, and about 9.5 h per training run. The quick check is a small test that exercises every stage of that pipeline end to end, without the cost. It confirms that on a fresh clone:

- the environment installs and the scorer passes its unit tests;
- the frozen splits still resolve to the downloaded images (the Hugging Face parquet file names have not changed);
- packing, vLLM serving, extraction with the shared prompt, and scoring all run and produce a sensible F1 (zero-shot 3B scores about 70 on CORD).

If it passes, the full runs differ only in data size and model. It runs on one GPU in about 15 minutes and needs only CORD (about 2 GB).

**Model used:** the **zero-shot** student, base `Qwen/Qwen2.5-VL-3B-Instruct` with no adapter. It is public, so the check needs no Hugging Face approval or trained checkpoint. The distilled student is optional and shown after the block.

```bash
python -m unittest discover tests                    # 31 tests, CPU, < 1 s
hf download naver-clova-ix/cord-v2 --repo-type dataset --local-dir Data/naver-clova-ix/cord-v2
python -m ie643.pack --split test --out Data/processed/quick --datasets cord-v2 --limit 20

.venv/bin/vllm serve Qwen/Qwen2.5-VL-3B-Instruct --max-model-len 16384 --gpu-memory-utilization 0.5 &
until curl -s http://127.0.0.1:8000/v1/models > /dev/null; do sleep 5; done   # wait for the server
python -m ie643.teacher --manifest Data/processed/quick/manifest.jsonl --model Qwen/Qwen2.5-VL-3B-Instruct \
  --pred-key student_pred --repetition-penalty 1.15 --max-tokens 4096
python -m ie643.eval --manifest Data/processed/quick/manifest.jsonl --pred-key student_pred
```

**Optional: the distilled student (KD w=none).** The adapter repo is gated, so request access and run `hf auth login` first. Then merge it, serve the merged model, and extract into its own key so the zero-shot results are kept. It scored about 92 on CORD versus about 70 zero-shot.

```bash
kill %1                                              # stop the zero-shot server
hf download Anshulky/ie643-qwen2.5vl-3b-kd-wnone --local-dir checkpoints/kd_wnone/adapter
python -m ie643.train.merge --adapter checkpoints/kd_wnone/adapter --out checkpoints/kd_wnone/merged
.venv/bin/vllm serve checkpoints/kd_wnone/merged --served-model-name kd \
  --max-model-len 16384 --gpu-memory-utilization 0.5 &
until curl -s http://127.0.0.1:8000/v1/models > /dev/null; do sleep 5; done
python -m ie643.teacher --manifest Data/processed/quick/manifest.jsonl --model kd \
  --pred-key kd_pred --repetition-penalty 1.15 --max-tokens 4096
python -m ie643.eval --manifest Data/processed/quick/manifest.jsonl --pred-key kd_pred
```

## 1. Splits

Frozen, leakage-safe uid lists (seed 643). They are committed to the repo, so **do not resample**. Rebuild them only if the protocol changes.

```bash
python -m ie643.splits --out splits --seed 643          # in-domain train/test
python -m ie643.splits --apply-exclusions               # re-apply content_duplicates.json to existing files
python -m ie643.splits.ood --data-root external_dataset --out splits_ood   # OOD (test only)
```

| Dataset | Train | Test | Notes |
|---|---:|---:|---|
| SROIE (task 3) | 717 | 347 | 8 train images without GT dropped |
| FATURA | 7,500 | 2,500 | dev merged into test |
| CORD-v2 | 773 | 200 | val merged into test |
| synthetic-receipts-ocr | 7,000 | 798 | 3k IT, 3k FR, 1k EN (US+UK) |
| invoices-and-receipts v2 | 2,781 | 395 | all of the official train set |
| **Total** | **18,771** | **4,240** | after removing 99 train duplicates of test images |
| OOD: DocILE / MC-OCR / ReceiptBench | — | 2,000 / 1,149 / 2,000 | evaluation only; `gt_alternatives` accepts other printed forms |

Each record in `splits/*.json` has `uid`, `split`, `source_split`, `image` (a file path, or a parquet `path/column/row_index` relative to `Data/`), and `source_gt` (the canonical JSON). `leakage_audit` must show `uid_overlap`, `locator_overlap`, and `content_overlap` all equal to 0. Validation data is never mixed into train.

## 2. Pre-process (image pack)

`ie643.pack` loads each uid, applies an EXIF transpose and converts to RGB, then runs Qwen `smart_resize` (sides are multiples of 28, area between 3,136 and 1,003,520 px, aspect ratio ≤ 200, LANCZOS). It writes a frozen JPEG (quality 92), or a PNG for grey/1-bit scans, plus `manifest.jsonl`. Teacher, student, and demo therefore all see **the same pixels**. Do not use `python -m ie643.preprocess` for this pool; it is a legacy CLI.

```bash
python -m ie643.pack --split test  --out Data/processed/test  --splits splits
python -m ie643.pack --split train --out Data/processed/train --splits splits
python -m ie643.pack --split test  --out Data/processed/ood   --splits splits_ood --data-root external_dataset
```

`--resume` skips images that already exist, `--limit N` runs a smoke test, and `--datasets sroie,cord-v2` packs a subset. Output layout: `pack.json`, `canonical_schema.json`, `manifest.jsonl`, `manifests/<dataset>.jsonl`, `images/<dataset>/<uid>.jpg|png`. Every manifest row keeps `source_gt` and the original and resized sizes.

## 3. Teacher labels (vLLM)

Serve one model at a time on port 8000. Wait for `Application startup complete` (check with `curl -s http://127.0.0.1:8000/v1/models`).

```bash
.venv/bin/vllm serve Qwen/Qwen3-VL-32B-Instruct --tensor-parallel-size 2 \
  --max-model-len 16384 --gpu-memory-utilization 0.85 \
  --max-logprobs 20 --logprobs-mode raw_logprobs
```

One pass over train stores the hard labels plus the top-20 soft labels (`soft/<dataset>/<uid>.npz`). The checks and the filter come next:

```bash
python -m ie643.teacher --manifest Data/processed/train/manifest.jsonl \
  --model Qwen/Qwen3-VL-32B-Instruct --pred-key teacher_gt \
  --repetition-penalty 1.15 --max-tokens 4096 \
  --logprobs 20 --workers 64 --checkpoint-every 64

python -m ie643.distill.check_soft   --manifest Data/processed/train/manifest.jsonl   # acceptance checks
python -m ie643.distill.label_filter --manifest Data/processed/train/manifest.jsonl   # -> label_filter.json (keep_uids)
```

`ie643.teacher` sends the image with `teacher_prompt()` only, never dataset hints. Decoding uses temperature 0, `json_object`, and repetition penalty 1.15 (never a frequency penalty, which breaks JSON quotes). The prediction is written to the manifest field named by `--pred-key`, so runs never overwrite each other. Useful flags: `--overwrite`, `--errors-only`, `--uids a,b`, and `--base-url` (for a port other than 8000). There is no fallback to `source_gt`.

## 4. Train (student, LoRA)

Every run uses the same setup: Qwen2.5-VL-3B with LoRA r 16 / α 32 / dropout 0.05 on q/k/v/o/gate/up/down of the LLM, with the vision encoder and merger frozen. Training is lr 1e-4 cosine, batch 4 × 8, 2 epochs, max length 4,096, dataset sampling ∝ n^0.5, seed 643, and no validation split. Only the targets differ.

```bash
# Teacher-label runs (configs/*.yaml) -> runs/<exp>/<variant>/seed<k>/final
python -m ie643.distill smoke --config configs/e4_kd_wnone.yaml       # loss must fall and stay finite
python -m ie643.distill train --config configs/e3_sft.yaml            # SFT: CE on teacher hard labels
python -m ie643.distill train --config configs/e4_kd_wnone.yaml       # KD (final)

# E3-v3: CE on dataset labels only (source_gt, no teacher), 1% masking rule -> checkpoints/e3_v3/adapter
python -m ie643.train --manifest Data/processed/train/manifest.jsonl --limit 32 --grad-accum 2 \
  --save-steps 4 --out checkpoints/e3_smoke                           # smoke test, a few minutes
python -m ie643.train --manifest Data/processed/train/manifest.jsonl --out checkpoints/e3_v3
```

KD loss: `α·CE(teacher token) + (1−α)·Σ w_t·KL₂₁(p_T‖p_S) / Σ w_t`, with α = 0.5 and τ = 1, over the top-20 tokens plus one tail bucket. In E3, values in `source_gt` are always trained. An empty field is trained as `null` only if that dataset fills the field in at least 1% of its rows; fields the dataset never labels are masked. Add `--resume` to continue an interrupted `distill train`. Each run takes about 9.5 h on one GPU (20 GB peak).

## 5. Evaluate

Merge an adapter into the base model, serve it with vLLM, extract predictions into a new `--pred-key`, then score them against `source_gt`. The prompt and decoding are identical for every model.

```bash
# 32B teacher (reference row): with the teacher server from step 3 still running
python -m ie643.teacher --manifest Data/processed/test/manifest.jsonl --model Qwen/Qwen3-VL-32B-Instruct \
  --pred-key teacher_gt --repetition-penalty 1.15 --max-tokens 4096 --workers 64
python -m ie643.eval --manifest Data/processed/test/manifest.jsonl --pred-key teacher_gt
# (same two commands with Data/processed/ood/manifest.jsonl for the OOD column)

# Merge once (works for any adapter folder: checkpoints/e3_v3/adapter, runs/e4/wnone/seed643/final, ...)
python -m ie643.train.merge --adapter checkpoints/e3_v3/adapter --out checkpoints/e3_v3/merged

.venv/bin/vllm serve checkpoints/e3_v3/merged --served-model-name e3 \
  --max-model-len 16384 --gpu-memory-utilization 0.5
# zero-shot: .venv/bin/vllm serve Qwen/Qwen2.5-VL-3B-Instruct --max-model-len 16384 --gpu-memory-utilization 0.5

# In-domain test
python -m ie643.teacher --manifest Data/processed/test/manifest.jsonl --model e3 --pred-key e3v3_pred \
  --repetition-penalty 1.15 --max-tokens 4096 --workers 16 --timeout 600
python -m ie643.eval --manifest Data/processed/test/manifest.jsonl --pred-key e3v3_pred

# OOD: same extract on Data/processed/ood/manifest.jsonl (copy the manifest first if several runs share it)
python -m ie643.teacher --manifest Data/processed/ood/manifest.jsonl --model e3 --pred-key e3v3_pred \
  --repetition-penalty 1.15 --max-tokens 4096 --workers 16 --timeout 600
python -m ie643.eval --manifest Data/processed/ood/manifest.jsonl --pred-key e3v3_pred

# Paired bootstrap CI between two runs (second minus first), and coverage vs a reference
python -m ie643.eval.bootstrap --manifest Data/processed/test/manifest.jsonl --pred-keys e3_pred,kd_pred
python -m ie643.eval --manifest Data/processed/test/manifest.jsonl --pred-key e3v3_pred \
  --ref-manifest Data/processed/test/manifest.jsonl --ref-key teacher_gt --coverage
```

Two runs must not write to the same manifest at the same time. To run several models in parallel, give each its own copy of the manifest, a different `--port`, and a matching `--base-url http://127.0.0.1:<port>/v1`.

**Scorer v3:** micro F1 is computed over fields that `source_gt` fills. Amounts match numerically to the cent (`54,000` = 54000, `1.234,56` = 1234.56). List rows match on name plus amount. GT cleaning is on by default (`--raw-gt` turns it off): placeholders such as `[]` and `-` count as empty, FATURA `payment_method` is not scored, and invoices-v2 dict text is unpacked. Other flags: `--strict` and `--count-missing`. Regression tests are in `tests/test_eval_match.py`.

| Model | Params | Trained on | In-domain micro | Macro | Real | Templ. | Unseen micro | Unseen macro |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Teacher (32B) | 33.4B | — (zero-shot) | 96.53 | 93.87 | 90.54 | 97.50 | 84.7† | 87.9† |
| Student zero-shot | 3.75B | — | 77.93* | 77.24 | 76.97 | 78.09 | 69.85 | 75.94 |
| E3-v3 | 3.75B | dataset labels | **97.86** | **92.79** | 86.99 | **99.62** | 46.64 | 55.54 |
| SFT | 3.75B | teacher, hard | 96.22 | 92.26 | 87.65 | 97.60 | 74.28 | 80.54 |
| KD (final) | 3.75B | teacher, hard + soft | 96.33 | 92.67 | **88.32** | 97.63 | **75.05** | **81.06** |

Micro / macro F1 (%). Bold: best student in each column. Macro = mean of the per-dataset F1 scores. Real = SROIE, CORD and invoices-v2; Templ. = FATURA and synthetic receipts. Unseen = DocILE, MC-OCR and ReceiptBench.
† The teacher's unseen-data scores were computed with the scorer version before the final amount-matching fix; with that version the students scored 1.8–2.2 points higher, so the teacher's lead on unseen data is probably overstated by about 2 points.
\* The zero-shot student's in-domain score was obtained with the earlier prompt that still contained the document-type and language hints; all other runs use the final hint-free prompt.

## 6. Attention rollout

This step uses HF transformers with eager attention, because vLLM cannot return attention, so stop vLLM first. Rollout runs on each model's own stored output, its `--raw-key`. An extract with `--pred-key kd_pred` stores that output as `kd_raw`.

```bash
U="cord::test:35,sroie::X51006620437"
python -m ie643.rollout run --model Qwen/Qwen2.5-VL-3B-Instruct --raw-key student_raw --tag student --uids "$U"
python -m ie643.rollout run --model Qwen/Qwen3-VL-32B-Instruct  --raw-key teacher_raw --tag teacher --uids "$U"
python -m ie643.rollout run --model Qwen/Qwen2.5-VL-3B-Instruct --adapter runs/e4/wnone/seed643/final \
  --raw-key kd_raw --tag kd --uids "$U"                              # a trained checkpoint
python -m ie643.rollout report                                        # -> Data/rollout/report/
python -m ie643.rollout.compare                                       # checkpoints side by side -> Data/rollout_cmp/report/
```

Attention sinks are excluded before rolling out: every token before the first image token, plus image tokens whose attention is more than 10 × the median (`--sink-ratio`). Per-field maps average the rollout rows of that value's tokens. `--key rollout_full` gives the classic N×N rollout, kept only as an ablation.

## 7. Review and demo

```bash
python -m ie643.review --root Data/processed/test        # side-by-side review UI at http://127.0.0.1:8765
```

`space/` is the Gradio demo: base model plus the KD adapter, merged at startup, with the same prompt, `smart_resize`, and parser as training. It is hosted at [huggingface.co/spaces/Anshulky/ie643-receipt-extractor](https://huggingface.co/spaces/Anshulky/ie643-receipt-extractor) on ZeroGPU. The adapter repos (`Anshulky/ie643-qwen2.5vl-3b-kd-wnone`, `…-sft-e3a`) are gated, so either run `hf auth login` or point `ADAPTER` at a local folder.

```bash
python space/sync_vendor.py                              # copy schema / parser / image code into space/vendor
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 python space/app.py
# ADAPTER=checkpoints/e3_v3/adapter python space/app.py  # any local adapter
# from a laptop: ssh -L 7860:<gpu-node>:7860 <user>@<login-node>, then open http://localhost:7860
```

## Layout

```
src/ie643/
  schema.py     canonical keys + teacher prompt
  splits/       uid JSON builder (+ ood.py for external sets)
  pack/         frozen image packs from splits
  preprocess/   dataset adapters, smart_resize
  teacher/      vLLM client, parse / repair (used for every model)
  distill/      soft-label checks, label filter, SFT / KD training
  train/        E3: CE on source_gt, adapter merge
  eval/         scorer, GT cleaning, bootstrap
  rollout/      attention rollout, reports, checkpoint comparison
  review/       HTML review UI
configs/        distill run configs
space/          Gradio demo (HF Space)
splits/ splits_ood/   frozen uid JSON (committed)
Data/processed/{train,test,ood}/   packed images + manifests (gitignored)
```
