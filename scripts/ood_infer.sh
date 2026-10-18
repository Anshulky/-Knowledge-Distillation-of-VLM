#!/bin/bash
# Inference on the evaluation-only datasets (DocILE + MC-OCR, Data/processed/ood) with any model.
# Same prompt and sampling as E2/E3. Pick the model and the field to write:
#   MODEL=Qwen/Qwen2.5-VL-3B-Instruct PRED_KEY=student_pred sbatch scripts/ood_infer.sh   # zero-shot student
#   MODEL=checkpoints/e3/merged       PRED_KEY=e3_pred      sbatch scripts/ood_infer.sh   # E3
# MODEL may also be a LoRA adapter folder (has adapter_config.json): it is merged first into
# the sibling folder merged/ (once), e.g. MODEL=checkpoints/kd_none/adapter -> checkpoints/kd_none/merged.
# Each run works in its own folder, Data/processed/ood_$PRED_KEY/ (manifest copy + link to the
# shared images) and its own port, so several runs can go at the same time.
# Then score: python -m ie643.eval --manifest Data/processed/ood_$PRED_KEY/manifest.jsonl --pred-key $PRED_KEY
#SBATCH --job-name=ood-infer
#SBATCH --partition=cn2_madhava
#SBATCH --account=optml_proj1
#SBATCH --qos=madhava
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

: "${MODEL:?set MODEL, e.g. MODEL=Qwen/Qwen2.5-VL-3B-Instruct}"
: "${PRED_KEY:?set PRED_KEY, e.g. PRED_KEY=student_pred}"
RUN_DIR=Data/processed/ood_$PRED_KEY
MANIFEST=$RUN_DIR/manifest.jsonl
PORT=$((8000 + SLURM_JOB_ID % 1000))  # jobs sharing a node must not share a port

cd ~/-Knowledge-Distillation-of-VLM
export PATH="$HOME/-Knowledge-Distillation-of-VLM/.venv/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
echo "Running on: $(hostname)  model=$MODEL  pred_key=$PRED_KEY  dir=$RUN_DIR  port=$PORT"

if [ -f "$MODEL/adapter_config.json" ]; then
  MERGED="$(dirname "$MODEL")/merged"
  [ -f "$MERGED/config.json" ] || python -m ie643.train.merge --adapter "$MODEL" --out "$MERGED" || exit 1
  MODEL="$MERGED"
  echo "serving merged adapter: $MODEL"
fi

# First run with this PRED_KEY: copy the packed manifest; a rerun resumes from the existing copy.
if [ ! -f "$MANIFEST" ]; then
  mkdir -p "$RUN_DIR"
  cp Data/processed/ood/manifest.jsonl Data/processed/ood/pack.json "$RUN_DIR"/
  ln -s ../ood/images "$RUN_DIR/images"
fi

vllm serve "$MODEL" --served-model-name ood-model --port $PORT \
  --max-model-len 16384 --gpu-memory-utilization 0.5 > logs/vllm-$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
for i in $(seq 1 720); do  # up to 60 min: weight loading from the shared home disk can take >10 min
  curl -s http://127.0.0.1:$PORT/v1/models > /dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || break
  sleep 5
done
if ! curl -s http://127.0.0.1:$PORT/v1/models > /dev/null; then
  echo "vLLM did not start - root cause from logs/vllm-$SLURM_JOB_ID.log:"
  grep -E "EngineCore.*(Error|error|Exception|Assertion)" logs/vllm-$SLURM_JOB_ID.log | head -15
  kill $VLLM_PID 2>/dev/null
  exit 1
fi

python -m ie643.teacher --manifest "$MANIFEST" --base-url http://127.0.0.1:$PORT/v1 \
  --model ood-model --pred-key "$PRED_KEY" \
  --repetition-penalty 1.15 --max-tokens 4096 --workers 16 --timeout 600

kill $VLLM_PID
