#!/bin/bash
# E3 on test: merge the LoRA adapter, serve it with vLLM, extract into e3_pred.
# Same prompt and sampling as E2, so the two are directly comparable.
#   sbatch scripts/e3_eval.sh
# Do not run while another job writes Data/processed/test/manifest.jsonl.
#SBATCH --job-name=e3-eval
#SBATCH --partition=cn2_madhava
#SBATCH --account=CHANGE_ME_optml_account
#SBATCH --qos=madhava
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

RUN=checkpoints/e3

cd ~/-Knowledge-Distillation-of-VLM
export PATH="$HOME/-Knowledge-Distillation-of-VLM/.venv/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
echo "Running on: $(hostname)"

[ -d "$RUN/merged" ] || python -m ie643.train.merge --adapter "$RUN/adapter" --out "$RUN/merged" || exit 1

vllm serve "$RUN/merged" --served-model-name e3 \
  --max-model-len 16384 --gpu-memory-utilization 0.5 > logs/vllm-$SLURM_JOB_ID.log 2>&1 &
VLLM_PID=$!
for i in $(seq 1 240); do
  curl -s http://127.0.0.1:8000/v1/models > /dev/null && break
  kill -0 $VLLM_PID 2>/dev/null || break
  sleep 5
done
if ! curl -s http://127.0.0.1:8000/v1/models > /dev/null; then
  echo "vLLM did not start - root cause from logs/vllm-$SLURM_JOB_ID.log:"
  grep -E "EngineCore.*(Error|error|Exception|Assertion)" logs/vllm-$SLURM_JOB_ID.log | head -15
  kill $VLLM_PID 2>/dev/null
  exit 1
fi

python -m ie643.teacher --manifest Data/processed/test/manifest.jsonl \
  --model e3 --pred-key e3_pred \
  --repetition-penalty 1.15 --max-tokens 4096 --workers 16 --timeout 600

kill $VLLM_PID
