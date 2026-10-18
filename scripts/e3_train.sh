#!/bin/bash
# E3-v2: student + CE only on source_gt, same conditions as the teacher-label SFT / KD runs
# (LoRA on the LLM, vision encoder and merger frozen; defaults in src/ie643/train/cli.py).
# Submit from the repo root:
#   mkdir -p logs && sbatch scripts/e3_train.sh
#SBATCH --job-name=e3-train
#SBATCH --partition=cn2_madhava
#SBATCH --account=optml_proj1
#SBATCH --qos=madhava
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

# Smoke test first (a few minutes): SMOKE=1 sbatch scripts/e3_train.sh
if [ "${SMOKE:-0}" = "1" ]; then
  EXTRA="--limit 32 --grad-accum 2 --save-steps 4 --out checkpoints/e3_smoke"
else
  EXTRA="--out checkpoints/e3_v2"
fi

cd ~/-Knowledge-Distillation-of-VLM
export PATH="$HOME/-Knowledge-Distillation-of-VLM/.venv/bin:$PATH"
echo "Running on: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

python -m ie643.train --manifest Data/processed/train/manifest.jsonl $EXTRA
