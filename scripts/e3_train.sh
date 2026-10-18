#!/bin/bash
# E3: student + CE only on source_gt (LoRA, vision encoder frozen). Submit from the repo root:
#   mkdir -p logs && sbatch scripts/e3_train.sh
#SBATCH --job-name=e3-train
#SBATCH --partition=cn2_madhava
#SBATCH --account=CHANGE_ME_optml_account
#SBATCH --qos=madhava
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out

# Smoke test first (a few minutes): SMOKE=1 sbatch scripts/e3_train.sh
if [ "${SMOKE:-0}" = "1" ]; then
  EXTRA="--limit 32 --max-val 8 --grad-accum 2 --eval-steps 4 --save-steps 4 --out checkpoints/e3_smoke"
else
  EXTRA="--out checkpoints/e3"
fi

cd ~/-Knowledge-Distillation-of-VLM
export PATH="$HOME/-Knowledge-Distillation-of-VLM/.venv/bin:$PATH"
echo "Running on: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

python -m ie643.train --manifest Data/processed/train/manifest.jsonl $EXTRA
