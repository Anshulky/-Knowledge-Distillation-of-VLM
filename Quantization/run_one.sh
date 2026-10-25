#!/bin/bash
set -f  # no glob expansion: quantization-config JSON contains "*"
# One quantization config: serve the merged KD student with vLLM, measure latency (100 docs, 1 at a time)
# and throughput (full 4,240-doc test, 64 concurrent), write result.json. Usage: run_one.sh NAME GPU PORT [vllm args...]
set -u
NAME=$1; GPU=$2; PORT=$3; shift 3; EXTRA="$*"
cd /backup/Anshul/IE643
MODEL=runs/e4/wnone/seed643/merged
D=Data/quant/$NAME
mkdir -p $D
if [ ! -f $D/manifest.jsonl ]; then
  cp Data/processed/test/manifest.jsonl Data/processed/test/pack.json $D/
  ln -sfn ../../processed/test/images $D/images
fi
export VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=$GPU
stamp() { date "+%F %T"; }
echo "[$(stamp)] $NAME: start vLLM ($EXTRA)"
.venv/bin/vllm serve $MODEL --served-model-name student --port $PORT --max-model-len 16384 \
  --gpu-memory-utilization 0.5 $EXTRA > $D/vllm.log 2>&1 &
PID=$!
ok=0
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:$PORT/v1/models >/dev/null && { ok=1; break; }
  kill -0 $PID 2>/dev/null || break
  sleep 5
done
if [ $ok = 0 ]; then
  echo "[$(stamp)] $NAME: vLLM failed to start"
  printf '{"name":"%s","args":"%s","load_ok":false}\n' "$NAME" "$EXTRA" > $D/result.json
  kill $PID 2>/dev/null; sleep 5; kill -9 $PID 2>/dev/null; exit 0
fi
GPU_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $GPU)
CLI=".venv/bin/python -m ie643.teacher --manifest $D/manifest.jsonl --base-url http://127.0.0.1:$PORT/v1 --model student --repetition-penalty 1.15 --max-tokens 4096 --timeout 900"
echo "[$(stamp)] $NAME: latency pass"
t0=$(date +%s.%N); $CLI --pred-key lat_pred --uids "$(cat Data/quant/latency_uids.txt)" --workers 1 > $D/latency.log 2>&1; t1=$(date +%s.%N)
echo "[$(stamp)] $NAME: throughput pass"
t2=$(date +%s.%N); $CLI --pred-key quant_pred --workers 64 --checkpoint-every 256 > $D/throughput.log 2>&1; t3=$(date +%s.%N)
kill -INT $PID 2>/dev/null; sleep 15; kill -9 $PID 2>/dev/null
.venv/bin/python - "$D" "$NAME" "$EXTRA" "$t0" "$t1" "$t2" "$t3" "$GPU_MIB" <<'PY'
import json, re, sys
D, name, extra, t0, t1, t2, t3, gpu_mib = sys.argv[1:]
log = open(f"{D}/vllm.log", errors="ignore").read()
w = re.findall(r"Model loading took ([\d.]+) ?GiB", log)
kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log)
rows = [json.loads(l) for l in open(f"{D}/manifest.jsonl")]
toks = sum(r.get("quant_completion_tokens") or 0 for r in rows if r.get("quant_pred") is not None)
done = sum(1 for r in rows if r.get("quant_pred") is not None)
lat_done = sum(1 for r in rows if r.get("lat_pred") is not None)
res = {"name": name, "args": extra, "load_ok": True,
       "weights_gib": float(w[-1]) if w else None,
       "kv_cache_tokens": int(kv[-1].replace(",", "")) if kv else None,
       "gpu_mib_after_load": int(gpu_mib),
       "latency_s_per_doc": (float(t1) - float(t0)) / max(lat_done, 1), "latency_docs": lat_done,
       "throughput_wall_s": float(t3) - float(t2), "throughput_docs": done,
       "throughput_img_s": done / (float(t3) - float(t2)), "tokens_per_s": toks / (float(t3) - float(t2))}
json.dump(res, open(f"{D}/result.json", "w"), indent=1); print(res)
PY
echo "[$(stamp)] $NAME: done"
