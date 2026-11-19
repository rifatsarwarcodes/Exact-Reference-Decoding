#!/bin/bash
# Full local backend sweep on the M4 (sequential, one heavy job at a time).
set -u
cd "$(dirname "$0")/.."
source .venv/bin/activate
export OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4   # fixed thread count (part of the backend config)
while pgrep -f scripts/make_fp64_ref.py >/dev/null; do sleep 15; done
python scripts/export_ref.py
REF=${REF:-fp64}
for B in torch-cpu-fp32 mlx-gpu-fp32 mlx-cpu-fp32 torch-mps-fp32 \
         llamacpp-cpu-f32 llamacpp-cpu-f32-kvf32 llamacpp-metal-f32 llamacpp-metal-f32-kvf32 \
         torch-cpu-bf16 torch-mps-bf16 mlx-gpu-bf16 mlx-cpu-bf16 llamacpp-cpu-bf16 llamacpp-metal-bf16 \
         torch-mps-fp16 mlx-gpu-fp16 torch-cpu-fp16 \
         llamacpp-cpu-q8_0 llamacpp-metal-q8_0 torch-cpu-fp64 \
         llamacpp-cpu-f32-faoff llamacpp-cpu-f32-kvf32-faoff llamacpp-metal-f32-faoff llamacpp-metal-f32-kvf32-faoff \
         llamacpp-cpu-bf16-kvf32-faoff llamacpp-metal-bf16-kvf32-faoff; do
  df -h / | tail -1
  if python -c "import json,sys; sys.exit(0 if len(json.load(open('results/raw/runs/smollm2-135m/$REF/$B.json'))['rows'])>=93 else 1)" 2>/dev/null; then echo "skip $B"; continue; fi
  python scripts/run_backends.py --ref $REF --backends $B 2>&1 | grep -v "Loading weights\|Redirects\|Warning\|warn("
done
python scripts/summarize.py results/raw/runs/smollm2-135m/$REF > results/summary_smollm2-135m_$REF.md
echo SWEEP DONE
