# Exact-reference LLM decoding: which greedy flips are legitimate?

**Status (2026-11-19):**
- **All 93 SmolLM2-135M greedy traces are certified:** 5,952/5,952 steps, with margins enclosed to within 3e-11 ([data/ref/](data/ref/)).
- **Honest-precision ensembles** are done for fp32, fp16, bf16 and bf16mm. Honest fp32 never flips on this set.
- **llama.cpp's default f32 has about 700× excess error.** The causes are the CPU flash-attention path and the f16 KV cache.
- **The Apple backend sweep and the full x86-64/ARM64 CI run are in progress.**
- Details are in [docs/research-log.md](docs/research-log.md) and [docs/methodology.md](docs/methodology.md).
**Idea ID:** NQ1 in [../reports/Low resource CS and robotics prospects.md](../reports/Low%20resource%20CS%20and%20robotics%20prospects.md), pick #5. P 4, F 4.5, 6–8 weeks.
**Background notes:** [../research_notes/Low resource CS and robotics prospects/numerics_quantum_verification.md](../research_notes/Low%20resource%20CS%20and%20robotics%20prospects/numerics_quantum_verification.md), card NQ1.
**Related:**
- The Thinking Tax pipeline ([../bangla-llm/](../bangla-llm/README.md)).
- EP2 ([../tolerance-hacking/](../tolerance-hacking/README.md)), for its interval-arithmetic tooling.
- EP12 (precision hacking in math RLVR).
- A good **TMLR** candidate. A TMLR acceptance counts toward ICML/ICLR reviewer eligibility.

## Question

The same model with the same prompt and greedy decoding gives different outputs on different backends: PyTorch CPU vs MPS, MLX, llama.cpp, x86 vs ARM. The existing work compares float runs **with each other**.

> **Against the exact (infinitely precise) model, which flips are numerically unavoidable, because the top-1 vs top-2 logit margin is smaller than any reasonable error bound? Which are excess error from the backend? How does this differ across CPU, ARM and Apple backends?**

## Why this is open (as of 2026-11-18)

| Paper | What it does | Gap left |
|---|---|---|
| [2506.09501](https://arxiv.org/abs/2506.09501) | Reproducibility of LLM inference under numerics | GPU; float vs float |
| [2609.26621](https://arxiv.org/abs/2609.26621) | BF16 vs FP16 diverge on 49–100% of prompts | No exact reference |
| [2605.19537](https://arxiv.org/abs/2605.19537) | Inference backends as a "silent hyperparameter" | GPU-centric |
| [2609.25624](https://arxiv.org/abs/2609.25624), [2611.10611](https://arxiv.org/abs/2611.10611), [2605.30218](https://arxiv.org/abs/2605.30218), [2601.06118](https://arxiv.org/abs/2601.06118) | More nondeterminism and batch-invariance work | GPU; float vs float |
| [2611.13730](https://arxiv.org/abs/2611.13730) "tritify" | **Closest:** exact integer runtime | Does not measure how far float backends are from exact |

There are at least 8 GPU systems papers on batch-invariant kernels; **don't build kernels.** The open slice is (a) the **certified exact reference** and (b) **CPU, ARM and Apple backends**. Scoop risk is **medium**: the area is very active.

## Method (draft)

1. **Models:** SmolLM2-135M/360M and Qwen3-0.6B (each ≤1.5 GB).
2. **Reference:** a float64 (or double-double) forward pass. At each greedy step, certify the top-1 vs top-2 margin with a rigorous error bound. Use interval arithmetic (Arb) on the final layers, plus an error-propagation bound for the rest; document how rigorous each part is.
3. **Backends:**
   - PyTorch-CPU, PyTorch-MPS;
   - MLX-Metal, MLX-CPU;
   - llama.cpp (Q8_0/BF16, NEON);
   - ONNX Runtime;
   - x86 via **free GitHub Actions runners**. Rosetta's AVX support is unverified, so don't rely on it.
4. **Metrics:** flip rate per backend, the share of flips at steps with certified-small margins ("legitimate") vs larger ("excess"), and benchmark-score deltas on a small eval set.
5. **Deliverable:** a "certified-greedy" reference-trace dataset that others can test against.

## Resources

- M4 CPU/MPS; models ≤1.5 GB. Run `df -h` before downloading.
- GitHub Actions minutes for x86. This needs a public repo, so ask the user before creating one.
- $0 API.

## Deliverables and venues

- **Outputs:** reference traces and certification code.
- **Venues:**
  - **TMLR (preferred)**;
  - NeurIPS 2027 Datasets & Benchmarks (around mid-May 2027, estimated);
  - EMNLP/ACL evaluation tracks;
  - MLSys workshops.
- **Author:** Md. Rifat Sarwar. No affiliation has been given.

## First steps

1. ~~Novelty re-check~~ Done 2026-11-19; see [docs/related/novelty-check-2026-11-19.md](docs/related/novelty-check-2026-11-19.md).
2. ~~Feasibility~~ Done. fp64 intervals explode at about 10⁶ per layer, so the transformer body runs in Arb (796 → 217 bits) and the lm_head in fp64 ball arithmetic.
3. Pilot: SmolLM2-135M, 93 prompts × 64 tokens. *In progress.*
4. ~~Write `docs/methodology.md`~~ Done.

## Reproduce

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install numpy torch transformers safetensors mlx mlx-lm python-flint huggingface_hub sentencepiece pyarrow pandas pyyaml
python scripts/make_fp64_ref.py                 # candidate traces (float64 numpy)
python scripts/certify.py --batch 2             # Arb certificate (slow: ~12 min/trace)
python scripts/run_backends.py --ref cert --backends torch-cpu-fp32 mlx-gpu-bf16 llamacpp-cpu-f32
python scripts/summarize.py results/raw/runs/smollm2-135m/cert
```

## Layout

```
exact-reference-decoding/
├── README.md  CLAUDE.md
├── docs/research-log.md   dated entries, newest first
├── docs/methodology.md
├── reference/             float64/interval reference forward pass
├── backends/              per-backend runners
├── .github/workflows/     x86-64 + ARM64 Linux backend runs (manual trigger)
├── results/
└── paper/
```
