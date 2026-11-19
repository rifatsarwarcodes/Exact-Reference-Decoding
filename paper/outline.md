# Paper outline (working draft)

**Working title:** *Certified Greedy Decoding: How Far Are CPU and Apple LLM Backends from the Exact Model?*
**Target:** TMLR (rolling submissions). Fallback: NeurIPS 2027 Datasets & Benchmarks.
**Author:** Md. Rifat Sarwar. The paper needs an AI-use statement.

## Claims, and the evidence each one needs

| # | Claim | Evidence | Status |
|---|---|---|---|
| C1 | The exact (real-arithmetic) greedy trace of a real LLM can be certified at practical cost: one teacher-forced Arb pass per trace, through the induction certificate | Certified traces for SmolLM2-135M/360M and Qwen3-0.6B; cost table; fraction of steps certified | 4/4 test steps; 93-trace run in progress |
| C2 | Working-precision interval methods cannot do this: fp64 ball radii grow about 10⁶ per layer | Per-layer radius plot at fp64 and at 256/768 bits | Measured on 135M |
| C3 | "fp32/fp64 oracle" references in prior work are not exact. HF `float64` silently computes RMSNorm, RoPE angles and the eager softmax in fp32 | torch-cpu-fp64 against the certified reference; code locations | Code located; sweep queued |
| C4 | Per-step backend error against exact, measured on the decode path, separates into "honest at nominal precision" and "excess" | Teacher-forced err_inf / err_dir per backend vs honest ensembles (random summation order) | fp32 ensemble ≈ torch/MLX fp32 (1.4e-5 vs 1.8e-5) |
| C5 | Excess error has identifiable causes (e.g. llama.cpp's default f16 KV cache) | llamacpp f32 with KV f16 vs KV f32, CPU vs Metal | f32 err ≈ 1.6e-2, ≈ 1000× torch; KV ablation queued |
| C6 | Most cross-backend flips at fp32 are legitimate: the exact margin is inside the honest envelope. At bf16/fp16 the honest envelope is about 0.25 logits, so flips are frequent and still mostly legitimate | Flip classification table | Pending |
| C7 | Quantization effects (exact bf16 model vs exact Q8_0 model) are separable from Q8_0 arithmetic error | Certified trace of the dequantized Q8_0 model | Planned |
| C8 | Results hold across ISAs: Apple M4 (NEON/AMX/Metal), x86-64 and ARM64 Linux | GitHub Actions runs | Workflow written |

## Sections

1. **Introduction:** greedy decoding is not portable, and float-vs-float comparisons can't say who is "right".
2. **Background and related work:**
   - margin-vs-error (Du et al. 2609.26621; tritify 2611.13730; MarginGate; Integer Alibi);
   - nondeterminism and batch invariance;
   - verified numerics (Arb; Rump; Murray 2603.13334);
   - Monte Carlo arithmetic (Verificarlo; 2610.01889).
3. **The exact model and the certificate:** definition; induction; Arb body with a precision schedule; fp64-ball lm_head; assumptions A0–A3; ties.
4. **Honest-precision ensembles and the legitimate/excess split.** Clearly labelled as an estimate.
5. **Setup:** models, prompts (GSM8K/WikiText), backends and versions, hardware.
6. **Results:** C1–C8, with figures:
   - radius growth;
   - err_inf distributions per backend vs the ensemble band;
   - flip margin vs envelope scatter;
   - first-divergence curves.
7. **Released artifact:** certified traces with hidden states, plus a harness for testing a new backend.
8. **Limitations:**
   - prompt set size;
   - small models only;
   - ensemble = one family of honest implementations;
   - A1 (BLAS order) assumption.
9. **AI-use statement.**
