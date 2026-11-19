# Methodology: what is certified and what is only estimated

This document states which parts of the pipeline are **certified** (rigorous enclosures; a claim holds if the stated assumptions hold) and which are **estimated** (statistical or empirical). It follows the project rule: never call an estimate "certified".

## 1. The exact model

For a checkpoint, the *exact model* is the real-arithmetic function defined as follows.

- **Weights:** the stored bf16 values. Every bf16 number is exactly representable in binary64 and in Arb.
- **Constants:** the real numbers named in `config.json`:
  - `rms_norm_eps = 1e-5` is taken as exactly 1/100000, which is *not* a float;
  - RoPE angles are exactly `pos · θ^(-2i/d)` with θ = 100000;
  - the attention scale is exactly `1/√d_head`.
- **Operations:** real-valued. RMSNorm, SiLU, softmax and the causal mask follow the HF Llama definition, with the rotate-half RoPE layout.

The exact greedy trace takes the argmax of the exact logits at every step. It is well defined except at exact ties.

**Exact ties are possible.** SmolLM2-135M has 2 groups of identical rows in its tied embedding/lm_head (tokens 5, 7, 9 and 45175). Those token pairs always have exactly equal logits. They are special tokens and have not been top-1 so far, but a certifier must be able to report a tie rather than an order.

**Quantized models get their own exact model.** For a GGUF Q8_0 model, the dequantized weights are exactly fp16 scale × int8, which is exact in binary64. Its exact model is defined the same way. This separates *quantization* (exact bf16 model vs exact Q8_0 model) from *arithmetic error* (Q8_0 runtime vs exact Q8_0 model). *(Q8_0 exact model: planned, not yet run.)*

## 2. Certified: the exact greedy trace and its margins

**Induction trick.** A candidate trace `g` comes from our float64 numpy decoder ([reference/llama.py](../reference/llama.py)); it need not be correct. One teacher-forced forward pass over `prompt + g[:n-1]` encloses the exact logits at all n decision positions. If step t is certified (the enclosure proves argmax = g[t]) for all t < k, then g[:k] **is** the exact greedy prefix. So certifying a whole trace costs one forward pass, not n passes.

**Transformer body: Arb ball arithmetic** ([reference/arb_forward.py](../reference/arb_forward.py), python-flint 0.9.0).
- Every operation runs in Arb with p-bit midpoints and rigorous radii: matmuls via `arb_mat`, then RMSNorm, RoPE (cos/sin from Arb), softmax with the mask, SiLU and the residual adds.
- RMSNorm gains are folded into the next weight matrix. This is exact, because a bf16×bf16 product has a 16-bit significand.
- **Precision schedule.**
  - Radii grow about 10^5.6 per layer, measured on SmolLM2-135M. This is the wrapping effect of box enclosures through about 6 matmuls per layer.
  - Layer l therefore uses p_l = ⌈3.33·(40 + 6·(L−l))⌉ + 64 bits, which is 796 bits at layer 0 and 217 bits at layer 29.
  - **The schedule affects only tightness, never rigour.** If an enclosure is too wide, the step simply fails to certify.
- **Output:** a float64 ball (mid, rad) enclosing the RMS-normalised final hidden state.

**lm_head: float64 ball arithmetic** ([reference/ball.py](../reference/ball.py)).
- One matmul against the 49152×576 matrix (final-norm gain folded in exactly).
- The radius is `γ_576·|h||W| + rad·|W|`. A single matmul has no wrapping, so the logit radius stays about 1e-11.

**Certificate.** Step t is certified iff `lo(L[i1]) > max_{j≠i1} hi(L[j])`. We also report a certified enclosure of the exact margin: `[lo_i1 − max hi_j, hi_i1 − max lo_j]`.

**Assumptions the certificate rests on**
- **A0:** IEEE-754 binary64 round-to-nearest for `+ − × ÷ √` on arm64 numpy.
- **A1:** the BLAS dgemm used by numpy (Apple Accelerate) computes each entry as a sum of n products in *some* order, with or without FMA, and with no Strassen-type algorithm. Then `|fl(AB) − AB| ≤ γ_n |A||B|`. This is used only in the lm_head step and in the (unused) fp64 ball path.
- **A2:** Arb and python-flint are correct. Arb is a mature library with a published correctness model.
- **A3:** converting bf16 to float64 with torch is exact. This is true by construction: bf16 is a truncated binary32.
- No libm function is trusted. The fp64 ball path implements `exp` itself (range reduction plus Taylor remainder), and Arb provides its own.

**Validation.** Over a 16-token test sequence, the Arb midpoints agree with our float64 decoder to 1.4e-13 in logits. All 4 decision steps certified, with a logit radius ≤ 1.3e-11.

## 3. Negative result: fp64 interval bounds do not work

A rigorous fp64 ball pass ([reference/ball.py](../reference/ball.py), `Fp64Model.forward_ball`) loses about 10⁶ per layer and reaches radius > 1 by layer 2. **Worst-case interval methods at working precision cannot certify a 30-layer LLM.** High-precision midpoints are required, and that is why Arb is used.

## 4. Estimated: backend error and "legitimacy"

Backends are run in two modes ([backends/common.py](../backends/common.py)).
- **free:** ordinary KV-cached greedy decoding, compared with the certified exact trace (first divergence step).
- **tf:** a teacher-forced *decode path*. The backend is fed the exact tokens one at a time, exactly as greedy decoding would be if it had not diverged. At each step we record the full-vocabulary error `max|L_B − L_exact|`, the directional error in `L[i1] − L[i2]`, and whether the backend's argmax flips.

All per-backend error statistics are **measurements on a finite prompt set, not certificates.** The reference side is certified; the backend side is observed.

**"Legitimate" flip (estimated).** A flip is *legitimate* if the exact margin at that step is within what an honest implementation at the backend's nominal precision could produce.
- **Provisional (implemented):** the margin is at most the p99 of that backend's own |directional error| over all steps ([scripts/summarize.py](../scripts/summarize.py)).
- **Planned:** a Monte Carlo arithmetic or random-summation-order ensemble at the backend's precision (fp32/bf16/fp16), run on the exact prefix. A flip is legitimate if the ensemble flips with probability ≥ α.

Both versions are **estimates** and will be labelled as such in the paper.

## 5. Reference data released

`data/ref/<model>/<prompt>.npz` will hold, per prompt:
- the prompt and certified trace token ids;
- the float64 final hidden state per step (logits = hidden @ lm_headᵀ);
- the certified prefix length, per-step certification flags, and certified margin bounds.
