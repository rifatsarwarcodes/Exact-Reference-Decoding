# Novelty check, 2026-11-19

The literature sweep was done by a sub-agent. It read the full text of 2611.13730 and 2609.26621, read abstracts for the rest, and ran about 16 web searches. Copies of the papers are in `pdfs/`. **Before citing, check every entry yourself.** Entries marked "abstract" were checked at abstract level only.

## Verdict

**The combination is novel.** No paper found:
- computes a rigorously enclosed (interval, ball or Arb) real-arithmetic reference for an LLM;
- certifies a greedy trace step by step;
- splits backend flips into legitimate and excess; or
- separates quantization error from arithmetic error using exact references for both the bf16 and the Q8_0 models.

No study of numerical agreement between MLX, MPS and llama.cpp was found. Each ingredient exists on its own, so scoop risk is **moderate**: at least 10 related preprints appeared between May and November 2026.

## Closest work and how we differ

| arXiv | What it does | How we differ |
|---|---|---|
| 2611.13730 tritify (Karim, Nov 2026), full text | Integer-exact ternary-quantized runtime. Bit-identical on ARM/x86. Checked against an fp32 HF reference; 509/512 argmax matches, misses at near-ties. States: "a top-1/top-2 reference gap exceeding 2ε is sufficient to preserve the argmax" | Their model is exact only for its own quantized integer arithmetic, and it is not certified against the real-valued model. Ours is the real-arithmetic model of the published weights, with rigorous enclosures |
| 2609.26621 Du et al., TMLR Sep 2026, full text | BF16 vs FP16 on NVIDIA GPUs. "Divergence is a margin event at the lm_head." FP32 oracle; error bound is a "heuristic estimate" | A certified reference instead of an FP32 one; CPU, ARM and Apple backends; a legitimate/excess split |
| 2608.13756 Integer Alibi (abstract) | Teacher-forced replay; margins predict flips (AUC 0.94) | No exact reference |
| 2603.13334 Murray (abstract) | Sound float-vs-real deviation bound, mechanised in Rocq, for small classifiers | Autoregressive LLMs; certified per-step argmax; backend audit |
| 2506.09501 LayerCast (NeurIPS 2025) | GPU nondeterminism; FP32 fix | Float vs float |
| 2605.30218 MarginGate (abstract) | Low margin predicts batch-induced flips | Float vs float |
| 2503.10251 Budzinskiy et al. (abstract) | Worst-case mixed-precision forward-error bounds for transformers | Theory only; we use a posteriori certificates |
| 2610.01889 Chatelain & de Oliveira Castro (abstract) | Verificarlo stochastic rounding on GPT-2 | Method precedent for our MCA "legitimacy" ensembles |
| 2511.20621 DiFR (abstract) | "Legitimate variation vs actual problems" against a trusted reference | Framing precedent |
| DetLLM (NeurIPS 2026 listing; no arXiv ID seen) | Qwen3-0.6B in exact integer arithmetic; identical hashes across CPUs and GPUs | Same as tritify: exact for its own integer model |

Also relevant:
- **Low overlap:** 2605.19537 (silent hyperparameter: 5 engines including llama.cpp), 2608.04714, 2609.25624, 2611.10611, 2601.06118, 2601.17768, 2604.22411, 2604.19790, 2606.00279.
- **MPS judged against CPU float64:** 2609.22991.
- **Background on float soundness:** 2003.03021, 2205.10159, 2510.23389.
- **Monte Carlo arithmetic on NN inference:** 2212.06361.
- **Arb certification of a PINN:** 2611.07395.
- **Exact rational GPT-2 circuit:** 2605.24033.

## Expected reviewer objections

1. **"Margin vs error is known."** Agreed. Present it as background and cite Du et al., tritify, MarginGate and Integer Alibi. Our contributions are the certified reference side and the legitimate/excess split on the backend side.
2. **"Why certify? fp64 is enough."** This is the biggest threat. We must show:
   - steps where certification is decisive (exact or near-exact ties, for example duplicate embedding rows);
   - how often fp64 alone is inconclusive;
   - that "fp32 oracle" references, as in Du et al., are themselves wrong at some steps.
3. **"Legitimate" is an estimate.** Ensembles from Monte Carlo arithmetic or random summation order are statistical, not certified. `methodology.md` must label them as estimates.
4. **DetLLM also uses Qwen3-0.6B.** Expect a comparison.

## To verify before citing

- Not checked: Higham & Mary (probabilistic rounding error), Johansson (Arb), Rump (verified matrix products, error bounds for floating-point dot products), the Verificarlo paper, and Blanchard, Higham & Higham (softmax/LSE, DOI 10.1093/imanum/draa038, seen in search).
- Not opened: GitHub `same-file-different-answers` (llama.cpp settings on CPU) and `hardware_rounding_error_predictor` (search snippets only).
