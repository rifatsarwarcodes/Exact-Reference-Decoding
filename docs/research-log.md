# Research log

Newest entries first. Each entry records what was done, what was found, and what was decided and why.

---

## 2026-11-19 (b): Reference pipeline built; first certified steps; backends run end to end

### Novelty re-check
Full text of 2611.13730 and 2609.26621, abstracts of about 25 more, and about 16 searches. Write-up: [related/novelty-check-2026-11-19.md](related/novelty-check-2026-11-19.md).

- **Verdict:** novel as a combination. No one has a rigorous real-arithmetic enclosure of an LLM, a certified greedy trace, or a legitimate/excess split. No study of numerical agreement between MLX, MPS and llama.cpp turned up either.
- **Closest work:**
  - tritify 2611.13730: exact only for its own integer-quantized model; checked against fp32.
  - Du et al. 2609.26621 (TMLR): FP32 "oracle", heuristic bound, GPU only.
- **Scoop risk:** moderate.
- **Main reviewer threat:** "why certify; fp64 is enough?" Finding 3 below begins to answer it.

### Findings
1. **fp64 interval bounds fail.** A rigorous float64 ball pass grows about 10⁶ per layer (wrapping through about 6 matmuls per layer) and exceeds 1 by layer 2. Worst-case intervals at working precision cannot certify a 30-layer model.
2. **Arb at about 768 bits certifies.** Test: SmolLM2-135M, T = 16, uniform 768 bits.
   - Final hidden relative radius 6.5e-53; logit radius ≤ 1.3e-11 (from the fp64 ball lm_head).
   - All 4 steps certified, with margins 0.131, 2.104, 1.338 and 0.151.
   - Run time was 403 s. A per-layer precision schedule (796 → 217 bits) is now the default.
3. **"float64" is not automatically exact.**
   - Our numpy float64 decode matches the exact logits to 1.4e-13.
   - HF transformers with `dtype=float64` differs from our float64 by 9.5e-6. The suspected cause is RoPE `inv_freq` computed in float32 inside HF. **Not yet verified.** This bears directly on Du et al.-style "FP32/FP64 oracle" references.
4. **SmolLM2-135M has exact ties.** There are 2 groups of identical embedding rows (tokens 5/7/9 and 45175), so the exact model has genuine ties for those pairs.
5. **First backend numbers** (3 prompts × 64 steps, against the fp64 candidate reference):
   - torch-cpu-fp32 and mlx-gpu-fp32: err_inf median about 1.8e-5, 0 flips.
   - **llamacpp-cpu-f32: err_inf median 1.6e-2, about 1000× larger.** The suspected cause is llama.cpp's default fp16 KV cache. A `-kvf32` variant has been added to test this.

### Built
- `reference/ball.py`: rigorous fp64 ball arithmetic, including an exp that does not use libm.
- `reference/llama.py`: float64 KV-cached decoder and fp64 ball forward pass.
- `reference/arb_forward.py`: Arb forward pass with batching and the precision schedule.
- `scripts/certify.py`: the induction certificate.
- Backends:
  - torch CPU/MPS in fp32/bf16/fp16/fp64;
  - MLX GPU/CPU;
  - llama.cpp through a C++ driver (`backends/llamacpp_driver.cpp`) that streams raw logits over a pipe.
- `.github/workflows/backends.yml`: x86-64 and ARM64 Linux runs. **The user approved the public repo on 2026-11-19.**

### Incidents
- **The disk filled to 0 bytes.** Free space went from 14 GB to 0, driven by swap (7.2/8 GB) while other sessions ran concurrent jobs. The first 25-minute certification batch was lost when its save hit ENOSPC.
- **Fixes:** references now store the 576-dim final hidden state instead of 49k logits, which is 40× smaller; batches are smaller; one heavy job runs at a time.

### Pinned versions
- **Model:** HuggingFaceTB/SmolLM2-135M @ `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.
- **Data:**
  - openai/gsm8k @ `740312add88f781978c0658806c59bc2815b9866` (test split, first 100 questions, keeping those ≤56 tokens: 43);
  - Salesforce/wikitext wikitext-2-raw-v1 @ `b08601e04326c79dfdd32d625aee71d232d685c3` (test split, first 50 paragraphs longer than 200 characters, truncated to 40 tokens).
  - The resulting prompt set is `data/prompts_v1.json` (93 prompts).
- **llama.cpp:** brew 0.7.0, build 11669, commit `620d2225cc436ab0b001e3200e590e3b286ef9ab`. The converter was sparse-cloned at the same commit into `third_party/llama.cpp`. It was read before running: it only applies an exact Q/K permutation, and its network calls are limited to `--remote`.
- **Python 3.11 venv:**
  - torch 2.15.0, transformers 5.23.0, mlx 0.32.4, mlx-lm 0.32.2;
  - numpy 2.4.6, python-flint 0.9.0, safetensors 0.8.0.

### Repository
- **Remote:** `git@github.com:rifatsarwarcodes/Exact-Reference-Decoding.git`, pushed through the SSH alias `github.com-rifatsarwarcodes`.
  - The repo appears **private**: the unauthenticated API returns 404.
  - Commits use the form `action(scope): message`, with no AI attribution, as the user instructed.
  - Author: `Md. Rifat Sarwar <272023465+rifatsarwarcodes@users.noreply.github.com>`.
- **CI:** pushing a tag `ci-*` starts the x86-64/ARM64 workflow; `ci-n<N>-*` limits it to N prompts. Each job pushes its results to `ci-results/<tag>-<run>/<runner>`.
- **Smoke run:** tag `ci-n3-smoke1`.
- **Open question:** ARM64 runners are free only for public repos, so the ARM job may not run while the repo is private.
- **Data:** `data/ref/smollm2-135m/` (93 traces, 25 MB) currently holds **fp64 candidates, not certified** (`certified_prefix = -1`). They will be replaced once certification finishes.

### Next
- [ ] Certify all 93 traces: background, about 12 min per trace, single process.
- [ ] Full backend sweep: torch CPU/MPS × fp32/bf16/fp16, MLX GPU/CPU × fp32/bf16/fp16, llama.cpp CPU/Metal × f32/bf16/q8_0 × KV f16/f32.
- [ ] Verify the HF float64 RoPE hypothesis.
- [ ] Push to GitHub (rifatsarwarcodes), then run the x86 and ARM workflow.
- [ ] Certify the exact Q8_0 model, to separate quantization from arithmetic error.
- [ ] Build the MCA/random-order ensemble for the legitimacy envelope.
- [ ] Add qk-norm to the Arb pass (Qwen3-0.6B), then run SmolLM2-360M.

---

## 2026-11-19: Project created

- Chosen from the low-resource deep dive: `../reports/Low resource CS and robotics prospects.md`. The background notes are in `../research_notes/Low resource CS and robotics prospects/`.
- Nothing installed or run yet. Competitor checks so far are title- and abstract-level only.
