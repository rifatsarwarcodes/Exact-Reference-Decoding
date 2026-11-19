# Backend classification, SmolLM2-135M (93 certified traces × 64 steps)

These are **estimates**: legitimacy is judged against honest-precision ensembles (see docs/methodology.md §4). The reference side is certified.

- "legit (ensemble)": at least one of 8 honest members at the backend's nominal precision flips at that step.
- "legit (envelope)": the exact margin is within the p99 |directional error| of that ensemble.
- "excess-error ratio": the backend's median err_inf divided by the honest median.
- q8_0 is a different exact model; it is shown against fp32 for reference only.

## x86-64 Linux, AMD EPYC 9V74 (GitHub Actions, run ci-full-v1)

| backend | profile | tf flips | legit (ensemble) | legit (envelope) | excess | err_inf median | excess-error ratio |
|---|---|---|---|---|---|---|---|
| llamacpp-cpu-bf16-kvf32-faoff | bf16mm | 14 | 14 | 0 | 0 | 1.0e-01 | 1.0x |
| llamacpp-cpu-bf16 | bf16mm | 28 | 24 | 4 | 0 | 1.0e-01 | 1.0x |
| llamacpp-cpu-f32-faoff | fp32 | 1 | 0 | 0 | 1 | 7.1e-03 | 586.0x |
| llamacpp-cpu-f32-kvf32-faoff | fp32 | 0 | 0 | 0 | 0 | 1.2e-05 | 1.0x |
| llamacpp-cpu-f32-kvf32 | fp32 | 1 | 0 | 0 | 1 | 1.3e-02 | 1076.1x |
| llamacpp-cpu-f32 | fp32 | 1 | 0 | 0 | 1 | 1.3e-02 | 1076.1x |
| llamacpp-cpu-q8_0 | q8_0 (vs fp32; different exact model) | 150 | 0 | 0 | 150 | 7.0e-01 | 58364.2x |
| torch-cpu-bf16 | bf16 | 103 | 92 | 11 | 0 | 2.9e-01 | 1.1x |
| torch-cpu-fp32 | fp32 | 0 | 0 | 0 | 0 | 2.5e-05 | 2.1x |

## ARM64 Linux, Neoverse-N2 (GitHub Actions, run ci-full-v1)

| backend | profile | tf flips | legit (ensemble) | legit (envelope) | excess | err_inf median | excess-error ratio |
|---|---|---|---|---|---|---|---|
| llamacpp-cpu-bf16-kvf32-faoff | bf16mm | 26 | 20 | 6 | 0 | 1.0e-01 | 1.0x |
| llamacpp-cpu-bf16 | bf16mm | 20 | 17 | 3 | 0 | 1.0e-01 | 1.0x |
| llamacpp-cpu-f32-faoff | fp32 | 1 | 0 | 0 | 1 | 7.1e-03 | 588.7x |
| llamacpp-cpu-f32-kvf32-faoff | fp32 | 0 | 0 | 0 | 0 | 1.3e-05 | 1.0x |
| llamacpp-cpu-f32-kvf32 | fp32 | 3 | 0 | 0 | 3 | 1.3e-02 | 1066.3x |
| llamacpp-cpu-f32 | fp32 | 3 | 0 | 0 | 3 | 1.3e-02 | 1066.3x |
| llamacpp-cpu-q8_0 | q8_0 (vs fp32; different exact model) | 168 | 0 | 0 | 168 | 7.1e-01 | 58469.5x |
| torch-cpu-bf16 | bf16 | 98 | 87 | 10 | 1 | 2.9e-01 | 1.1x |
| torch-cpu-fp32 | fp32 | 0 | 0 | 0 | 0 | 2.8e-05 | 2.3x |

## Apple M4 (local; sweep partial)

| backend | profile | tf flips | legit (ensemble) | legit (envelope) | excess | err_inf median | excess-error ratio |
|---|---|---|---|---|---|---|---|
| llamacpp-cpu-f32-kvf32 | fp32 | 2 | 0 | 0 | 2 | 1.6e-02 | 1288.3x |
| llamacpp-cpu-f32 | fp32 | 2 | 0 | 0 | 2 | 1.6e-02 | 1288.3x |
| llamacpp-metal-f32-kvf32 | fp32 | 2 | 0 | 0 | 2 | 8.3e-03 | 688.6x |
| llamacpp-metal-f32 | fp32 | 2 | 0 | 0 | 2 | 8.3e-03 | 688.6x |
| mlx-cpu-bf16 | bf16 | 104 | 89 | 14 | 1 | 2.9e-01 | 1.1x |
| mlx-cpu-fp32 | fp32 | 0 | 0 | 0 | 0 | 1.8e-05 | 1.5x |
| mlx-gpu-bf16 | bf16 | 103 | 87 | 16 | 0 | 2.9e-01 | 1.1x |
| mlx-gpu-fp32 | fp32 | 0 | 0 | 0 | 0 | 1.5e-05 | 1.3x |
| torch-cpu-bf16 | bf16 | 100 | 88 | 11 | 1 | 2.8e-01 | 1.1x |
| torch-cpu-fp32 | fp32 | 0 | 0 | 0 | 0 | 1.8e-05 | 1.5x |
| torch-mps-bf16 | bf16 | 95 | 85 | 10 | 0 | 2.9e-01 | 1.1x |
