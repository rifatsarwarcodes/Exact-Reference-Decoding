# Honest-precision ensembles, SmolLM2-135M (run `ens-m8-v1`, GitHub Actions ubuntu-24.04 x86-64)

**This is an estimate, not a certificate.** There are 8 random-summation-order members per profile, run teacher-forced along the reference traces: 93 prompts × 64 steps.

The reference traces are fp64 candidates, which agree with exact to about 1e-13. Certification is in progress.

| Profile | Member-steps | Member flips | Steps where ≥1 member flips | Median err_inf | p99 abs err_dir (envelope) |
|---|---|---|---|---|---|
| fp32 | 47,616 | 0 | 0 / 5,952 | 1.21e-05 | 8.65e-06 |
| fp16 | 47,616 | 74 | 27 / 5,952 | 3.33e-02 | 2.73e-02 |
| bf16 | 47,616 | 741 | 195 / 5,952 | 2.65e-01 | 2.13e-01 |

**Exact top-1/top-2 margins over the 5,952 steps:**
- minimum 5.54e-05;
- quantiles: 0.1% at 3.3e-03, 1% at 1.9e-02, 5% at 9.3e-02, median 1.54;
- share of steps with margin below a threshold: < 1e-4: 0.02%; < 1e-3: 0.03%; < 1e-2: 0.47%; < 1e-1: 5.3%.

**Reading:**
- At honest fp32, no greedy flip is expected on this prompt set: the smallest exact margin is about 6.4× the fp32 envelope. Any fp32 backend flip here is excess.
- At bf16, about 3% of steps are within the reach of an honest implementation.
