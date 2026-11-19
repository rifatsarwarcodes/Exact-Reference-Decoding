"""Legitimate vs excess flips (ESTIMATE; see docs/methodology.md section 4).

For backend B with nominal precision profile P (fp32 / bf16 / fp16):
  * A teacher-forced flip of B at (prompt, t) is
      legit-ens  if >= 1 honest member of profile P also flips at (prompt, t)
      legit-env  else if the exact margin <= q-quantile of |err_dir| pooled over the P ensemble
      excess     otherwise
  * excess-error ratio = median err_inf(B) / median err_inf(P ensemble)
Quantized backends (q8_0) define a different exact model; they are reported against the fp32
envelope for reference only and flagged.

usage: python scripts/classify.py RUNS_DIR ENS_DIR [--q 0.99]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("runs")
ap.add_argument("ens")
ap.add_argument("--q", type=float, default=0.99)
a = ap.parse_args()


def profile_of(name: str) -> str:
    if name.startswith("llamacpp") and "-bf16" in name:
        return "bf16mm"            # ggml: bf16 weights, matmul inputs rounded to bf16, f32 elsewhere
    for p in ("fp32", "f32", "bf16", "fp16", "fp64", "q8_0"):
        if f"-{p}" in name:
            return {"f32": "fp32"}.get(p, p)
    return "?"


ens = defaultdict(dict)                     # profile -> id -> record
for f in Path(a.ens).rglob("*.json"):
    R = json.loads(f.read_text())
    ens[R["profile"]][R["id"]] = R
env, med = {}, {}
for p, recs in ens.items():
    d = np.abs(np.concatenate([np.array(m["err_dir"]) for r in recs.values() for m in r["runs"]]))
    e = np.concatenate([np.array(m["err_inf"]) for r in recs.values() for m in r["runs"]])
    env[p] = float(np.quantile(d, a.q))
    med[p] = float(np.median(e))
    nflip = sum(sum(m["flip"]) for r in recs.values() for m in r["runs"])
    nst = sum(len(m["flip"]) for r in recs.values() for m in r["runs"])
    print(f"ensemble {p}: prompts={len(recs)} member-steps={nst} member-flips={nflip} "
          f"median err_inf={med[p]:.2e} q{a.q} |err_dir|={env[p]:.2e}")

print()
print("| backend | profile | tf flips | legit (ensemble) | legit (envelope) | excess | err_inf median | excess-error ratio |")
print("|---|---|---|---|---|---|---|---|")
for f in sorted(Path(a.runs).glob("*.json")):
    R = json.loads(f.read_text())
    b = R["backend"]; p = profile_of(b)
    pe = "fp32" if p in ("q8_0", "fp64") else p
    if pe not in ens:
        continue
    le = lv = ex = n = 0
    errs = []
    for row in R["rows"]:
        er = ens[pe].get(row["id"])
        for s in row["steps"]:
            errs.append(s["err_inf"])
            if not s["flip"]:
                continue
            n += 1
            if er is not None and any(m["flip"][s["t"]] for m in er["runs"]):
                le += 1
            elif s["ref_margin"] <= env[pe]:
                lv += 1
            else:
                ex += 1
    em = float(np.median(errs))
    flag = " (vs fp32; different exact model)" if p == "q8_0" else ""
    print(f"| {b} | {p}{flag} | {n} | {le} | {lv} | {ex} | {em:.1e} | {em / med[pe]:.1f}x |")
