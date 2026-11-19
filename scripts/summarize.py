"""Summarise backend runs (one JSON per backend) into a markdown table.

Columns
  tf flips      teacher-forced decode steps where backend argmax != exact argmax
  flip margins  exact top-1/top-2 margins at those steps (max)
  free div      prompts whose free-running greedy trace differs from the exact trace
  1st div       median first-divergence step among diverged prompts
  err_inf       median / p99 over steps of max_v |L_backend - L_exact|
  |err_dir|     p99 of the error in the top-1 minus top-2 logit difference
  envelope      share of flips with exact margin <= p99 |err_dir| of the same backend
                (provisional, empirical; the planned legitimacy test uses MCA ensembles)
"""
import json
import sys
from pathlib import Path

import numpy as np

d = Path(sys.argv[1])
rows = []
for f in sorted(d.glob("*.json")):
    R = json.loads(f.read_text())
    st = [s for r in R["rows"] for s in r["steps"]]
    n_new = R["n_new"]
    flips = [s for s in st if s["flip"]]
    einf = np.array([s["err_inf"] for s in st])
    edir = np.abs(np.array([s["err_dir"] for s in st]))
    p99d = float(np.quantile(edir, 0.99))
    fd = [r["first_div"] for r in R["rows"] if r["first_div"] < n_new]
    env = np.mean([s["ref_margin"] <= p99d for s in flips]) if flips else float("nan")
    rows.append((R["backend"], len(R["rows"]), len(flips), len(st),
                 max((s["ref_margin"] for s in flips), default=float("nan")),
                 len(fd), float(np.median(fd)) if fd else float("nan"),
                 float(np.median(einf)), float(np.quantile(einf, 0.99)), p99d, env,
                 sum(s["b_tie"] for s in st)))
print("| backend | prompts | tf flips | max flip margin | free div | median 1st div | err_inf med | err_inf p99 | p99 abs err_dir | flips in envelope | backend ties |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]}/{r[3]} | {r[4]:.2e} | {r[5]}/{r[1]} | {r[6]:.0f} | {r[7]:.1e} | {r[8]:.1e} | {r[9]:.1e} | {r[10]:.2f} | {r[11]} |")
