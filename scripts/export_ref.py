"""Export slim reference traces to data/ref/<tag>/ (committed; used by CI and released).

Prefers certified traces (results/raw/ref/<tag>/cert); falls back to fp64 candidates and marks
them `certified_prefix = -1` (NOT certified).
"""
import argparse
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="smollm2-135m")
a = ap.parse_args()
raw = Path(f"results/raw/ref/{a.tag}")
out = Path(f"data/ref/{a.tag}"); out.mkdir(parents=True, exist_ok=True)
n_c = n_f = 0
for f in sorted((raw / "fp64").glob("*.npz")):
    c = raw / "cert" / f.name
    if c.exists():
        z = np.load(c)
        np.savez_compressed(out / f.name, prompt=z["prompt"], gen=z["gen"], hidden=z["hidden"],
                            certified_prefix=z["certified_prefix"], cert=z["cert"],
                            margin_lo=z["margin_lo"], margin_hi=z["margin_hi"],
                            logit_rad_max=z["logit_rad_max"])
        n_c += 1
    else:
        z = np.load(f)
        np.savez_compressed(out / f.name, prompt=z["prompt"], gen=z["gen"], hidden=z["hidden"],
                            certified_prefix=np.array(-1))
        n_f += 1
print(f"exported {n_c} certified + {n_f} uncertified (fp64 candidate) traces to {out}")
