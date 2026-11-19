"""Honest-precision ensembles (ESTIMATE): K random-summation-order members per profile,
teacher-forced along the reference trace. Output mirrors backend runs, with per-member lists.

usage: python scripts/run_ensemble.py --profiles fp32 bf16 --members 8 [--ids ...] [--out DIR]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from backends.common import load_ref  # noqa: E402
from reference.ensemble import HonestDecoder  # noqa: E402
from reference.llama import Fp64Model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/SmolLM2-135M")
ap.add_argument("--tag", default="smollm2-135m")
ap.add_argument("--ref", default=None)
ap.add_argument("--profiles", nargs="+", default=["fp32", "bf16", "fp16"])
ap.add_argument("--members", type=int, default=8)
ap.add_argument("--n_new", type=int, default=64)
ap.add_argument("--ids", nargs="*", default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()

ref_dir = Path(a.ref or f"data/ref/{a.tag}")
out = Path(a.out or f"results/raw/ensembles/{a.tag}"); out.mkdir(parents=True, exist_ok=True)
ids = a.ids or sorted(p.stem for p in ref_dir.glob("*.npz"))
fm = Fp64Model(a.model, with_abs=False)
for prof in a.profiles:
    hd = HonestDecoder(fm, prof)
    for pid in ids:
        f = out / f"{prof}__{pid}.json"
        if f.exists():
            continue
        ref = load_ref(ref_dir, pid, a.model)
        gen = [int(t) for t in ref["gen"][: a.n_new]]
        Lr = ref["logits"][: a.n_new]
        i1 = np.array(gen)
        lr_o = Lr.copy(); lr_o[np.arange(len(gen)), i1] = -np.inf
        i2 = lr_o.argmax(-1)
        rows = np.arange(len(gen))
        ref_margin = Lr[rows, i1] - Lr[rows, i2]
        t0 = time.time()
        mem = []
        for s in range(a.members):
            hd.seed(1000 * s + 7)
            L = hd.tf_logits(list(ref["prompt"]), gen)
            b1 = L.argmax(-1)
            mem.append(dict(
                err_inf=np.abs(L - Lr).max(-1).tolist(),
                err_dir=((L[rows, i1] - L[rows, i2]) - ref_margin).tolist(),
                flip=(b1 != i1).tolist(),
                tie=[bool(np.sum(L[t] == L[t, b1[t]]) > 1) for t in rows]))
        f.write_text(json.dumps(dict(profile=prof, id=pid, members=a.members, ref_margin=ref_margin.tolist(),
                                     i1=i1.tolist(), i2=i2.tolist(), runs=mem,
                                     seconds=time.time() - t0)))
        print(f"{prof} {pid} {time.time()-t0:.0f}s flips/member="
              f"{np.mean([sum(m['flip']) for m in mem]):.2f}", flush=True)
