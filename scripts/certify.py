"""Certify fp64 candidate greedy traces against the exact real-arithmetic model.

For each prompt: seq = prompt + gen[:n-1]; one teacher-forced Arb forward over seq encloses
the exact logits at all n decision positions. Step t is certified iff the enclosure proves
argmax = gen[t]. If steps 0..t-1 are certified, gen[:t] IS the exact greedy prefix
(induction), so the certified prefix length is the first uncertified step.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from reference import ball as B  # noqa: E402
from reference.arb_forward import ArbModel  # noqa: E402
from reference.llama import Fp64Model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/SmolLM2-135M")
ap.add_argument("--tag", default="smollm2-135m")
ap.add_argument("--prompts", default="data/prompts_v1.json")
ap.add_argument("--ids", nargs="*", default=None)
ap.add_argument("--n_new", type=int, default=64)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--target", type=float, default=40)
ap.add_argument("--per_layer", type=float, default=6)
a = ap.parse_args()

src = Path(f"results/raw/ref/{a.tag}/fp64")
dst = Path(f"results/raw/ref/{a.tag}/cert"); dst.mkdir(parents=True, exist_ok=True)
prompts = json.load(open(a.prompts))
if a.ids:
    prompts = [p for p in prompts if p["id"] in set(a.ids)]
todo = [p for p in prompts if (src / f"{p['id']}.npz").exists() and not (dst / f"{p['id']}.npz").exists()]
print(f"{len(todo)} to certify", flush=True)

fm = Fp64Model(a.model, with_abs=False)
am = ArbModel(fm, prec=(a.target, a.per_layer))
for b0 in range(0, len(todo), a.batch):
    batch = todo[b0:b0 + a.batch]
    refs = [np.load(src / f"{p['id']}.npz") for p in batch]
    seqs = [list(p["ids"]) + [int(x) for x in r["gen"][: a.n_new - 1]] for p, r in zip(batch, refs)]
    t0 = time.time()
    outs = am.forward(seqs, log=[])
    for p, r, (mid, rad) in zip(batch, refs, outs):
        P = len(p["ids"])
        sl = slice(P - 1, P - 1 + a.n_new)
        lb = am.lm_head_ball(mid[sl], rad[sl])
        i1, cert, mlo, mhi, i2 = B.certify_argmax(lb)
        gen = np.array(r["gen"][: a.n_new])
        match = i1 == gen
        ok = cert & match
        cpl = int(np.argmin(ok)) if not ok.all() else a.n_new  # certified prefix length
        fp64_logits = r["hidden"][: a.n_new] @ fm.wt["lm_head.weight"]
        fp64_err = np.abs(fp64_logits - lb.m).max(-1)
        np.savez(dst / f"{p['id']}.npz", gen=gen, hidden=mid[sl] * fm.w["model.norm.weight"], hidden_mid=mid[sl], hidden_rad=rad[sl],
                 logit_rad_max=lb.r.max(-1), cert=ok, margin_lo=mlo, margin_hi=mhi, i2=i2,
                 fp64_err_inf=fp64_err, certified_prefix=cpl, prompt=np.array(p["ids"]),
                 seconds=time.time() - t0, prec=np.array([am.prec_at(l) for l in range(fm.L)]))
        print(f"{p['id']}: certified {ok.sum()}/{a.n_new} prefix={cpl} "
              f"min_margin_lo={mlo.min():.3e} logit_rad<={lb.r.max():.1e} fp64_err<={fp64_err.max():.1e}",
              flush=True)
    print(f"batch done in {time.time()-t0:.0f}s", flush=True)
