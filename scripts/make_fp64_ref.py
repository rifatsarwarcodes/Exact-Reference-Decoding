"""Candidate exact traces: float64 numpy KV-cached greedy decoding + full next-token logits."""
import sys, json, time, argparse, numpy as np
from pathlib import Path
sys.path.insert(0, '.')
from reference.llama import Fp64Model
ap = argparse.ArgumentParser()
ap.add_argument('--model', default='models/SmolLM2-135M'); ap.add_argument('--tag', default='smollm2-135m')
ap.add_argument('--prompts', default='data/prompts_v1.json'); ap.add_argument('--n_new', type=int, default=64)
ap.add_argument('--out', default=None); ap.add_argument('--limit', type=int, default=0)
a = ap.parse_args()
out = Path(a.out or f'results/raw/ref/{a.tag}/fp64'); out.mkdir(parents=True, exist_ok=True)
M = Fp64Model(a.model, with_abs=False)
prompts = json.load(open(a.prompts))
for P in (prompts[:a.limit] if a.limit else prompts):
    f = out / f"{P['id']}.npz"
    if f.exists(): continue
    t = time.time()
    gen, hid = M.greedy(P['ids'], a.n_new, return_logits=True)
    np.savez(f, gen=np.array(gen), hidden=hid[:a.n_new], prompt=np.array(P['ids']))
    print(P['id'], round(time.time() - t, 1), flush=True)
