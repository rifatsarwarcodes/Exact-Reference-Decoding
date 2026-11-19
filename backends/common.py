"""Shared harness: run a backend in two modes against a reference trace.

A backend implements:
    name: str
    start(ids: list[int]) -> np.ndarray[V]     # prefill; logits for next token
    step(tok: int) -> np.ndarray[V]            # append one token; logits for next token
    reset()

Modes (per prompt):
  free  : ordinary KV-cached greedy decoding, n_new tokens  -> first divergence vs exact
  tf    : teacher-forced *decode path* along the exact trace (feed exact tokens one by one,
          exactly like greedy decoding would if it had not diverged); per-step logit error
          vs the reference logits.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np


_LM = {}


def lm_head(model_dir) -> np.ndarray:
    """float64 lm_head^T (d, V); bf16 weights are exact in float64."""
    if model_dir not in _LM:
        import torch
        from safetensors.torch import load_file
        sd = load_file(f"{model_dir}/model.safetensors")
        w = sd.get("lm_head.weight", sd["model.embed_tokens.weight"])
        _LM[model_dir] = np.ascontiguousarray(w.to(torch.float64).numpy().T)
    return _LM[model_dir]


def load_ref(ref_dir: Path, pid: str, model_dir: str):
    """Reference logits are recomputed from the stored final hidden state (float64)."""
    z = np.load(ref_dir / f"{pid}.npz")
    d = {k: z[k] for k in z.files}
    d["logits"] = d["hidden"] @ lm_head(model_dir)
    return d


def run(backend, prompts, ref_dir: Path, out_path: Path, n_new: int, model_dir: str):
    rows = []
    t0 = time.time()
    for P in prompts:
        pid, ids = P["id"], P["ids"]
        ref = load_ref(ref_dir, pid, model_dir)
        gen = [int(t) for t in ref["gen"][:n_new]]
        Lr = ref["logits"][:n_new]                    # (n_new, V) float64 reference logits
        # ---- free-running greedy
        backend.reset()
        lg = backend.start(ids)
        free = []
        for t in range(n_new):
            nxt = int(np.argmax(lg))
            free.append(nxt)
            if t + 1 < n_new:
                lg = backend.step(nxt)
        fd = next((i for i, (a, b) in enumerate(zip(free, gen)) if a != b), n_new)
        # ---- teacher-forced decode along the exact trace
        backend.reset()
        lg = backend.start(ids)
        steps = []
        for t in range(n_new):
            Lb = np.asarray(lg, dtype=np.float64)
            lr = Lr[t]
            i1 = int(gen[t])
            lr_o = lr.copy(); lr_o[i1] = -np.inf
            i2 = int(np.argmax(lr_o))
            b1 = int(np.argmax(Lb))
            lb_o = Lb.copy(); lb_o[b1] = -np.inf
            diff = Lb - lr
            steps.append(dict(
                t=t, i1=i1, i2=i2, b1=b1,
                ref_margin=float(lr[i1] - lr[i2]),
                b_margin=float(Lb[b1] - lb_o.max()),
                b_tie=bool(Lb[b1] == lb_o.max()),
                err_inf=float(np.abs(diff).max()),
                err_rms=float(np.sqrt((diff ** 2).mean())),
                err_dir=float((Lb[i1] - Lb[i2]) - (lr[i1] - lr[i2])),
                flip=bool(b1 != i1),
            ))
            if t + 1 < n_new:
                lg = backend.step(i1)
        rows.append(dict(id=pid, free=free, first_div=fd, steps=steps))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dict(backend=backend.name, meta=getattr(backend, "meta", {}),
                                        n_new=n_new, seconds=time.time() - t0, rows=rows)))
    return rows
