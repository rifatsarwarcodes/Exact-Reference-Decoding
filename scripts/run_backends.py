"""Run backends (free + teacher-forced) against a reference directory.

usage: python scripts/run_backends.py --backends torch-cpu-fp32 mlx-gpu-bf16 ... [--limit N]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
from backends.common import run  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/SmolLM2-135M")
ap.add_argument("--tag", default="smollm2-135m")
ap.add_argument("--ref", default="fp64", help="fp64 | cert | path to a directory of reference npz files")
ap.add_argument("--prompts", default="data/prompts_v1.json")
ap.add_argument("--n_new", type=int, default=64)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--backends", nargs="+", required=True)
ap.add_argument("--driver", default=None, help="path to llamacpp_driver binary")
ap.add_argument("--gguf_dir", default="models/gguf")
a = ap.parse_args()

prompts = json.load(open(a.prompts))
ref_dir = Path(a.ref) if Path(a.ref).is_dir() else Path(f"results/raw/ref/{a.tag}/{a.ref}")
ref_name = ref_dir.name
prompts = [p for p in prompts if (ref_dir / f"{p['id']}.npz").exists()]
if a.limit:
    prompts = prompts[: a.limit]

for name in a.backends:
    fam, dev, dt = name.split("-")[:3]
    if fam == "torch":
        from backends.torch_backend import TorchBackend
        be = TorchBackend(a.model, device=dev, dtype=dt)
    elif fam == "mlx":
        from backends.mlx_backend import MlxBackend
        be = MlxBackend(a.model, device=dev, dtype=dt)
    elif fam == "numpy":
        from backends.numpy_backend import NumpyBackend
        be = NumpyBackend(a.model)
    elif fam == "llamacpp":
        from backends.llamacpp_backend import LlamaCppBackend
        kw = dict(kv=name.split("-kv")[1].split("-")[0]) if "-kv" in name else {}
        be = LlamaCppBackend(a.tag, device=dev, wtype=dt, driver=a.driver, gguf_dir=a.gguf_dir, **kw)
    else:
        raise SystemExit(name)
    out = Path(f"results/raw/runs/{a.tag}/{ref_name}/{be.name}.json")
    rows = run(be, prompts, ref_dir, out, a.n_new, a.model)
    nflip = sum(s["flip"] for r in rows for s in r["steps"])
    nst = sum(len(r["steps"]) for r in rows)
    ndiv = sum(r["first_div"] < a.n_new for r in rows)
    print(f"{be.name:24s} prompts={len(rows)} tf-flips={nflip}/{nst} free-diverged={ndiv}/{len(rows)}", flush=True)
    del be
