"""Generate Kaggle CPU kernels that certify shards of the reference traces.

Each kernel: clones this repo at a pinned commit, downloads the model at a pinned revision,
and runs `scripts/certify.py` in N_WORKERS parallel processes on its shard (CPU only, no GPU).
Outputs land in /kaggle/working/cert/*.npz; fetch with `kaggle kernels output`.

usage: python kaggle/make_certify_kernels.py --commit <sha> --kernels 4 [--skip id ...]
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--commit", required=True)
ap.add_argument("--kernels", type=int, default=4)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--tag", default="smollm2-135m")
ap.add_argument("--model_repo", default="HuggingFaceTB/SmolLM2-135M")
ap.add_argument("--model_rev", default="93efa2f097d58c2a74874c7e644dbc9b0cee75a2")
ap.add_argument("--skip", nargs="*", default=[])
ap.add_argument("--user", default="rifatsarwarcodes")
ap.add_argument("--out", default="kaggle/kernels")
a = ap.parse_args()

ids = sorted(p.stem for p in Path(f"data/ref/{a.tag}").glob("*.npz"))
ids = [i for i in ids if i not in set(a.skip)]
shards = [ids[k::a.kernels] for k in range(a.kernels)]

RUN = r'''
import os, subprocess, sys, shutil, time
COMMIT = "{commit}"
IDS = {ids}
W = {workers}
def sh(c):
    print("+", c, flush=True); subprocess.run(c, shell=True, check=True)
sh("pip install -q python-flint==0.9.0 safetensors==0.8.0 huggingface_hub")
sh("git clone -q https://github.com/{user}/Exact-Reference-Decoding.git /kaggle/repo")
sh(f"git -C /kaggle/repo checkout -q {{COMMIT}}")
os.chdir("/kaggle/repo")
from huggingface_hub import snapshot_download
snapshot_download("{model_repo}", revision="{model_rev}", local_dir="models/m",
                  allow_patterns=["*.json", "*.safetensors", "*.txt"])
sh("nproc; lscpu | head -20; python -c 'import numpy, flint; print(numpy.__version__, flint.__version__); numpy.show_config()'")
groups = [IDS[w::W] for w in range(W)]
procs = []
os.makedirs("/kaggle/working/logs", exist_ok=True)
for w, g in enumerate(groups):
    if not g: continue
    log = open(f"/kaggle/working/logs/worker{{w}}.log", "w")
    procs.append(subprocess.Popen([sys.executable, "scripts/certify.py", "--model", "models/m", "--tag", "{tag}",
        "--src", "data/ref/{tag}", "--dst", "/kaggle/working/cert", "--batch", "1", "--ids", *g],
        stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")))
t0 = time.time()
while any(p.poll() is None for p in procs):
    time.sleep(60)
    n = len(os.listdir("/kaggle/working/cert")) if os.path.isdir("/kaggle/working/cert") else 0
    print(f"{{time.time()-t0:7.0f}}s  certified files: {{n}}/{{len(IDS)}}", flush=True)
print("exit codes", [p.returncode for p in procs])
'''

for k, shard in enumerate(shards):
    d = Path(a.out) / f"certify-{k}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(RUN.format(commit=a.commit, ids=json.dumps(shard), workers=a.workers,
                                         user=a.user, model_repo=a.model_repo, model_rev=a.model_rev,
                                         tag=a.tag))
    meta = dict(id=f"{a.user}/nq1-certify-{a.tag}-{k}", title=f"nq1-certify-{a.tag}-{k}",
                code_file="run.py", language="python", kernel_type="script", is_private=True,
                enable_gpu=False, enable_tpu=False, enable_internet=True,
                dataset_sources=[], competition_sources=[], kernel_sources=[])
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    print(d, len(shard), "traces")
