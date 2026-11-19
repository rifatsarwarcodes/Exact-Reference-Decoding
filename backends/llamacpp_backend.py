from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

DRIVER = Path(__file__).parent / "llamacpp_driver"


class LlamaCppBackend:
    """device: cpu|metal ; wtype: f32|bf16|q8_0 ; kv: f16 (llama.cpp default) | f32."""

    def __init__(self, tag, device="cpu", wtype="f32", kv="f16", flash="auto", threads=4,
                 driver=None, gguf_dir="models/gguf"):
        self.name = f"llamacpp-{device}-{wtype}" + ("" if kv == "f16" else f"-kv{kv}") + \
                    ("" if flash == "auto" else f"-fa{flash}")
        gguf = Path(gguf_dir) / f"{tag}-{wtype}.gguf"
        drv = (Path(driver) if driver else DRIVER).resolve()
        ngl = "99" if device == "metal" else "0"
        off = "1" if device == "metal" else "0"
        self.p = subprocess.Popen([str(drv), str(gguf), ngl, str(threads), kv, flash, off],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        ready = self.p.stderr.readline().decode()
        assert ready.startswith("READY"), ready
        self.V = int(ready.split("V=")[1])
        self.meta = dict(llama_cpp="commit 620d2225c", driver=str(drv), device=device,
                         wtype=wtype, kv=kv, flash=flash, threads=threads)

    def reset(self):
        self.p.stdin.write(b"R\n"); self.p.stdin.flush()
        assert self.p.stdout.read(3) == b"OK\n"

    def _fwd(self, ids):
        self.p.stdin.write(("E %d %s\n" % (len(ids), " ".join(map(str, ids)))).encode())
        self.p.stdin.flush()
        buf = self.p.stdout.read(4 * self.V)
        return np.frombuffer(buf, dtype=np.float32).astype(np.float64)

    def start(self, ids):
        return self._fwd(list(ids))

    def step(self, tok):
        return self._fwd([tok])

    def __del__(self):
        try:
            self.p.stdin.write(b"Q\n"); self.p.stdin.flush(); self.p.wait(timeout=5)
        except Exception:
            pass
