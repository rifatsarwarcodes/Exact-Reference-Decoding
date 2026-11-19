"""Honest-implementation ensembles at a nominal precision (ESTIMATE, not a certificate).

Each member evaluates the same network with IEEE arithmetic at the profile's precision, but with
a random summation order in every reduction (matmul inner dimension split into randomly
permuted chunks accumulated in random order; RMSNorm/softmax sums over permuted inputs).
Every member is therefore a *legitimate* implementation of that precision. The spread of the
ensemble estimates what the precision alone can do; backend errors far outside it are excess.

Profiles
  fp32 : weights (bf16-exact) in fp32, all activations and accumulation fp32.
  bf16 : activations stored in bf16 between ops (as HF/MLX bf16 do); matmul accumulation fp32;
         RMSNorm and softmax computed in fp32; logits rounded to bf16.
  fp16 : as bf16 but with float16 storage.
"""
from __future__ import annotations

import math

import numpy as np

from .llama import Fp64Model

try:
    import ml_dtypes  # noqa: F401
    _BF16 = np.dtype("bfloat16")
except Exception:  # pragma: no cover
    _BF16 = None


def _round_bf16(x: np.ndarray) -> np.ndarray:
    """Round-to-nearest-even float32 -> bfloat16 -> float32 (bit manipulation)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    b = x.view(np.uint32)
    lsb = (b >> 16) & 1
    r = (b + 0x7FFF + lsb) & 0xFFFF0000
    out = r.view(np.float32)
    return np.where(np.isfinite(x), out, x)


class HonestDecoder:
    def __init__(self, fm: Fp64Model, profile: str = "fp32", chunk: int = 128):
        self.fm = fm
        self.profile = profile
        self.chunk = chunk
        self.w32 = {k: v.astype(np.float32) for k, v in fm.w.items()}
        self.wt32 = {k: v.astype(np.float32) for k, v in fm.wt.items()}
        self.rng = np.random.default_rng(0)

    def seed(self, s: int):
        self.rng = np.random.default_rng(s)

    # ---------------------------------------------------------------- storage rounding
    def st(self, x):
        if self.profile == "fp32":
            return x.astype(np.float32)
        if self.profile == "bf16":
            return _round_bf16(x)
        if self.profile == "fp16":
            return x.astype(np.float16).astype(np.float32)
        raise ValueError(self.profile)

    # ---------------------------------------------------------------- random-order reductions
    def mm(self, x, w):
        """x (..., n) @ w (n, k) in fp32 with random inner-dimension chunking/order."""
        n = w.shape[0]
        perm = self.rng.permutation(n)
        chunks = [perm[i:i + self.chunk] for i in range(0, n, self.chunk)]
        self.rng.shuffle(chunks)
        acc = None
        for c in chunks:
            part = (x[..., c] @ w[c]).astype(np.float32)
            acc = part if acc is None else (acc + part).astype(np.float32)
        return acc

    def rsum(self, x, axis=-1):
        n = x.shape[axis]
        perm = self.rng.permutation(n)
        return np.take(x, perm, axis=axis).sum(axis=axis, keepdims=True, dtype=np.float32)

    def rms(self, x, g):
        x = x.astype(np.float32)
        ms = self.rsum(x * x) / np.float32(x.shape[-1])
        return self.st(x / np.sqrt(ms + np.float32(float(self.fm.eps))) * g)

    # ---------------------------------------------------------------- teacher-forced decode
    def tf_logits(self, prompt: list[int], gen: list[int]) -> np.ndarray:
        """Logits (len(gen), V) along the given trace, KV-cached decode path."""
        fm = self.fm
        T = len(prompt) + len(gen)
        half = fm.hd // 2
        inv = (np.float32(fm.theta) ** (-np.arange(0, fm.hd, 2, dtype=np.float32) / np.float32(fm.hd)))
        ang = np.arange(T, dtype=np.float32)[:, None] * inv[None]
        cos = np.concatenate([np.cos(ang)] * 2, -1).astype(np.float32)
        sin = np.concatenate([np.sin(ang)] * 2, -1).astype(np.float32)
        rot = lambda x: np.concatenate([-x[..., half:], x[..., :half]], -1)
        K = [np.zeros((fm.nkv, T, fm.hd), np.float32) for _ in range(fm.L)]
        V = [np.zeros((fm.nkv, T, fm.hd), np.float32) for _ in range(fm.L)]
        out = []
        pos = 0
        cur = list(prompt)
        rep = fm.nh // fm.nkv
        scale = np.float32(1.0 / math.sqrt(fm.hd))
        for step in range(len(gen)):
            n = len(cur)
            ps = np.arange(pos, pos + n)
            x = self.st(self.w32["model.embed_tokens.weight"][cur])
            for l in range(fm.L):
                p = f"model.layers.{l}."
                h = self.rms(x, self.w32[p + "input_layernorm.weight"])
                q = self.st(self.mm(h, self.wt32[p + "self_attn.q_proj.weight"])).reshape(n, fm.nh, fm.hd).transpose(1, 0, 2)
                k = self.st(self.mm(h, self.wt32[p + "self_attn.k_proj.weight"])).reshape(n, fm.nkv, fm.hd).transpose(1, 0, 2)
                v = self.st(self.mm(h, self.wt32[p + "self_attn.v_proj.weight"])).reshape(n, fm.nkv, fm.hd).transpose(1, 0, 2)
                q = self.st(q * cos[ps] + rot(q) * sin[ps])
                k = self.st(k * cos[ps] + rot(k) * sin[ps])
                K[l][:, ps] = k; V[l][:, ps] = v
                kk = np.repeat(K[l][:, :pos + n], rep, 0); vv = np.repeat(V[l][:, :pos + n], rep, 0)
                s = self.mm_batched(q, kk.transpose(0, 2, 1)) * scale
                mask = ps[:, None] < np.arange(pos + n)[None, :]
                s = np.where(mask[None], -np.inf, s)
                e = np.exp(s - s.max(-1, keepdims=True)).astype(np.float32)
                a = self.st(e / self.rsum(e))
                o = self.st(self.mm_batched(a, vv)).transpose(1, 0, 2).reshape(n, fm.nh * fm.hd)
                x = self.st(x + self.st(self.mm(o, self.wt32[p + "self_attn.o_proj.weight"])))
                h = self.rms(x, self.w32[p + "post_attention_layernorm.weight"])
                g = self.st(self.mm(h, self.wt32[p + "mlp.gate_proj.weight"]))
                u = self.st(self.mm(h, self.wt32[p + "mlp.up_proj.weight"]))
                act = self.st(self.st(g / (1 + np.exp(-g))) * u)
                x = self.st(x + self.st(self.mm(act, self.wt32[p + "mlp.down_proj.weight"])))
            h = self.rms(x[-1:], self.w32["model.norm.weight"])
            lg = self.mm(h, self.wt32["lm_head.weight"])[0]
            out.append(self.st(lg).astype(np.float64))
            pos += n
            cur = [gen[step]]
        return np.stack(out)

    def mm_batched(self, a, b):
        """(H, m, n) @ (H, n, k) with random order over n."""
        n = a.shape[-1]
        perm = self.rng.permutation(n)
        chunks = [perm[i:i + self.chunk] for i in range(0, n, self.chunk)]
        self.rng.shuffle(chunks)
        acc = None
        for c in chunks:
            part = (a[..., c] @ b[:, c, :]).astype(np.float32)
            acc = part if acc is None else (acc + part).astype(np.float32)
        return acc
