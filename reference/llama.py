"""Exact-semantics Llama-family forward pass (SmolLM2, also Qwen3 with qk-norm).

Two implementations of the *same real-arithmetic function*:
  * Fp64Model.greedy(): plain float64 numpy, KV-cached; produces a candidate trace.
  * Fp64Model.forward_ball(): teacher-forced over a whole sequence in rigorous ball
    arithmetic (reference/ball.py); encloses the exact logits at every position.

The "exact model" is: weights = the stored bf16 values (exact in float64); all
constants are the real numbers named in config (eps = 1e-5 exactly, RoPE angles
pos * theta^(-2i/d) exactly); all operations in real arithmetic.
"""
from __future__ import annotations

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from . import ball as B


def _load_bf16_as_f64(path: Path) -> dict[str, np.ndarray]:
    # safetensors.numpy cannot hold bf16; read via torch then convert exactly.
    import torch
    from safetensors.torch import load_file as tload
    sd = tload(str(path))
    return {k: v.to(torch.float64).numpy() for k, v in sd.items()}


def rope_tables(T: int, head_dim: int, theta: float, prec: int = 200):
    """cos/sin balls of shape (T, head_dim) for HF rotate_half convention, from Arb."""
    from flint import arb, ctx
    old = ctx.prec
    ctx.prec = prec
    half = head_dim // 2
    th = arb(Fraction(theta).numerator) / arb(Fraction(theta).denominator)
    inv = [th ** (arb(-2 * i) / head_dim) for i in range(half)]
    cm = np.empty((T, half)); cr = np.empty((T, half))
    sm = np.empty((T, half)); sr = np.empty((T, half))
    for p in range(T):
        for i in range(half):
            ang = inv[i] * p
            for tgt_m, tgt_r, val in ((cm, cr, ang.cos()), (sm, sr, ang.sin())):
                f = float(val.mid())
                tgt_m[p, i] = f
                tgt_r[p, i] = float(abs(arb(f) - val).upper()) * 2 + B.TINY
    ctx.prec = old
    cat = lambda a: np.concatenate([a, a], -1)
    return B.Ball(cat(cm), cat(cr)), B.Ball(cat(sm), cat(sr))


class Fp64Model:
    def __init__(self, model_dir: str | Path, weights: dict[str, np.ndarray] | None = None, with_abs: bool = True):
        model_dir = Path(model_dir)
        self.cfg = json.loads((model_dir / "config.json").read_text())
        c = self.cfg
        self.L = c["num_hidden_layers"]; self.d = c["hidden_size"]
        self.nh = c["num_attention_heads"]; self.nkv = c["num_key_value_heads"]
        self.hd = c.get("head_dim") or self.d // self.nh
        self.eps = Fraction(str(c["rms_norm_eps"]))
        self.theta = c["rope_theta"]
        self.qk_norm = c.get("model_type") == "qwen3"
        if weights is None:
            files = sorted(model_dir.glob("*.safetensors"))
            weights = {}
            for f in files:
                weights.update(_load_bf16_as_f64(f))
        self.w = weights
        if "lm_head.weight" not in self.w or c.get("tie_word_embeddings"):
            self.w["lm_head.weight"] = self.w["model.embed_tokens.weight"]
        # pre-transpose for x @ W
        self.wt = {}
        for k, v in self.w.items():
            if v.ndim == 2 and "embed_tokens" not in k:
                self.wt[k] = np.ascontiguousarray(v.T)
        self.wt_abs = {k: np.abs(v) for k, v in self.wt.items()} if with_abs else {}
        for k in list(self.wt):
            self.w.pop(k, None)   # keep only transposed copies of 2-D weights
        self._rope_cache = None

    # ------------------------------------------------------------------ fp64
    def _rope_f64(self, T):
        half = self.hd // 2
        inv = self.theta ** (-np.arange(0, self.hd, 2, dtype=np.float64) / self.hd)
        ang = np.arange(T, dtype=np.float64)[:, None] * inv[None]
        c, s = np.cos(ang), np.sin(ang)
        return np.concatenate([c, c], -1), np.concatenate([s, s], -1)

    def _rms(self, x, g):
        eps = float(self.eps)
        return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g

    @staticmethod
    def _rot(x):
        h = x.shape[-1] // 2
        return np.concatenate([-x[..., h:], x[..., :h]], -1)

    def greedy(self, prompt: list[int], n_new: int, return_logits=False):
        """float64 KV-cached greedy decoding (candidate trace).
        With return_logits=True also returns the final normed hidden states (n_new+1, d)."""
        T = len(prompt) + n_new
        cos, sin = self._rope_f64(T)
        K = [np.zeros((self.nkv, T, self.hd)) for _ in range(self.L)]
        V = [np.zeros((self.nkv, T, self.hd)) for _ in range(self.L)]
        toks = list(prompt)
        out_logits = []
        pos = 0
        cur = list(prompt)
        while True:
            n = len(cur)
            x = self.w["model.embed_tokens.weight"][cur]          # (n, d)
            ps = np.arange(pos, pos + n)
            for l in range(self.L):
                p = f"model.layers.{l}."
                h = self._rms(x, self.w[p + "input_layernorm.weight"])
                q = (h @ self.wt[p + "self_attn.q_proj.weight"]).reshape(n, self.nh, self.hd).transpose(1, 0, 2)
                k = (h @ self.wt[p + "self_attn.k_proj.weight"]).reshape(n, self.nkv, self.hd).transpose(1, 0, 2)
                v = (h @ self.wt[p + "self_attn.v_proj.weight"]).reshape(n, self.nkv, self.hd).transpose(1, 0, 2)
                if self.qk_norm:
                    q = self._rms(q, self.w[p + "self_attn.q_norm.weight"])
                    k = self._rms(k, self.w[p + "self_attn.k_norm.weight"])
                q = q * cos[ps] + self._rot(q) * sin[ps]
                k = k * cos[ps] + self._rot(k) * sin[ps]
                K[l][:, ps] = k; V[l][:, ps] = v
                kk = K[l][:, : pos + n]; vv = V[l][:, : pos + n]
                rep = self.nh // self.nkv
                kk = np.repeat(kk, rep, 0); vv = np.repeat(vv, rep, 0)
                s = q @ kk.transpose(0, 2, 1) / math.sqrt(self.hd)
                mask = ps[:, None] < np.arange(pos + n)[None, :]
                s = np.where(mask[None], -np.inf, s)
                s = s - s.max(-1, keepdims=True)
                e = np.exp(s); a = e / e.sum(-1, keepdims=True)
                o = (a @ vv).transpose(1, 0, 2).reshape(n, self.nh * self.hd)
                x = x + o @ self.wt[p + "self_attn.o_proj.weight"]
                h = self._rms(x, self.w[p + "post_attention_layernorm.weight"])
                g = h @ self.wt[p + "mlp.gate_proj.weight"]
                u = h @ self.wt[p + "mlp.up_proj.weight"]
                x = x + (g / (1 + np.exp(-g)) * u) @ self.wt[p + "mlp.down_proj.weight"]
            h = self._rms(x[-1:], self.w["model.norm.weight"])
            logits = (h @ self.wt["lm_head.weight"])[0]
            if return_logits:
                out_logits.append(h[0].copy())   # final normed hidden; logits = h @ lm_head^T
            pos += n
            if len(toks) >= T:
                break
            nxt = int(logits.argmax())
            toks.append(nxt)
            cur = [nxt]
        gen = toks[len(prompt):]
        return (gen, np.stack(out_logits)) if return_logits else gen

    def logits_from_hidden(self, h: np.ndarray) -> np.ndarray:
        return h @ self.wt["lm_head.weight"]

    # ------------------------------------------------------------------ ball
    def _rms_ball(self, x: B.Ball, g: np.ndarray) -> B.Ball:
        n = x.m.shape[-1]
        s = B.div_exact(B.sum_last(B.square(x)), float(n))
        s = B.add(s, B.const(self.eps))
        return B.mul_exact(B.mul(x, B.rsqrt(s)), g)

    def _rope_ball(self, x: B.Ball, cos: B.Ball, sin: B.Ball) -> B.Ball:
        h = x.m.shape[-1] // 2
        rot = B.Ball(np.concatenate([-x.m[..., h:], x.m[..., :h]], -1),
                     np.concatenate([x.r[..., h:], x.r[..., :h]], -1))
        return B.add(B.mul(x, cos), B.mul(rot, sin))

    def forward_ball(self, tokens: list[int], keep_hidden=False) -> B.Ball:
        """Teacher-forced forward over the full sequence; returns logits Ball (T, V)."""
        T = len(tokens)
        if self._rope_cache is None or self._rope_cache[0].m.shape[0] < T:
            self._rope_cache = rope_tables(max(T, 256), self.hd, self.theta)
        cos = self._rope_cache[0][:T]; sin = self._rope_cache[1][:T]
        x = B.exact(self.w["model.embed_tokens.weight"][tokens])
        rep = self.nh // self.nkv
        causal = np.triu(np.ones((T, T), bool), 1)
        scale = 1.0 / math.sqrt(self.hd)
        if math.frexp(scale)[0] != 0.5:
            raise NotImplementedError("non power-of-two attention scale")
        hidden_r = []
        for l in range(self.L):
            p = f"model.layers.{l}."
            h = self._rms_ball(x, self.w[p + "input_layernorm.weight"])
            def proj(name, nheads):
                y = B.matmul_exact(h, self.wt[p + name], self.wt_abs[p + name])
                return y.reshape(T, nheads, self.hd).T(1, 0, 2)
            q = proj("self_attn.q_proj.weight", self.nh)
            k = proj("self_attn.k_proj.weight", self.nkv)
            v = proj("self_attn.v_proj.weight", self.nkv)
            if self.qk_norm:
                q = self._rms_ball(q, self.w[p + "self_attn.q_norm.weight"])
                k = self._rms_ball(k, self.w[p + "self_attn.k_norm.weight"])
            q = self._rope_ball(q, cos, sin)
            k = self._rope_ball(k, cos, sin)
            kk = B.Ball(np.repeat(k.m, rep, 0), np.repeat(k.r, rep, 0))
            vv = B.Ball(np.repeat(v.m, rep, 0), np.repeat(v.r, rep, 0))
            s = B.scale_pow2(B.matmul(q, kk.T(0, 2, 1)), scale)        # (nh, T, T)
            # softmax with causal mask: shift by max of mids (any shift is exact semantics)
            sm = np.where(causal[None], -np.inf, s.m)
            mx = sm.max(-1, keepdims=True)
            d = B.sub(s, B.exact(np.broadcast_to(mx, s.m.shape)))
            d = B.Ball(np.where(causal[None], 0.0, d.m), np.where(causal[None], 0.0, d.r))
            e = B.exp(d)
            e = B.Ball(np.where(causal[None], 0.0, e.m), np.where(causal[None], 0.0, e.r))
            a = B.div(e, B.sum_last(e))
            o = B.matmul(a, vv).T(1, 0, 2).reshape(T, self.nh * self.hd)
            x = B.add(x, B.matmul_exact(o, self.wt[p + "self_attn.o_proj.weight"],
                                        self.wt_abs[p + "self_attn.o_proj.weight"]))
            h = self._rms_ball(x, self.w[p + "post_attention_layernorm.weight"])
            g = B.matmul_exact(h, self.wt[p + "mlp.gate_proj.weight"], self.wt_abs[p + "mlp.gate_proj.weight"])
            u = B.matmul_exact(h, self.wt[p + "mlp.up_proj.weight"], self.wt_abs[p + "mlp.up_proj.weight"])
            act = B.mul(B.silu(g), u)
            x = B.add(x, B.matmul_exact(act, self.wt[p + "mlp.down_proj.weight"],
                                        self.wt_abs[p + "mlp.down_proj.weight"]))
            if keep_hidden:
                hidden_r.append(float((x.r / (np.abs(x.m) + 1e-30)).max()))
        h = self._rms_ball(x, self.w["model.norm.weight"])
        logits = B.matmul_exact(h, self.wt["lm_head.weight"], self.wt_abs["lm_head.weight"])
        if keep_hidden:
            self.last_hidden_relrad = hidden_r
        return logits
