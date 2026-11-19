"""Rigorous Arb (ball arithmetic, p-bit midpoints) teacher-forced forward pass.

Encloses the exact real-arithmetic final hidden state of a Llama-family model
(currently: no qk-norm, i.e. SmolLM2) for every position of a token sequence.
The lm_head (one matmul) is then done in rigorous float64 ball arithmetic
(reference/ball.py), which is tight because a single matmul has no wrapping.

Everything here is rigorous given python-flint/Arb correctness: Arb tracks
midpoint p-bit rounding and propagates radii for every operation.
"""
from __future__ import annotations

import math
import time
from fractions import Fraction

import numpy as np
from flint import arb, arb_mat, ctx

from . import ball as B
from .llama import Fp64Model


def _to_arb_mat(a: np.ndarray) -> arb_mat:
    a = np.ascontiguousarray(a, dtype=np.float64)
    return arb_mat(a.shape[0], a.shape[1], a.ravel().tolist())  # floats are exact in arb


def _rows(m: arb_mat) -> list[list[arb]]:
    e = m.entries(); c = m.ncols()
    return [e[i * c:(i + 1) * c] for i in range(m.nrows())]


def _from_rows(rows: list[list[arb]]) -> arb_mat:
    return arb_mat(len(rows), len(rows[0]), [v for r in rows for v in r])


class ArbModel:
    def __init__(self, fm: Fp64Model, prec=(40, 6)):
        self.fm = fm
        self.prec = prec
        self.eps = fm.eps
        self._wcache: dict[int, dict[str, arb_mat]] = {}

    def _layer_weights(self, l: int) -> dict[str, arb_mat]:
        fm = self.fm; p = f"model.layers.{l}."
        g1 = fm.w[p + "input_layernorm.weight"][:, None]
        g2 = fm.w[p + "post_attention_layernorm.weight"][:, None]
        # fold RMSNorm gains into the following projections: bf16*bf16 is exact in f64
        def fold(g, name):
            w = fm.wt[p + name] * g
            assert np.array_equal(w / g, fm.wt[p + name]) or np.all(g != 0)
            return _to_arb_mat(w)
        qkv = np.concatenate([fm.wt[p + "self_attn.q_proj.weight"],
                              fm.wt[p + "self_attn.k_proj.weight"],
                              fm.wt[p + "self_attn.v_proj.weight"]], 1) * g1
        gu = np.concatenate([fm.wt[p + "mlp.gate_proj.weight"],
                             fm.wt[p + "mlp.up_proj.weight"]], 1) * g2
        return {
            "qkv": _to_arb_mat(qkv),
            "o": _to_arb_mat(fm.wt[p + "self_attn.o_proj.weight"]),
            "gu": _to_arb_mat(gu),
            "down": _to_arb_mat(fm.wt[p + "mlp.down_proj.weight"]),
        }

    def _rs(self, X: list[list[arb]]) -> list[arb]:
        d = len(X[0]); eps = arb(self.eps.numerator) / self.eps.denominator
        out = []
        for row in X:
            s = arb(0)
            for v in row:
                s += v * v
            out.append((s / d + eps).rsqrt())
        return out

    def prec_at(self, l: int) -> int:
        """Per-layer midpoint precision. Radii grow ~10^5.6 per layer (measured); we budget
        10^6 per remaining layer plus `target` digits. Rigour does not depend on this
        schedule (Arb accounts for all rounding); only tightness does."""
        if isinstance(self.prec, int):
            return self.prec
        target, per_layer = self.prec
        return int((target + per_layer * (self.fm.L - l)) * 3.33) + 64

    def forward(self, seqs: list[list[int]], log=None):
        """Teacher-forced forward for a batch of sequences (processed layer by layer).
        Returns list of (h_mid f64 (T,d), h_rad f64 (T,d)) enclosing the final RMS-normalised
        hidden state BEFORE the final RMSNorm gain (that gain is folded into the lm_head)."""
        fm = self.fm
        nh, nkv, hd = fm.nh, fm.nkv, fm.hd
        rep = nh // nkv; half = hd // 2
        nq = nh * hd; nk = nkv * hd
        Ts = [len(s) for s in seqs]; Tmax = max(Ts)
        ctx.prec = self.prec_at(0)
        th = arb(Fraction(fm.theta).numerator) / Fraction(fm.theta).denominator
        inv = [th ** (arb(-2 * i) / hd) for i in range(half)]
        cos = [[(inv[i] * t).cos() for i in range(half)] for t in range(Tmax)]
        sin = [[(inv[i] * t).sin() for i in range(half)] for t in range(Tmax)]
        scale = arb(1) / arb(hd).sqrt()
        Xs = [[[arb(float(v)) for v in fm.w["model.embed_tokens.weight"][t]] for t in s] for s in seqs]
        offs = np.cumsum([0] + Ts)

        def rope(vec, t):
            out = list(vec)
            c = cos[t]; sn = sin[t]
            for i in range(half):
                a, b = vec[i], vec[i + half]
                out[i] = a * c[i] - b * sn[i]
                out[i + half] = b * c[i] + a * sn[i]
            return out

        t0 = time.time()
        for l in range(fm.L):
            ctx.prec = self.prec_at(l)
            W = self._layer_weights(l)
            # ---- attention (projections batched over all sequences)
            Xall = [row for X in Xs for row in X]
            rs = self._rs(Xall)
            H = _from_rows([[v * r for v in row] for row, r in zip(Xall, rs)])
            QKVall = _rows(H * W["qkv"])
            new_Xs = []
            Oall = []
            for si, T in enumerate(Ts):
                QKV = QKVall[offs[si]:offs[si + 1]]
                if fm.qk_norm:   # Qwen3: per-head RMSNorm (with gain) on q and k before RoPE
                    gq = [arb(float(v)) for v in fm.w[f"model.layers.{l}.self_attn.q_norm.weight"]]
                    gk = [arb(float(v)) for v in fm.w[f"model.layers.{l}.self_attn.k_norm.weight"]]
                    def qkn(vec, g):
                        r = self._rs([vec])[0]
                        return [v * r * gi for v, gi in zip(vec, g)]
                else:
                    qkn = lambda vec, g: vec
                    gq = gk = None
                Q = [[rope(qkn(QKV[t][h * hd:(h + 1) * hd], gq), t) for h in range(nh)] for t in range(T)]
                K = [[rope(qkn(QKV[t][nq + h * hd: nq + (h + 1) * hd], gk), t) for h in range(nkv)] for t in range(T)]
                Vv = [[QKV[t][nq + nk + h * hd: nq + nk + (h + 1) * hd] for h in range(nkv)] for t in range(T)]
                O = [[None] * nh for _ in range(T)]
                for h in range(nh):
                    kv = h // rep
                    Qm = _from_rows([Q[t][h] for t in range(T)])
                    Km = _from_rows([K[t][kv] for t in range(T)])
                    Vm = _from_rows([Vv[t][kv] for t in range(T)])
                    S = _rows(Qm * Km.transpose())
                    P = []
                    for t in range(T):
                        row = [S[t][j] * scale for j in range(t + 1)]
                        mx = max(float(v.mid()) for v in row)
                        e = [(v - mx).exp() for v in row]
                        z = arb(0)
                        for v in e:
                            z += v
                        zi = 1 / z
                        P.append([v * zi for v in e] + [arb(0)] * (T - t - 1))
                    Om = _rows(_from_rows(P) * Vm)
                    for t in range(T):
                        O[t][h] = Om[t]
                Oall.extend([[v for h in range(nh) for v in O[t][h]] for t in range(T)])
            A = _rows(_from_rows(Oall) * W["o"])
            Xall = [[x + a for x, a in zip(xr, ar)] for xr, ar in zip(Xall, A)]
            # ---- MLP
            rs = self._rs(Xall)
            H = _from_rows([[v * r for v in row] for row, r in zip(Xall, rs)])
            GU = _rows(H * W["gu"])
            I = len(GU[0]) // 2
            act = _from_rows([[row[j] / (1 + (-row[j]).exp()) * row[I + j] for j in range(I)] for row in GU])
            D = _rows(act * W["down"])
            Xall = [[x + a for x, a in zip(xr, ar)] for xr, ar in zip(Xall, D)]
            Xs = [Xall[offs[si]:offs[si + 1]] for si in range(len(Ts))]
            del W, H, GU, act, D, A, QKVall
            if log is not None:
                rr = max(float(v.rad()) / (abs(float(v.mid())) + 1e-300) for row in Xall for v in row)
                log.append((l, ctx.prec, time.time() - t0, rr))
                print(f"  layer {l:2d} prec {ctx.prec} t={time.time()-t0:7.1f}s maxrelrad {rr:.2e}", flush=True)
        outs = []
        for X in Xs:
            rs = self._rs(X)
            Hn = [[v * r for v in row] for row, r in zip(X, rs)]
            mid = np.array([[float(v.mid()) for v in row] for row in Hn])
            # |true - fl(mid_arb)| <= |fl(mid)-mid_arb| + rad_arb, enclosed by Arb and rounded up
            rad = np.array([[float((abs(arb(m) - v)).upper()) for m, v in zip(mr, row)] for mr, row in zip(mid, Hn)])
            outs.append((mid, B.up(rad * 2.0)))
        return outs

    def lm_head_ball(self, mid, rad) -> B.Ball:
        g = self.fm.w["model.norm.weight"][:, None]
        Wl = self.fm.wt["lm_head.weight"] * g            # exact: bf16 * bf16
        return B.matmul_exact(B.Ball(mid, rad), Wl)

    def logits_ball(self, tokens: list[int], log=None) -> B.Ball:
        (mid, rad), = self.forward([tokens], log=log)
        return self.lm_head_ball(mid, rad)
