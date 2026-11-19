"""Rigorous midpoint-radius ("ball") arithmetic on numpy float64 arrays.

A Ball (m, r) encloses every real x with |x - m| <= r (elementwise).

Rigour assumptions (documented in docs/methodology.md):
  A0  IEEE-754 binary64, round-to-nearest-even, for + - * / sqrt (numpy on arm64).
  A1  BLAS dgemm computes each output entry as a sum of n products in *some* order
      (any order, with or without FMA; no Strassen-like algorithms). Then
      |fl(A@B) - A@B| <= gamma_n |A|@|B| holds (Higham, Thm 3.5 / Rump 2012).
  A2  Underflow contributions are absorbed by an absolute term TINY per value.
No libm function (exp, log, sin, cos, ...) is trusted: exp is implemented here
with rigorous range reduction + Taylor remainder; sin/cos come from Arb.

Radius computations are themselves done in round-to-nearest; every radius is
finally passed through `up`, which multiplies by (1 + 2^-40) and adds TINY.
That over-covers up to ~8000 accumulated relative roundings of size u,
far more than any single radius expression here performs.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

U = 2.0 ** -53                      # unit roundoff for binary64
INFL = 1.0 + 2.0 ** -40
TINY = 2.0 ** -1000                 # absorbs underflow (>> n * 2^-1074)


def up(x):
    return x * INFL + TINY


def gamma(n: int) -> float:
    # gamma_n = n u / (1 - n u), rounded upward generously
    return (n * U / (1.0 - n * U)) * INFL


class Ball:
    __slots__ = ("m", "r")

    def __init__(self, m, r=None):
        self.m = np.asarray(m, dtype=np.float64)
        self.r = np.zeros_like(self.m) if r is None else np.asarray(r, dtype=np.float64)

    @property
    def shape(self):
        return self.m.shape

    def __getitem__(self, idx):
        return Ball(self.m[idx], self.r[idx])

    def T(self, *axes):
        return Ball(np.transpose(self.m, axes or None), np.transpose(self.r, axes or None))

    def reshape(self, *s):
        return Ball(self.m.reshape(*s), self.r.reshape(*s))

    def lo(self):
        return self.m - self.r          # rounding here is irrelevant: used only via certify()

    def hi(self):
        return self.m + self.r


def exact(x) -> Ball:
    """Ball for float64 values that are exactly the intended reals (e.g. bf16 weights)."""
    return Ball(np.asarray(x, dtype=np.float64))


def const(q: Fraction | int | str) -> Ball:
    """Ball enclosing a rational constant (e.g. 1e-5 = 1/100000, which is not a float)."""
    q = Fraction(q)
    m = float(q)                                       # correctly rounded
    err = abs(Fraction(m) - q)
    r = float(err) * 2.0 + TINY                        # float(err) may round down; x2 covers it
    return Ball(np.float64(m), np.float64(r))


# ---------------------------------------------------------------- elementwise
def add(a: Ball, b: Ball) -> Ball:
    m = a.m + b.m
    return Ball(m, up(a.r + b.r + U * np.abs(m)))


def sub(a: Ball, b: Ball) -> Ball:
    m = a.m - b.m
    return Ball(m, up(a.r + b.r + U * np.abs(m)))


def mul(a: Ball, b: Ball) -> Ball:
    m = a.m * b.m
    r = np.abs(a.m) * b.r + a.r * np.abs(b.m) + a.r * b.r + U * np.abs(m)
    return Ball(m, up(r))


def mul_exact(a: Ball, w) -> Ball:
    """a * w where w is an exact float64 array/scalar."""
    m = a.m * w
    return Ball(m, up(a.r * np.abs(w) + U * np.abs(m)))


def scale_pow2(a: Ball, c: float) -> Ball:
    """Multiply by an exact power of two (exact unless under/overflow)."""
    assert c != 0 and math.frexp(abs(c))[0] == 0.5
    return Ball(a.m * c, up(a.r * abs(c)))


def div_exact(a: Ball, c: float) -> Ball:
    m = a.m / c
    return Ball(m, up(a.r / abs(c) + U * np.abs(m)))


def square(a: Ball) -> Ball:
    m = a.m * a.m
    return Ball(m, up(2.0 * np.abs(a.m) * a.r + a.r * a.r + U * m))


def div(a: Ball, b: Ball) -> Ball:
    """a / b, requires |b.m| > b.r elementwise."""
    bm_abs = np.abs(b.m)
    den_lo = (bm_abs - b.r) * (1.0 - 4 * U)
    if np.any(den_lo <= 0):
        raise FloatingPointError("division by ball containing 0")
    m = a.m / b.m
    # |a'/b' - a/b| <= (ar*|b| + |a|*br) / (|b| (|b|-br))  , plus rounding of m
    r = (a.r * bm_abs + np.abs(a.m) * b.r) / (bm_abs * den_lo) + U * np.abs(m)
    return Ball(m, up(r * (1 + 8 * U)))


def rsqrt(s: Ball) -> Ball:
    """1/sqrt(s), requires s > 0 on the ball. Lipschitz: |f'| <= 0.5 (s-r)^-1.5."""
    lo = (s.m - s.r) * (1.0 - 4 * U)
    if np.any(lo <= 0):
        raise FloatingPointError("rsqrt of nonpositive ball")
    m = 1.0 / np.sqrt(s.m)                     # sqrt and / correctly rounded: rel err <= ~2u
    lip = 0.5 / (lo * np.sqrt(lo)) * (1 + 8 * U)
    r = lip * s.r + 3 * U * np.abs(m)
    return Ball(m, up(r))


# ---------------------------------------------------------------- reductions / matmul
def sum_last(a: Ball, keepdims=True) -> Ball:
    n = a.m.shape[-1]
    g = gamma(n)
    m = a.m.sum(-1, keepdims=keepdims)
    s_abs = np.abs(a.m).sum(-1, keepdims=keepdims)
    s_r = a.r.sum(-1, keepdims=keepdims)
    return Ball(m, up((g * s_abs + s_r) * (1 + 2 * g)))


def matmul_exact(x: Ball, w: np.ndarray, w_abs: np.ndarray | None = None) -> Ball:
    """x @ w with w an exact float64 matrix (n x k)."""
    n = w.shape[0]
    g = gamma(n)
    wa = np.abs(w) if w_abs is None else w_abs
    m = x.m @ w
    S = np.abs(x.m) @ wa
    R = x.r @ wa
    return Ball(m, up((g * S + R) * (1 + 2 * g)))


def matmul(a: Ball, b: Ball) -> Ball:
    """Ball @ Ball (batched over leading dims)."""
    n = a.m.shape[-1]
    g = gamma(n)
    am_abs = np.abs(a.m)
    bm_abs = np.abs(b.m)
    m = a.m @ b.m
    S = am_abs @ bm_abs
    R = am_abs @ b.r + a.r @ up(bm_abs + b.r)
    return Ball(m, up((g * S + R) * (1 + 2 * g)))


# ---------------------------------------------------------------- exp (no libm)
_LN2 = Fraction(
    "0.6931471805599453094172321214581765680755001343602552541206800094933936219696947156058633269964186875"
)  # 100 digits; |ln2 - _LN2| < 1e-100
_LN2_HI = float(_LN2)
_LN2_LO = float(_LN2 - Fraction(_LN2_HI))
_LN2_ERR = float(abs(_LN2 - Fraction(_LN2_HI) - Fraction(_LN2_LO))) * 2 + 1e-99  # covers 1e-100 truncation
_NTAYLOR = 18
_COEF = [const(Fraction(1, math.factorial(k))) for k in range(_NTAYLOR + 1)]


def exp_point(x: np.ndarray) -> Ball:
    """Rigorous enclosure of exp(x) for float64 x (treated as exact reals).
    Values below -745 return ball [0, TINY]-ish; overflow raises."""
    x = np.asarray(x, dtype=np.float64)
    if np.any(x > 700):
        raise FloatingPointError("exp overflow")
    under = x < -740
    xs = np.where(under, 0.0, x)
    k = np.rint(xs / _LN2_HI)
    # r = x - k*ln2, ln2 = HI + LO + e
    xb = exact(xs)
    t1 = mul_exact(exact(k), _LN2_HI)             # k*HI with rounding
    t2 = mul_exact(exact(k), _LN2_LO)
    r = sub(sub(xb, t1), t2)
    r = Ball(r.m, up(r.r + np.abs(k) * _LN2_ERR))
    if np.any(np.abs(r.m) + r.r > 0.36):
        raise FloatingPointError("range reduction failed")
    # Horner, ball arithmetic
    p = Ball(np.full_like(r.m, _COEF[_NTAYLOR].m), np.full_like(r.m, _COEF[_NTAYLOR].r))
    for j in range(_NTAYLOR - 1, -1, -1):
        p = mul(p, r)
        p = Ball(p.m + _COEF[j].m, up(p.r + _COEF[j].r + U * np.abs(p.m + _COEF[j].m)))
    # Taylor remainder for |r| <= 0.36: |R| <= |r|^(N+1)/(N+1)! * e^0.36 < 2e-26 * ...
    rem = (0.36 ** (_NTAYLOR + 1)) / math.factorial(_NTAYLOR + 1) * 1.44
    p = Ball(p.m, up(p.r + rem))
    km = k.astype(np.int64)
    m = np.ldexp(p.m, km)
    rr = np.ldexp(p.r, km)
    out_r = up(rr + U * np.abs(m))                 # ldexp may round in subnormal range
    m = np.where(under, 0.0, m)
    out_r = np.where(under, TINY, out_r)
    return Ball(m, out_r)


def exp(a: Ball) -> Ball:
    """exp over a ball: exp([m-r, m+r]) within exp(m) * [e^-r, e^r]."""
    if np.any(a.r > 0.25):
        raise FloatingPointError("exp: input radius too large")
    e = exp_point(a.m)
    # e^r - 1 <= r (1 + r) for r <= 0.25
    grow = a.r * (1.0 + a.r)
    return Ball(e.m, up(e.r + (np.abs(e.m) + e.r) * grow * (1 + 4 * U)))


def silu(a: Ball) -> Ball:
    """x * sigmoid(x). Point value rigorously, input radius via global Lipschitz 1.0999 < 1.1."""
    xm = exact(a.m)
    big = a.m < -700                                # exp(-x) would overflow: |silu| < 1e-300
    xm_safe = exact(np.where(big, 0.0, a.m))
    e = exp_point(-xm_safe.m)
    den = Ball(1.0 + e.m, up(e.r + U * (1.0 + e.m)))
    y = div(xm_safe, den)
    m = np.where(big, 0.0, y.m)
    r = np.where(big, 1e-290, y.r)
    return Ball(m, up(r + 1.1 * a.r))


def certify_argmax(logits: Ball):
    """Per row: (top1 idx, certified?, margin lower bound, margin upper bound, runner-up idx by mid).
    Certified iff lo[top1] > max_{j != top1} hi[j]; then the exact argmax is top1.
    Exact margin L1 - L2 lies in [lo_i - max_j hi_j, hi_i - max_j lo_j]."""
    m, r = logits.m, logits.r
    i1 = m.argmax(-1)
    rows = np.arange(m.shape[0])
    lo = m - r
    hi = m + r
    lo_i = lo[rows, i1]
    hi_i = hi[rows, i1]
    hi_o = hi.copy(); hi_o[rows, i1] = -np.inf
    lo_o = lo.copy(); lo_o[rows, i1] = -np.inf
    m_o = m.copy(); m_o[rows, i1] = -np.inf
    i2 = m_o.argmax(-1)
    # directed-rounding safety on the bound subtractions
    mlo = (lo_i - hi_o.max(-1))
    mhi = (hi_i - lo_o.max(-1))
    slack = 4 * U * (np.abs(lo_i) + np.abs(hi_o.max(-1)))
    mlo = mlo - slack - TINY
    mhi = mhi + slack + TINY
    cert = mlo > 0
    return i1, cert, mlo, mhi, i2
