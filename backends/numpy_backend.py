"""Our float64 numpy implementation as a 'backend' (KV-cached decode path)."""
from __future__ import annotations

import platform

import numpy as np

from reference.llama import Fp64Model


class NumpyBackend:
    def __init__(self, model_dir):
        self.m = Fp64Model(model_dir, with_abs=False)
        self.name = "numpy-cpu-fp64"
        self.meta = dict(numpy=np.__version__, machine=platform.machine())
        self._ids = None

    # Simple (non-incremental) implementation: re-run greedy with n_new=0 on the growing prefix
    # is O(T^2); instead keep the full token list and call the cached decoder once per step.
    def reset(self):
        self._ids = []

    def _logits(self):
        _, hid = self.m.greedy(self._ids, 0, return_logits=True)
        return self.m.logits_from_hidden(hid[-1])

    def start(self, ids):
        self._ids = list(ids)
        return self._logits()

    def step(self, tok):
        self._ids.append(int(tok))
        return self._logits()
