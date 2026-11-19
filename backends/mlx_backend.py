from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

DT = {"fp32": mx.float32, "bf16": mx.bfloat16, "fp16": mx.float16}


class MlxBackend:
    def __init__(self, model_dir, device="gpu", dtype="fp32"):
        self.name = f"mlx-{device}-{dtype}"
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
        self.model, _ = load(str(model_dir))
        self.model.set_dtype(DT[dtype])
        import mlx_lm
        self.meta = dict(mlx=mx.__version__, mlx_lm=mlx_lm.__version__, device=device, dtype=dtype)
        self.cache = None

    def reset(self):
        self.cache = make_prompt_cache(self.model)

    def _fwd(self, ids):
        lg = self.model(mx.array([ids]), cache=self.cache)[0, -1]
        lg = lg.astype(mx.float32)
        mx.eval(lg)
        return np.array(lg).astype(np.float64)

    def start(self, ids):
        return self._fwd(list(ids))

    def step(self, tok):
        return self._fwd([tok])
