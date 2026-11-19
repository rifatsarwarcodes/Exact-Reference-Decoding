from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, DynamicCache

DT = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16, "fp64": torch.float64}


class TorchBackend:
    def __init__(self, model_dir, device="cpu", dtype="fp32", attn="sdpa"):
        self.name = f"torch-{device}-{dtype}" + ("" if attn == "sdpa" else f"-{attn}")
        self.dev = torch.device(device)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=DT[dtype], attn_implementation=attn).to(self.dev).eval()
        self.meta = dict(torch=torch.__version__, device=device, dtype=dtype, attn=attn,
                         threads=torch.get_num_threads())
        self.cache = None

    def reset(self):
        self.cache = DynamicCache()

    @torch.no_grad()
    def _fwd(self, ids):
        out = self.model(input_ids=torch.tensor([ids], device=self.dev),
                         past_key_values=self.cache, use_cache=True)
        self.cache = out.past_key_values
        return out.logits[0, -1].to(torch.float64).cpu().numpy()

    def start(self, ids):
        return self._fwd(list(ids))

    def step(self, tok):
        return self._fwd([tok])
