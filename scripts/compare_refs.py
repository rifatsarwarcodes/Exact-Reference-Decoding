"""Compare two reference directories: token traces and final hidden states / logits."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from backends.common import lm_head  # noqa: E402

a, b = Path(sys.argv[1]), Path(sys.argv[2])
model = sys.argv[3] if len(sys.argv) > 3 else "models/SmolLM2-135M"
W = lm_head(model)
same, n, worst = 0, 0, 0.0
for f in sorted(b.glob("*.npz")):
    g = a / f.name
    if not g.exists():
        continue
    x, y = np.load(g), np.load(f)
    k = min(len(x["gen"]), len(y["gen"]))
    n += 1
    same += int(np.array_equal(x["gen"][:k], y["gen"][:k]))
    worst = max(worst, float(np.abs(x["hidden"][:k] @ W - y["hidden"][:k] @ W).max()))
print(f"| refs compared | identical traces | max logit diff |\n|---|---|---|\n| {n} | {same} | {worst:.2e} |")
