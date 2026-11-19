import sys, numpy as np
sys.path.insert(0,'.')
from reference.llama import Fp64Model
from reference import ball as B
M=Fp64Model('models/SmolLM2-135M')
ids=[504, 3575, 282, 4649, 314, 7042, 30, 378, 3575, 282, 4653, 314]
# monkeypatch ops to log relative radius
orig={n:getattr(B,n) for n in ['matmul_exact','matmul','exp','div','rsqrt','add','silu']}
log=[]
def wrap(n):
    f=orig[n]
    def g(*a,**k):
        out=f(*a,**k)
        rel=(out.r/(np.abs(out.m)+1e-300))
        log.append((n, float(np.max(out.r)), float(np.median(rel))))
        return out
    return g
for n in orig: setattr(B,n,wrap(n))
try:
    M.forward_ball(ids)
except Exception as e: print('ERR',e)
for i,(n,r,rel) in enumerate(log[:60]): print(i,n,'%.2e'%r,'%.2e'%rel)
