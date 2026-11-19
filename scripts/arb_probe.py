import sys, time, numpy as np
sys.path.insert(0,'.')
from reference.llama import Fp64Model
from reference.arb_forward import ArbModel
from reference import ball as B
M=Fp64Model('models/SmolLM2-135M')
ids=[504, 3575, 282, 4649, 314, 7042, 30, 378, 3575, 282, 4653, 314]
prec=int(sys.argv[1])
gen,hid=M.greedy(ids,4,return_logits=True); lg=M.logits_from_hidden(hid)[:4]
seq=ids+gen[:4]
A=ArbModel(M,prec); log=[]
t=time.time(); lb=A.logits_ball(seq,log=log); print('prec',prec,'T',len(seq),'time',round(time.time()-t,1))
for l,tt,rr in log[::3]+[log[-1]]: print(' layer',l,'t=%.1f'%tt,'max rel rad %.2e'%rr)
print('logit rad max %.2e'%lb.r.max())
print('|arb mid - fp64 logits| at gen steps', np.abs(lb.m[len(ids)-1:]-lg).max())
i1,cert,mlo,mhi,i2=B.certify_argmax(lb[len(ids)-1:len(ids)+3]); print(cert, mlo)
