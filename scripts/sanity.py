import time, numpy as np, torch, sys
sys.path.insert(0,'.')
from transformers import AutoModelForCausalLM, AutoTokenizer
from reference.llama import Fp64Model
from reference import ball as B
md='models/SmolLM2-135M'
tok=AutoTokenizer.from_pretrained(md)
ids=tok("The capital of France is Paris. The capital of Germany is").input_ids
print('prompt len',len(ids))
t=time.time(); M=Fp64Model(md); print('load',time.time()-t)
t=time.time(); gen,hid=M.greedy(ids,32,return_logits=True); lg=M.logits_from_hidden(hid)[:32]; print('fp64 greedy',time.time()-t, repr(tok.decode(gen)))
hf=AutoModelForCausalLM.from_pretrained(md,torch_dtype=torch.float64).eval()
with torch.no_grad():
    L=hf(torch.tensor([ids+gen])).logits[0].numpy()
ours=lg  # logits at positions len(ids)-1 ... len(ids)+31
ref=L[len(ids)-1:len(ids)+32]
print('max |numpy fp64 - HF fp64|', np.abs(ours-ref).max())
t=time.time(); lb=M.forward_ball(ids+gen, keep_hidden=True); dt=time.time()-t
print('ball forward',dt,'s for T=',len(ids+gen))
print('max |ball mid - fp64 decode logits|', np.abs(lb.m[len(ids)-1:len(ids)+32]-ours).max())
print('logit radius: max',lb.r.max(),'median',np.median(lb.r))
print('hidden rel rad per layer',['%.1e'%v for v in M.last_hidden_relrad[::5]])
i1,cert,mlo,mhi,i2=B.certify_argmax(lb[len(ids)-1:len(ids)+31])
print('certified',cert.sum(),'/',len(cert),'argmax==gen', (i1==np.array(gen[:31])).all())
print('margins lo',np.round(mlo[:10],4))
