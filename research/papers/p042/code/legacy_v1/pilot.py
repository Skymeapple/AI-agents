import time, numpy as np
from sae import *
D=ZipfData(1024,96,1.4,6,seed=0)
print("p range",D.p[:3],D.p[-1], "expected occurrences per 1M tokens of last", D.p[-1]*1e6)
S=TopKSAE(96,2048,8,seed=0,lr=1e-3)
t=time.time()
for s in range(1,4001):
    x,_=D.sample(512); l=S.step(x)
    if s%500==0:
        mx,rec=recovery(S,D.F)
        T=s*512
        m=D.p*T
        print(s,round(time.time()-t,1),"loss",round(float(l),4),"fvu",round(fvu(S,D),4),"rec",rec.sum(),
              "dead",((S.t-S.last_fired)>200).sum(),
              "rec by m bins", [round(rec[(m>=a)&(m<b)].mean(),2) if ((m>=a)&(m<b)).any() else None for a,b in [(0,10),(10,30),(30,100),(100,300),(300,1e9)]])
