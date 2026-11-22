import os,sys,time; os.environ["OMP_NUM_THREADS"]="1"
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from sae import ZipfData,TopKSAE,recovery,fvu
N,D,M,K,L0=int(sys.argv[1]),int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4]),float(sys.argv[5]); a=float(sys.argv[6]); T=int(sys.argv[7]); lr=float(sys.argv[8]) if len(sys.argv)>8 else 1e-3
data=ZipfData(N,D,a,L0); sae=TopKSAE(D,M,K,lr=lr); B=256
print('pmin',data.p.min(),'pmax',data.p.max(), 'n capped',(data.p>=0.5).sum())
t0=time.time()
ck=set(np.geomspace(100,T//B,10).astype(int))
for s in range(1,T//B+1):
    x,_=data.sample(B); sae.step(x)
    if s in ck:
        mx,r=recovery(sae,data.F); m=data.p*s*B
        bins=[0,10,30,100,300,1000,3000,1e4,1e9]
        h=[f"{r[(m>=bins[i])&(m<bins[i+1])].mean():.2f}" if ((m>=bins[i])&(m<bins[i+1])).sum() else '-' for i in range(len(bins)-1)]
        print(s*B, f"rec={r.mean():.3f} fvu={fvu(sae,data):.3f} dead={((sae.t-sae.last_fired)>200).sum()} t={time.time()-t0:.0f}s", h, flush=True)
