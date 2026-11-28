"""Nonlinear (1-hidden-layer MLP, 512 units) board probe on the last-layer residual stream of the
trained Othello GPT and of its untrained twin, stratified by square age. Same splits as analyze.py.
Usage: CKPT=gpt_6k.pt python mlp_probe.py ; output mlp_probe_<ckpt>.json"""
import json, os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import OUT, load_lab, valid_mask, stratified
from gpt import GPT
from analyze import resid
torch.set_num_threads(int(os.environ.get("NTHREADS", "2"))); torch.manual_seed(0)
ck = os.environ.get("CKPT", "gpt_6k.pt")
ptr = tuple(x[:5000] for x in load_lab(["lab3"])); pte = tuple(x[:2000] for x in load_lab(["lab4"]))
vtr, vte = torch.as_tensor(valid_mask(ptr[3])), torch.as_tensor(valid_mask(pte[3]))
ytr = torch.as_tensor(ptr[1][vtr.numpy()].astype(np.int64)); yte = pte[1][vte.numpy()]; ate = pte[2][vte.numpy()]
torch.manual_seed(0); twin = GPT()
tr = GPT(); tr.load_state_dict(torch.load(os.path.join(OUT, ck))["model"])
res = {}
for name, m in [("trained", tr), ("untrained_twin", twin)]:
    m.eval(); t0 = time.time()
    Xtr = resid(m, ptr[0])[1][-1][vtr]; Xte = resid(m, pte[0])[1][-1][vte]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-5
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 512), nn.ReLU(), nn.Linear(512, 192))
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for ep in range(6):
        perm = torch.randperm(len(ytr))
        for i in range(0, len(ytr), 1024):
            j = perm[i:i + 1024]
            loss = F.cross_entropy(net((Xtr[j] - mu) / sd).view(-1, 3), ytr[j].view(-1)); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = torch.cat([net((Xte[i:i + 20000] - mu) / sd).view(-1, 64, 3).argmax(-1) for i in range(0, len(Xte), 20000)]).numpy()
    res[name] = stratified(pred == yte, ate)
    print(name, {k: round(v[0], 4) for k, v in res[name].items()}, f"{time.time()-t0:.0f}s", flush=True)
json.dump(res, open(os.path.join(OUT, f"mlp_probe_{ck.replace('.pt','')}.json"), "w"), indent=1)
