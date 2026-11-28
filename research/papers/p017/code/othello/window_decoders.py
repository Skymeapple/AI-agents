"""Lower-bound estimates of the Othello suffix ceiling Acc*_k: dedicated MLP decoders that see
only the last k move tokens plus the absolute move index t (what any k-suffix-measurable
predictor with positional information can use), trained with board supervision.
Train: data_lab1+lab2 (20k games); test: data_lab4. Output: window_mlp.json"""
import json, os, sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import OUT, load_lab, valid_mask, stratified
torch.set_num_threads(int(os.environ.get("NTHREADS", "2"))); torch.manual_seed(0)
KS = [int(x) for x in sys.argv[1:]] or [0, 1, 2, 3, 4, 6, 8, 12, 16]


def windows(toks, lens, k):
    N, L = toks.shape
    v = valid_mask(lens)
    gi, ti = np.nonzero(v)
    idx = np.zeros((len(gi), max(k, 1)), np.int64)   # slot s holds token at t-s, offset by slot*61
    for s in range(k):
        tt = ti - s
        tok = np.where(tt >= 0, toks[gi, np.maximum(tt, 0)], 60)
        idx[:, s] = s * 61 + tok
    if k == 0:
        idx[:, 0] = 60  # constant pad token -> decoder uses t only
    return gi, ti, idx


class Dec(nn.Module):
    def __init__(self, k, h=512):
        super().__init__()
        self.e = nn.EmbeddingBag(max(k, 1) * 61, h, mode="sum"); self.t = nn.Embedding(60, h)
        self.f = nn.Sequential(nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 64 * 3))

    def forward(self, idx, t):
        return self.f(self.e(idx) + self.t(t)).view(-1, 64, 3)


tr = load_lab(["lab1", "lab2"]); te = load_lab(["lab4"])
res = json.load(open(os.path.join(OUT, "window_mlp.json"))) if os.path.exists(os.path.join(OUT, "window_mlp.json")) else {}
for k in KS:
    t0 = time.time()
    gi, ti, idx = windows(tr[0], tr[3], k)
    y = torch.as_tensor(tr[1][gi, ti].astype(np.int64))
    idx, tt = torch.as_tensor(idx), torch.as_tensor(ti)
    m = Dec(k); opt = torch.optim.Adam(m.parameters(), 1e-3)
    E = 4; bs = 1024; nb = len(y) // bs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=E * nb)
    for ep in range(E):
        perm = torch.randperm(len(y))
        for i in range(nb):
            j = perm[i * bs:(i + 1) * bs]
            loss = F.cross_entropy(m(idx[j], tt[j]).reshape(-1, 3), y[j].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    gj, tj, idj = windows(te[0], te[3], k)
    with torch.no_grad():
        pred = torch.cat([m(torch.as_tensor(idj[i:i + 8192]), torch.as_tensor(tj[i:i + 8192])).argmax(-1)
                          for i in range(0, len(tj), 8192)]).numpy()
    corr = pred == te[1][gj, tj]
    st = stratified(corr, te[2][gj, tj])
    # train accuracy (memorisation check)
    with torch.no_grad():
        ptr = m(idx[:50000], tt[:50000]).argmax(-1).numpy()
    st["train_all"] = (float((ptr == y[:50000].numpy()).mean()), 50000)
    res[str(k)] = st
    print(k, {a: round(b[0], 4) for a, b in st.items()}, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(os.path.join(OUT, "window_mlp.json"), "w"), indent=1)
