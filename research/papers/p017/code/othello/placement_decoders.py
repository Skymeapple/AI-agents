"""'Placement' floors for Othello: decoders that see WHICH squares each player has played so far
(order-free, split by move parity, i.e. disc placement without flips) plus the move index t,
optionally together with the last k moves in order. Placement is a permutation-invariant,
parity-weighted bag of the history: an attention layer that averages token embeddings with a
parity-dependent sign can expose it without simulating any flip.
Train: lab1+lab2; test: lab4. Output: placement.json"""
import json, os, sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import OUT, load_lab, valid_mask, stratified
torch.set_num_threads(int(os.environ.get("NTHREADS", "1"))); torch.manual_seed(0)


def feats(toks, lens, k):
    """Sparse index lists: parity-split bag (index p*60+tok, p = parity of the move's position relative
    to t, so 'moves by the player who made move t' vs 'by the other') + optional last-k slots."""
    v = valid_mask(lens); gi, ti = np.nonzero(v)
    L = 60
    rows = []
    pos = np.arange(L)
    for g, t in zip(gi, ti):
        tk = toks[g, :t + 1]
        par = (t - pos[:t + 1]) % 2
        idx = list(par * 60 + tk)
        for s in range(k):
            if t - s >= 0:
                idx.append(120 + s * 60 + toks[g, t - s])
        rows.append(idx)
    return gi, ti, rows


def pack(rows):
    off = np.cumsum([0] + [len(r) for r in rows[:-1]])
    return torch.as_tensor(np.concatenate(rows)), torch.as_tensor(off)


class Dec(nn.Module):
    def __init__(self, k, h=512):
        super().__init__()
        self.e = nn.EmbeddingBag(120 + 60 * max(k, 1), h, mode="sum"); self.t = nn.Embedding(60, h)
        self.f = nn.Sequential(nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 64 * 3))

    def forward(self, flat, off, t):
        return self.f(self.e(flat, off) + self.t(t)).view(-1, 64, 3)


tr = load_lab(["lab1", "lab2"]); te = load_lab(["lab4"])
res = {}
for k in [int(x) for x in sys.argv[1:]] or [0, 4, 8]:
    t0 = time.time()
    gi, ti, rows = feats(tr[0], tr[3], k)
    y = torch.as_tensor(tr[1][gi, ti].astype(np.int64)); tt = torch.as_tensor(ti)
    m = Dec(k); opt = torch.optim.Adam(m.parameters(), 1e-3)
    E, bs = 4, 1024; nb = len(y) // bs
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=E * nb)
    for ep in range(E):
        perm = np.random.default_rng(ep).permutation(len(y))
        for i in range(nb):
            j = perm[i * bs:(i + 1) * bs]
            fl, of = pack([rows[q] for q in j])
            loss = F.cross_entropy(m(fl, of, tt[j]).reshape(-1, 3), y[j].reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    gj, tj, rj = feats(te[0], te[3], k)
    preds = []
    with torch.no_grad():
        for i in range(0, len(rj), 8192):
            fl, of = pack(rj[i:i + 8192]); preds.append(m(fl, of, torch.as_tensor(tj[i:i + 8192])).argmax(-1).numpy())
    corr = np.concatenate(preds) == te[1][gj, tj]
    res[f"placement+last{k}"] = stratified(corr, te[2][gj, tj])
    print(k, {a: round(b[0], 4) for a, b in res[f"placement+last{k}"].items()}, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(os.path.join(OUT, "placement.json"), "w"), indent=1)
