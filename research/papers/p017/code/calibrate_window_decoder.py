"""Calibrate learned window decoders against the EXACT suffix ceiling in the torus world.
Same decoder family as the Othello window decoders (slot-wise embedding sum + t embedding ->
2-layer ReLU MLP), trained on windows of the last k (obs, move) pairs. Output: calib_window.json"""
import json, os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import run_experiment as R
from world import TorusWorld, exact_curve
torch.set_num_threads(1); torch.manual_seed(0)
out = {}
for mode in ["balanced2", "landmark"]:
    for seed in [0, 1, 2]:
        rng = np.random.default_rng(seed); w = TorusWorld(5, mode, seed)
        tr, trp, _, _ = w.sample(14000, R.T, rng); w.sample(6000, R.T, rng)
        te, tep, temv, teob = w.sample(2000, R.T, rng)
        ce = exact_curve(w, temv, teob, R.EVAL_T)
        rec = {"exact": {}, "mlp": {}}
        for k in [1, 2, 3, 4, 6, 8, 12, 16]:
            def win(toks, pos):
                idx, ts, ys = [], [], []
                for t in R.EVAL_T:
                    kk = min(k, t)
                    seg = toks[:, 2 * (t - kk):2 * t]                    # obs_{t-kk} .. m_t
                    pad = np.full((len(toks), 2 * k - seg.shape[1]), w.vocab)
                    seg = np.concatenate([pad, seg], 1)
                    idx.append(seg + np.arange(2 * k)[None] * (w.vocab + 1)); ts.append(np.full(len(toks), t)); ys.append(pos[:, t])
                return torch.as_tensor(np.concatenate(idx)), torch.as_tensor(np.concatenate(ts)), torch.as_tensor(np.concatenate(ys))
            Xi, Xt, Y = win(tr, trp); Ti, Tt, TY = win(te, tep)
            e = nn.EmbeddingBag(2 * k * (w.vocab + 1), 512, mode="sum"); te_ = nn.Embedding(25, 512)
            f = nn.Sequential(nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 25))
            ps = list(e.parameters()) + list(te_.parameters()) + list(f.parameters())
            opt = torch.optim.Adam(ps, 1e-3); E, bs = 6, 1024; nb = len(Y) // bs
            sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=E * nb)
            for ep in range(E):
                perm = torch.randperm(len(Y))
                for i in range(nb):
                    j = perm[i * bs:(i + 1) * bs]
                    loss = F.cross_entropy(f(e(Xi[j]) + te_(Xt[j])), Y[j]); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            with torch.no_grad():
                acc = float((f(e(Ti) + te_(Tt)).argmax(1) == TY).float().mean())
            rec["exact"][k] = ce[k]; rec["mlp"][k] = acc
            print(mode, seed, k, f"exact={ce[k]:.4f} mlp={acc:.4f}", flush=True)
        out[f"{mode}_s{seed}"] = rec
        json.dump(out, open(os.path.join(R.RES, "calib_window.json"), "w"), indent=1)
