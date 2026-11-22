"""Track next-obs accuracy, probe accuracy, k_eff and CRA over training of a GRU.

Usage: python training_dynamics.py --mode landmark --seed 0 --epochs 40
Output: ../results/dyn_<mode>_s<seed>.json
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import run_experiment as R
from world import TorusWorld
from models import GRULM, fit_readout

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="landmark")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--checkpoints", nargs="+", type=int, default=[1, 2, 5, 10, 20, 30, 40])
a = ap.parse_args()
torch.set_num_threads(int(os.environ.get("NTHREADS", "4")))
rng = np.random.default_rng(a.seed)
torch.manual_seed(a.seed)
world = TorusWorld(5, a.mode, a.seed)
T = R.T
tr_toks, _, _, _ = world.sample(14000, T, rng)
pr_toks, pr_pos, pr_mv, pr_ob = world.sample(6000, T, rng)
te_toks, te_pos, te_mv, te_ob = world.sample(2000, T, rng)
curve = R.suffix_curve(world, te_mv, te_ob, te_pos, R.EVAL_T)

m = GRULM(world.vocab, 64)
opt = torch.optim.Adam(m.parameters(), lr=3e-3)
X = torch.as_tensor(tr_toks)
bs = 128
out = {"mode": a.mode, "seed": a.seed, "suffix_curve": curve, "checkpoints": []}
for ep in range(1, a.epochs + 1):
    m.train()
    perm = torch.randperm(X.shape[0])
    for i in range(X.shape[0] // bs):
        b = X[perm[i * bs:(i + 1) * bs]]
        logits, _ = m(b[:, :-1])
        lg, tg = logits[:, 1::2], b[:, 1:][:, 1::2]
        loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tg.reshape(-1))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if ep in a.checkpoints:
        m.eval()
        H_pr = R.encode(m, pr_toks, "rnn"); H_te = R.encode(m, te_toks, "rnn")
        Xp = R.hid_at(H_pr, R.EVAL_T).reshape(-1, 64); Xt = R.hid_at(H_te, R.EVAL_T).reshape(-1, 64)
        yp = pr_pos[:, R.EVAL_T].reshape(-1); yt = te_pos[:, R.EVAL_T].reshape(-1)
        ot = te_ob[:, R.EVAL_T].reshape(-1)
        with torch.no_grad():
            ol = m.out(Xt)[:, 4:]
        po = ol.argmax(1).numpy()
        W, b_ = fit_readout(Xp, yp, world.S)
        pa = float(((Xt @ W.T + b_).argmax(1).numpy() == yt).mean())
        keff = next((k for k, acc in enumerate(curve) if acc >= pa), None)
        ro = R.rollout_test(m, "rnn", world, te_toks, te_pos, te_mv, H_te, W, None, None, a.seed)
        rec = {"epoch": ep, "obs_acc": float((po == ot).mean()),
               "obs_nll": float(F.cross_entropy(ol, torch.as_tensor(ot)).item()),
               "probe_acc": pa, "k_eff": keff,
               "cra_probe": ro["probe_subspace"]["cf_disc"], "cra_random": ro["random_subspace"]["cf_disc"],
               "cra_full": ro["full_swap"]["cf_disc"]}
        if a.mode == "landmark":
            rec["landmark_recall"] = float((po[ot == 1] == 1).mean())
        out["checkpoints"].append(rec)
        print(rec, flush=True)
json.dump(out, open(os.path.join(R.RES, f"dyn_{a.mode}_s{a.seed}.json"), "w"), indent=1)
