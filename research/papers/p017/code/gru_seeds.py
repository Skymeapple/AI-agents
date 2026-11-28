"""Review M5: 10 independent seeds of the trained GRU per non-local world, at 10 and 30 epochs
of the SAME one-cycle schedule as the main runs (removes the schedule confound). Seeds 0-2 at
10 epochs reproduce the main-table GRUs. k_eff uses the exact finite-time ceiling.

Usage: python gru_seeds.py --mode landmark --seeds 0 1 2 3 4 --epochs 10
Output: ../results/seeds/gru_<mode>_e<epochs>_s<seed>.json
"""
import argparse, json, os, time
import numpy as np, torch
import run_experiment as R
from world import TorusWorld, exact_curve
from models import GRULM, train_lm, fit_readout

ap = argparse.ArgumentParser()
ap.add_argument("--mode"); ap.add_argument("--seeds", nargs="+", type=int); ap.add_argument("--epochs", type=int, default=10)
a = ap.parse_args()
out_dir = os.path.join(R.RES, "seeds"); os.makedirs(out_dir, exist_ok=True)
for seed in a.seeds:
    t0 = time.time()
    rng = np.random.default_rng(seed); torch.manual_seed(seed)
    w = TorusWorld(5, a.mode, seed)
    tr, _, _, _ = w.sample(14000, R.T, rng)
    pr, pr_pos, _, pr_ob = w.sample(6000, R.T, rng)
    te, te_pos, te_mv, te_ob = w.sample(2000, R.T, rng)
    ce = exact_curve(w, te_mv, te_ob, R.EVAL_T)
    g = GRULM(w.vocab, 64)
    train_lm(g, tr, epochs=a.epochs, bs=128, lr=6e-3, seed=seed)
    g.eval()
    Hp, Ht = R.encode(g, pr, "rnn"), R.encode(g, te, "rnn")
    Xp = R.hid_at(Hp, R.EVAL_T).reshape(-1, 64); Xt = R.hid_at(Ht, R.EVAL_T).reshape(-1, 64)
    yp = pr_pos[:, R.EVAL_T].reshape(-1); yt = te_pos[:, R.EVAL_T].reshape(-1); ot = te_ob[:, R.EVAL_T].reshape(-1)
    with torch.no_grad():
        po = g.out(Xt)[:, 4:].argmax(1).numpy()
    W, b = fit_readout(Xp, yp, w.S)
    pa = float(((Xt @ W.T + b).argmax(1).numpy() == yt).mean())
    keff = next((k for k, x in enumerate(ce) if x >= pa - 1e-12), None)
    ro = R.rollout_test(g, "rnn", w, te, te_pos, te_mv, Ht, W, None, None, seed)
    rec = {"mode": a.mode, "seed": seed, "epochs": a.epochs, "obs_acc": float((po == ot).mean()),
           "probe_acc": pa, "k_eff_exact": keff, "exact_curve": ce,
           "cra": {k: v["cf_disc"] for k, v in ro.items()}}
    if a.mode == "landmark":
        rec["landmark_recall"] = float((po[ot == 1] == 1).mean())
    json.dump(rec, open(os.path.join(out_dir, f"gru_{a.mode}_e{a.epochs}_s{seed}.json"), "w"), indent=1)
    print(a.mode, a.epochs, seed, f"obs={rec['obs_acc']:.3f} probe={pa:.3f} keff={keff}",
          {k: round(v, 3) for k, v in rec["cra"].items()}, f"{time.time()-t0:.0f}s", flush=True)
