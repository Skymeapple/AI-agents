"""Experiment 4 (revision): does the certification lag reduce to margin / loss / neural-collapse
dynamics?  Modular addition MLP (same architecture and optimiser as Experiment 1), more seeds,
finer checkpoints (every 250 steps), larger certificate sample (700 held-out pairs), and a
weight-decay sweep.  Usage: python3 exp_controls.py  -> results/controls_mlp_*.json"""
import sys, os, json, time
import numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
from nn import MLP
from ckpt_metrics import checkpoint_stats
import exp_modadd as E

RES = os.path.join(os.path.dirname(__file__), "..", "results")
EVERY = 250
NCERT = 700


def run(cfg):
    frac, seed, wd, steps = cfg
    X, Y, tr, te, rng = E.data(frac, seed)
    net = MLP([2 * E.P, 128, 128, E.P], rng)
    sub = np.random.default_rng(1000 + seed).choice(te, size=min(NCERT, len(te)), replace=False)
    hist = []
    t0 = time.time()
    for s in range(steps + 1):
        if s % EVERY == 0:
            logits, reps = net.forward(X)
            F = reps[-1]
            st = checkpoint_stats(F[tr], Y[tr], F[sub], Y[sub], net.W[-1], net.b[-1], logits[tr], logits[sub], E.P, per_point=(s % 500 == 0))
            st["test_acc_full"] = float((logits[te].argmax(1) == Y[te]).mean())
            st["test_loss_full"] = float(np.mean(-np.log(np.exp(logits[te] - logits[te].max(1, keepdims=True))[np.arange(len(te)), Y[te]]
                                                 / np.exp(logits[te] - logits[te].max(1, keepdims=True)).sum(1))))
            st["weight_norm"] = float(np.sqrt(sum((p ** 2).sum() for p in net.params)))
            st["step"] = s
            hist.append(st)
            if s % 2000 == 0:
                print(cfg, s, round(st["test_acc_full"], 3), round(st["C"], 3), round(st["C_vis"], 3),
                      round(st["M_q50"], 3), round(st["NC1_test"], 3), round(time.time() - t0), flush=True)
        if s < steps:
            net.step(X[tr], Y[tr], lr=1e-3, wd=wd)
    res = {"arch": "mlp", "frac": frac, "seed": seed, "wd": wd, "steps": steps, "n_cert": len(sub), "history": hist}
    with open(os.path.join(RES, f"controls_mlp_f{frac}_s{seed}_wd{wd}.json"), "w") as f:
        json.dump(res, f)
    return cfg


if __name__ == "__main__":
    cfgs = [(0.5, s, 1.0, 12000) for s in range(5)]
    cfgs += [(0.6, s, 1.0, 12000) for s in range(3)]
    cfgs += [(0.7, s, 1.0, 12000) for s in range(3)]
    cfgs += [(0.5, s, 3.0, 12000) for s in range(3)]
    cfgs += [(0.5, s, 0.3, 24000) for s in range(3)]
    if len(sys.argv) > 1:
        cfgs = [c for c in cfgs if f"wd{c[2]}" in sys.argv[1:] or f"f{c[0]}" in sys.argv[1:]]
    cfgs.sort(key=lambda c: -c[3])
    with Pool(4) as pool:
        for c in pool.imap_unordered(run, cfgs):
            print("done", c, flush=True)
