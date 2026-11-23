"""Experiment 0b (added in revision): clean ablation of lambda annealing.
Identical trainer (train_anneal: 3000 steps, batch 4096, Adam lr 1e-2) with either the
geometric 1 -> lambda schedule over 70% of training ('anneal') or lambda fixed from
step 1 ('fixed', warm -> 0). GaussianShift d'=1.5, linear and width-16 policies.
Run: OMP_NUM_THREADS=1 python3 exp0b_ablation.py   (~2 min on 4 cores)
"""
import json, os
from multiprocessing import Pool
import numpy as np
from common import GaussianShift, train_anneal, RESULTS

LAMS = [4, 16, 32, 64]
SEEDS = [0, 1, 2]


def job(args):
    h, lam, s, sched = args
    env = GaussianShift(8, 1.5, 0)
    rng = np.random.default_rng(10007 * s + 31 * lam + h)   # same seed for both schedules
    m = train_anneal(env, h, lam, rng, steps=3000, batch=4096, lr=1e-2,
                     warm=0.7 if sched == "anneal" else 1e-9)
    r = np.random.default_rng(424242 + s)
    a = float(m.prob(env.sample(400000, "E", r)).mean())
    b = float(m.prob(env.sample(400000, "D", r)).mean())
    R = float(env.bayes_roc(max(a, 1e-9)))
    return dict(width=h, lam=lam, seed=s, schedule=sched, alpha=a, beta=b, frontier=R,
                efficiency=b / R if R > 0 else None)


def main():
    jobs = [(h, l, s, sc) for h in [0, 16] for l in LAMS for s in SEEDS for sc in ["anneal", "fixed"]]
    with Pool(4) as p:
        pts = p.map(job, jobs)
    summ = {}
    for h in [0, 16]:
        for l in LAMS:
            for sc in ["anneal", "fixed"]:
                s = [q for q in pts if q["width"] == h and q["lam"] == l and q["schedule"] == sc]
                summ[f"w{h}_lam{l}_{sc}"] = dict(eff=[round(q["efficiency"], 3) for q in s],
                                                 alpha=[q["alpha"] for q in s], beta=[q["beta"] for q in s])
    json.dump(dict(points=pts, summary=summ), open(os.path.join(RESULTS, "exp0b_ablation.json"), "w"), indent=1)
    for k, v in summ.items():
        print(k, v["eff"], ["%.2g" % a for a in v["alpha"]])


if __name__ == "__main__":
    main()
