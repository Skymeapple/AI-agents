"""E1: utility/attack-success trade-off under a noisy out-of-band provenance flag.
Legit and injected in-data instructions have identical token/position distributions; they
differ only in the provenance flag, which is correct with prob q.  Hence TV(view_L, view_A)=2q-1
exactly, and Theorem 1 predicts ASR >= U - (2q-1) for every model."""
import json, sys, time
import numpy as np, torch
from multiprocessing import Pool
import toy

QS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
WS = [0.25, 1.0, 4.0]          # loss weight on DEL_A (sets the operating point)
SEEDS = [0, 1]
STEPS = 1200
TEMPL = [[]]                   # bare injected op: identical content distribution to legit


def run(args):
    q, w, seed = args
    torch.set_num_threads(1)
    m = toy.train_model({"CLEAN": 1, "INJ": 1, "DEL_L": 1, "DEL_A": 1}, steps=STEPS, seed=seed, q=q,
                        weights={"DEL_A": w}, templates=TEMPL)
    rng = np.random.default_rng(10_000 + seed)
    n = 6000
    res = {"q": q, "w": w, "seed": seed, "delta": 2 * q - 1}
    x, r, y, meta = toy.make_batch(rng, n, "DEL_L", q=q, templates=TEMPL)
    p = toy.predict(m, x, r).argmax(1)
    res["U"] = float((p == y).mean())
    x, r, y, meta = toy.make_batch(rng, n, "DEL_A", q=q, templates=TEMPL)
    p = toy.predict(m, x, r).argmax(1)
    res["ASR"] = float((p == meta["table"][np.arange(n), meta["op_att"]]).mean())
    res["noop_on_attack"] = float((p == toy.NOOP_CLASS).mean())
    for k in ["CLEAN", "INJ"]:
        x, r, y, meta = toy.make_batch(rng, n, k, q=q, templates=TEMPL)
        p = toy.predict(m, x, r).argmax(1)
        res["acc_" + k] = float((p == y).mean())
    res["bound_ASR_min"] = res["U"] - res["delta"]
    res["n_eval"] = n
    print(json.dumps(res), flush=True)
    return res


if __name__ == "__main__":
    t = time.time()
    jobs = [(q, w, s) for q in QS for w in WS for s in SEEDS]
    with Pool(4) as pool:
        out = pool.map(run, jobs)
    json.dump(out, open("../results/e1_tradeoff.json", "w"), indent=1)
    print("done", time.time() - t)
