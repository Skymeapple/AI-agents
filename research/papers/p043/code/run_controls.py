"""Test of Proposition 3 in trained transformers: adding cue-without-lie negatives
('controls') to supervised probe training. Also a probe trained on two triggers
(A+B) and tested on the third (C). Loads models saved by run_main.py.

usage: python run_controls.py --variant dep --seeds 0 1 2 3 4 5
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from world import World
from model import get_acts
from run_obfuscation import load
import detectors as D

RES = os.path.join(os.path.dirname(__file__), "..", "results")
TEST = ["lieA_instructed", "lieB_incentive", "lieC_sleeper", "cue_no_lie", "adv_no_secret"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--variant", default="dep")
    a = ap.parse_args()
    out = []
    for seed in a.seeds:
        w = World(seed, truth_independent_A=(a.variant == "indep"))
        p = load(a.variant, seed)
        rng = np.random.default_rng(seed + 7)
        fit_e = w.probe_train_ents[:64]; te = w.probe_test_ents

        def acts(c, ents, reps=4):
            X, Y, T, E = w.eval_set(c, ents, rng, reps)
            return get_acts(p, X)[1]
        Hh = acts("honest", fit_e); Ha = acts("lieA_instructed", fit_e)
        Hc = acts("cue_no_lie", fit_e, reps=16)     # few protected entities -> more nuisance reps
        Hn = acts("adv_no_secret", fit_e)
        Hb = acts("lieB_incentive", fit_e)
        Ht = acts("honest", te)
        T = {c: acts(c, te) for c in TEST}
        L1 = Hh.shape[1]
        r = {"seed": seed, "LR_plain": {c: [] for c in TEST}, "LR_controls": {c: [] for c in TEST},
             "LR_AB_plus_controls": {c: [] for c in TEST}}
        for l in range(L1):
            pr0 = D.fit_lr_probe(Hh[:, l], Ha[:, l])
            pr1 = D.fit_lr_probe(np.concatenate([Hh[:, l], Hc[:, l]]), Ha[:, l])
            pr2 = D.fit_lr_probe(np.concatenate([Hh[:, l], Hc[:, l], Hn[:, l]]),
                                 np.concatenate([Ha[:, l], Hb[:, l]]))
            for name, pr in [("LR_plain", pr0), ("LR_controls", pr1), ("LR_AB_plus_controls", pr2)]:
                sh = D.lr_score(pr, Ht[:, l])
                for c in TEST:
                    r[name][c].append(D.auroc(sh, D.lr_score(pr, T[c][:, l])))
        out.append(r)
        print(seed, {k: {c: round(float(np.max(v[c]) if c.startswith("lie") else np.min(v[c])), 3)
                         for c in TEST} for k, v in r.items() if k != "seed"}, flush=True)
    json.dump(out, open(os.path.join(RES, f"controls_{a.variant}.json"), "w"), indent=1)
