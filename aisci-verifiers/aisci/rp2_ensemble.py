"""Task 1.4 / RP2 simulation: can stacked weak judges approximate a strong verifier?

SYNTHETIC. Each hypothesis has a hidden truth and two hidden traits: how
'mainstream' it is and whether it falls in the simulator's domain of validity.
Judges err in the structured ways real ones do:
  - simulation: sharp inside its validity domain, near-random outside it
  - literature: biased toward mainstream claims, true or not
  - critic: a learned model with broad, moderate noise
The ensemble is fitted on 'historical' cases and tested on later held-out cases.
This shows the method works under stated assumptions; it is not evidence about
real biology judges, which RP2 must collect (2,000 hypotheses with lab outcomes).
"""
from __future__ import annotations

import json

import numpy as np

from .verifiers import EnsembleVerifier, SoftJudge, wilson


def make_cases(n, rng, base_rate=0.3):
    truth = rng.random(n) < base_rate
    mainstream = rng.random(n)
    in_domain = rng.random(n) < 0.6
    return [dict(id=i, truth=bool(t), mainstream=float(m), in_domain=bool(d),
                 noise=rng.normal(size=3), shared=float(rng.normal()))
            for i, (t, m, d) in enumerate(zip(truth, mainstream, in_domain))]


CORR = 0.0  # weight of an error shared by all judges (a field-wide misconception)


def sig(z):
    return 1 / (1 + np.exp(-z))


def sim_score(c):
    sharp = 2.5 if c["in_domain"] else 0.15
    return float(sig(sharp * (1 if c["truth"] else -1) + 0.8 * c["noise"][0] + CORR * c["shared"]))


def lit_score(c):
    return float(sig(0.9 * (1 if c["truth"] else -1) + 2.2 * (c["mainstream"] - 0.5) + 0.8 * c["noise"][1] + CORR * c["shared"]))


def critic_score(c):
    return float(sig(1.1 * (1 if c["truth"] else -1) + 1.0 * c["noise"][2] + CORR * c["shared"]))


def rates(judge_fn, cases):
    wrong = [c for c in cases if not c["truth"]]
    right = [c for c in cases if c["truth"]]
    fp = sum(judge_fn(c) for c in wrong)
    fr = sum(not judge_fn(c) for c in right)
    return {"false_pass_rate": fp / len(wrong), "false_pass_95ci": wilson(fp, len(wrong)),
            "false_reject_rate": fr / len(right), "n_wrong": len(wrong)}


def run(seed=0, n_train=2000, n_test=2000):
    rng = np.random.default_rng(seed)
    train, test = make_cases(n_train, rng), make_cases(n_test, rng)
    judges = [SoftJudge("simulation", sim_score), SoftJudge("literature", lit_score),
              SoftJudge("critic", critic_score)]
    ens = EnsembleVerifier(judges).fit(train, [c["truth"] for c in train])

    # Compare at a matched false-reject budget: pick each judge's threshold on TRAIN
    # so that it rejects at most 30% of true hypotheses, then evaluate on TEST.
    def thresholded(score_fn):
        s_true = np.array([score_fn(c) for c in train if c["truth"]])
        t = float(np.quantile(s_true, 0.30))
        return (lambda c: score_fn(c) >= t), t

    out = {}
    for name, fn in [("simulation", sim_score), ("literature", lit_score),
                     ("critic", critic_score), ("ensemble", ens.prob)]:
        judge, t = thresholded(fn)
        out[name] = {"threshold": t, **rates(judge, test)}
    single = min(out[k]["false_pass_rate"] for k in ("simulation", "literature", "critic"))
    out["summary"] = {
        "best_single_false_pass": single,
        "ensemble_false_pass": out["ensemble"]["false_pass_rate"],
        "relative_reduction": 1 - out["ensemble"]["false_pass_rate"] / single,
        "hypothesis_halved": out["ensemble"]["false_pass_rate"] <= single / 2,
        "kill_criterion_triggered(<15%)": (1 - out["ensemble"]["false_pass_rate"] / single) < 0.15,
        "weights": dict(zip(["bias", "simulation", "literature", "critic"], ens.w.round(3).tolist())),
    }
    return out


def sweep(corrs=(0.0, 1.0, 2.0, 3.0), seeds=range(5)):
    global CORR
    out = {}
    for c in corrs:
        CORR = c
        red = [run(seed=s)["summary"]["relative_reduction"] for s in seeds]
        out[f"shared_error_{c}"] = {"mean_relative_reduction": float(np.mean(red)),
                                    "min": float(np.min(red)), "max": float(np.max(red))}
    CORR = 0.0
    return out


if __name__ == "__main__":
    print(json.dumps({"detail_independent_seed_0": run(0), "correlation_sweep": sweep()}, indent=2))
