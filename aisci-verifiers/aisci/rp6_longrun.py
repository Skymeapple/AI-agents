"""RP6 simulation: do errors compound over long agent runs, and do audits stop them?

SYNTHETIC agent on top of the real WorldModel (provenance and retraction logic
are the production code). Base error rates per claim type are taken from the
Kosmos evaluation (data 85.5%, literature 82.1%, synthesis 57.9% accurate).
Derived claims inherit falsehood from false parents with fixed probabilities;
those inheritance rates are assumptions, and RP6 must measure them on real runs.

Design (matches RP6 in the study): 15 matched seeds x 2 arms (audit / no audit),
200 cycles, 10 false claims seeded at cycle 20.
"""
from __future__ import annotations

import json
import random

import numpy as np

from .worldmodel import Claim, Evidence, WorldModel

ERR = {"data": 0.145, "literature": 0.179, "synthesis": 0.421}
MIX = [("data", 0.4), ("literature", 0.3), ("synthesis", 0.3)]
INHERIT = {"synthesis": 0.8, "data": 0.5, "literature": 0.5}


def simulate(seed, cycles=200, per_cycle=5, audit_every=0, audit_size=0,
             detect=0.8, false_alarm=0.02, seed_cycle=20, n_seeded=10):
    rng = random.Random(seed)
    wm = WorldModel()
    truth: dict[str, bool] = {}
    seeded: dict[str, int | None] = {}
    gen = audits = 0
    acc_curve = []
    n = 0

    def live():
        return [k for k, c in wm.claims.items() if c.status == "active"]

    for t in range(1, cycles + 1):
        for _ in range(per_cycle):
            r, acc_p, ctype = rng.random(), 0.0, "data"
            for name, p in MIX:
                acc_p += p
                if r <= acc_p:
                    ctype = name
                    break
            pool = live()
            k = 2 if ctype == "synthesis" else (1 if rng.random() < 0.5 else 0)
            if len(pool) < k:
                ctype, k = "data", 0
            parents = rng.sample(pool, k) if k else []
            false = rng.random() < ERR[ctype]
            if any(not truth[p] for p in parents) and rng.random() < INHERIT[ctype]:
                false = True
            n += 1
            cid = f"c{n}"
            wm.add(Claim(cid, f"claim {n}", ctype, 0.7, [Evidence("code", f"nb/{cid}.ipynb")],
                         parents, t))
            truth[cid] = not false
            gen += 1
        if t == seed_cycle:
            for i in range(n_seeded):
                n += 1
                cid = f"seed{i}"
                wm.add(Claim(cid, "seeded false claim", "data", 0.95,
                             [Evidence("data", f"seeded/{i}")], [], t))
                truth[cid] = False
                seeded[cid] = None
        if audit_every and t % audit_every == 0:
            sus = [k for k, c in wm.claims.items() if c.status == "suspect"]
            pool = live()
            rng.shuffle(pool)
            batch = (sus + pool)[:audit_size]
            for cid in batch:
                audits += 1
                bad = (not truth[cid] and rng.random() < detect) or (truth[cid] and rng.random() < false_alarm)
                if bad:
                    wm.retract(cid, "audit failed", t)
                    if cid in seeded and seeded[cid] is None:
                        seeded[cid] = t
                elif wm.claims[cid].status == "suspect":
                    wm.claims[cid].status = "active"  # re-verified
        act = live()
        acc_curve.append(sum(truth[k] for k in act) / len(act) if act else 1.0)

    xs = np.arange(10, cycles + 1)
    slope = float(np.polyfit(xs, np.array(acc_curve[9:]), 1)[0] * 100)
    caught = [v - seed_cycle for v in seeded.values() if v is not None]
    return {
        "acc_c10": acc_curve[9], "acc_c50": acc_curve[49], "acc_c200": acc_curve[-1],
        "slope_per_100_cycles": slope,
        "seeded_caught": len(caught), "median_cycles_to_catch": float(np.median(caught)) if caught else None,
        "audit_overhead": audits / gen, "traceability": wm.traceability(),
    }


def arm(seeds, **kw):
    rs = [simulate(s, **kw) for s in seeds]
    keys = ["acc_c10", "acc_c50", "acc_c200", "slope_per_100_cycles", "seeded_caught", "audit_overhead", "traceability"]
    out = {k: float(np.mean([r[k] for r in rs])) for k in keys}
    m = [r["median_cycles_to_catch"] for r in rs if r["median_cycles_to_catch"] is not None]
    out["median_cycles_to_catch"] = float(np.median(m)) if m else None
    return out


def run():
    seeds = range(15)
    res = {"no_audit": arm(seeds)}
    for every, size in [(5, 5), (5, 10), (2, 10), (1, 10)]:
        res[f"audit_every{every}_size{size}"] = arm(seeds, audit_every=every, audit_size=size)
    base = res["no_audit"]["acc_c200"]
    for k, v in res.items():
        if k == "no_audit":
            continue
        gain = (v["acc_c200"] - base) * 100
        v["points_retained_at_c200"] = gain
        v["kill(overhead>40% & gain<10pts)"] = v["audit_overhead"] > 0.40 and gain < 10
        v["hypothesis_flat(|slope|<2pts/100)"] = abs(v["slope_per_100_cycles"]) < 2
    return res


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
