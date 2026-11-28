"""Recompute the Lean tables from audited proofs only (output of audit.py).

Usage: python audit_summary.py <audit.json ...>
"""
import collections
import json
import sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import lean_floor as lf


def main():
    rows = sum((json.load(open(f)) for f in sys.argv[1:]), [])
    print(f"{len(rows)} candidates; {sum(r['ok'] for r in rows)} pass the strict check")
    print("failure reasons:", collections.Counter(r["why"].split(" at line")[0].split("[")[0] for r in rows if not r["ok"]).most_common())
    tac = {sp: collections.defaultdict(set) for sp in ("valid", "test")}
    claimed = {sp: collections.defaultdict(set) for sp in ("valid", "test")}
    model = collections.defaultdict(set)
    model_claimed = collections.defaultdict(set)
    for r in rows:
        if r["label"].startswith("model"):
            model_claimed[r["source"]].add(r["name"])
            if r["ok"]:
                model[r["source"]].add(r["name"])
        else:
            claimed[r["split"]][r["label"]].add(r["name"])
            if r["ok"]:
                tac[r["split"]][r["label"]].add(r["name"])
    for sp in ("valid", "test"):
        a = set().union(*claimed[sp].values())
        b = set().union(*tac[sp].values())
        print(f"{sp}: any tactic, claimed {len(a)}, audited {len(b)}; lost: {sorted(a - b)}")
    # greedy cover on validation, scored on test
    order = [t for t in lf.TACTICS if t in tac["valid"] or t in tac["test"]]
    chosen, cov = [], set()
    while True:
        best = max(order, key=lambda t: len(tac["valid"][t] - cov))
        if len(tac["valid"][best] - cov) == 0:
            break
        chosen.append(best)
        cov |= tac["valid"][best]
        tcov = set().union(*(tac["test"][t] for t in chosen))
        print(f"greedy {len(chosen):2d}: + {best[:60]:60s} validation {len(cov):3d} ({len(cov) / 244:.1%}), test {len(tcov):3d} ({len(tcov) / 244:.1%})")
    for t in ("grind", "norm_num at * <;> grind"):
        print(f"single {t}: validation {len(tac['valid'][t])}, test {len(tac['test'][t])}")
    r12 = [t for t in lf.TACTICS[:lf.ROUND3]]
    print(f"any of rounds 1-2 on test: {len(set().union(*(tac['test'][t] for t in r12)))}")
    any_test = set().union(*tac["test"].values())
    chosen_test = set().union(*(tac["test"][t] for t in chosen))
    print(f"any of the 52 on test: {len(any_test)} ({len(any_test) / 244:.1%}); validation-chosen {len(chosen)} on test: {len(chosen_test)} ({len(chosen_test) / 244:.1%})")
    total = set(any_test)
    for rnd, prefix in (("model, 4 samples", "lean-prover-llm-"), ("model, 8 more", "lean-prover-llm2-"), ("repair round 3", "lean-prover-repair-"),
                        ("repair round 4", "lean-prover-repair2-"), ("repair round 5", "lean-prover-repair3-")):
        got = set().union(*[v for k, v in model.items() if k.startswith(prefix) and k[len(prefix):].isdigit()] or [set()])
        cl = set().union(*[v for k, v in model_claimed.items() if k.startswith(prefix) and k[len(prefix):].isdigit()] or [set()])
        new = got - total
        total |= got
        print(f"{rnd}: claimed {len(cl)}, audited {len(got)} (lost {sorted(cl - got)}); new {len(new)}; running total {len(total)} ({len(total) / 244:.4f})")
    honest = chosen_test | (total - any_test)
    print(f"RESULT total on test: {len(total)} of 244 = {len(total) / 244:.4f}; with validation-chosen tactics: {len(honest)} = {len(honest) / 244:.4f}")


if __name__ == "__main__":
    main()
