"""Shapley decomposition of the value of hardening inputs (uses results/E3.csv)."""
import csv, itertools, json, math, os
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
rows = list(csv.DictReader(open(os.path.join(RES, "E3.csv"))))
P = ["logic", "cowos", "hbm"]
out = {}
for mode in ("priority", "proportional"):
    for metric in ("mean", "CVaR99"):
        v = {}
        for r in rows:
            if r["mode"] != mode: continue
            key = frozenset([] if r["immune"] == "none" else r["immune"].split("+"))
            v[key] = float(r[metric])
        base = v[frozenset()]
        val = lambda S: base - v[frozenset(S)]   # risk reduction from hardening S
        sh = {}
        for p in P:
            tot = 0.0
            others = [q for q in P if q != p]
            for k in range(len(others) + 1):
                for S in itertools.combinations(others, k):
                    w = math.factorial(k) * math.factorial(len(P) - k - 1) / math.factorial(len(P))
                    tot += w * (val(set(S) | {p}) - val(S))
            sh[p] = tot / base * 100
        # interaction: v(logic+cowos) - v(logic) - v(cowos)
        inter = (val({"logic", "cowos"}) - val({"logic"}) - val({"cowos"})) / base * 100
        out[f"{mode}_{metric}"] = dict(shapley_pct=sh, logic_cowos_interaction_pct=inter)
        print(mode, metric, {k: round(x, 1) for k, x in sh.items()}, "interaction(logic,cowos)=%.1f%%" % inter)
json.dump(out, open(os.path.join(RES, "shapley.json"), "w"), indent=1)
