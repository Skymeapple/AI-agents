"""Fraction of stored successful adversarial spans (5 per system/goal/k/seed) containing a forged in-band
marker SIG(7) immediately followed by an op token (8-11), and containing any structural token."""
import json
e3 = json.load(open("../results/e3_adaptive.json"))
SIG, OPS, STRUCT = 7, {8, 9, 10, 11}, {1, 2, 3, 4, 5, 6, 7}
out = {}
for s in ["UNDEF", "INBAND", "OOB", "GATED"]:
    n = sig = st = 0
    for r in e3:
        if r["system"] != s: continue
        for k, v in r.items():
            if k.startswith("examples_") and not k.endswith("k1"):
                for ex in v:
                    n += 1
                    sig += any(ex[i] == SIG and ex[i + 1] in OPS for i in range(len(ex) - 1))
                    st += any(t in STRUCT for t in ex)
    out[s] = dict(n_examples=n, frac_SIG_then_op=sig / n if n else None, frac_any_structural=st / n if n else None)
print(json.dumps(out, indent=1))
json.dump(out, open("../results/e3_forgery_stats.json", "w"), indent=1)
