"""FabNER on CPU: can a classical model that is shown the dictionary's answer beat the dictionary?

The dictionary of training entities scores 83.4 F1 on the public test split and a
conditional random field 82.8. They fail differently: the dictionary cannot find a
phrase it has not seen, the CRF loses recall on seen ones. This script stacks them:
the CRF receives the dictionary's tag for each word as a feature. For training
sentences the dictionary is built from the other nine tenths of the training set
(cross-fitting), so the CRF sees the dictionary as fallible as it is at test time.

Also reports the simple union: dictionary output plus CRF entities that do not overlap it.

Usage: python fabner_stack.py <train.parquet> <validation.parquet> <test.parquet> <labels.json>
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

import pandas as pd
import sklearn_crfsuite

from fabner_ceiling import spans, f1, feats


def build(data):
    seen = defaultdict(Counter)
    for s, t in data:
        for a, b, ty in spans(t):
            seen[tuple(x.lower() for x in s[a:b])][ty] += 1
    return seen, max(len(k) for k in seen)


def dict_tag(s, seen, maxlen):
    low, out, i = [x.lower() for x in s], ["O"] * len(s), 0
    while i < len(s):
        for n in range(min(maxlen, len(s) - i), 0, -1):
            k = tuple(low[i:i + n])
            if k in seen:
                ty = seen[k].most_common(1)[0][0]
                out[i:i + n] = ["B-" + ty] + ["I-" + ty] * (n - 1)
                i += n - 1
                break
        i += 1
    return out


def stack_feats(s, d):
    rows = feats(s)
    for i, f in enumerate(rows):
        f["d"] = d[i]
        f["d-1"] = d[i - 1] if i else "<pad>"
        f["d+1"] = d[i + 1] if i + 1 < len(s) else "<pad>"
        f["d|w"] = d[i] + "|" + s[i].lower()
        f["s4"] = s[i][-4:].lower()
        f["p4"] = s[i][:4].lower()
    return rows


def union(d, c):
    out = list(d)
    for a, b, ty in spans(c):
        if all(x == "O" for x in out[a:b]):
            out[a:b] = ["B-" + ty] + ["I-" + ty] * (b - a - 1)
    return out


def main():
    names = json.load(open(sys.argv[4]))
    load = lambda p: [(list(map(str, r.tokens)), [names[t] for t in r.ner_tags]) for r in pd.read_parquet(p).itertuples()]
    tr, va, te = load(sys.argv[1]), load(sys.argv[2]), load(sys.argv[3])
    full, ml = build(tr)
    K = 10
    dtr = [None] * len(tr)
    for k in range(K):
        sk, mk = build([x for i, x in enumerate(tr) if i % K != k])
        for i in range(k, len(tr), K):
            dtr[i] = dict_tag(tr[i][0], sk, mk)
    dva = [dict_tag(s, full, ml) for s, _ in va]
    dte = [dict_tag(s, full, ml) for s, _ in te]
    gold = [t for _, t in te]
    print("held-out dictionary F1 on training folds: %.1f (%.1f, %.1f)" % f1([t for _, t in tr], dtr))

    def fit(X, Xva, Xte, label):
        best = None
        for c2 in (0.01, 0.05, 0.2):
            crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.05, c2=c2, max_iterations=150, all_possible_transitions=True)
            crf.fit(X, [t for _, t in tr])
            v = f1([t for _, t in va], crf.predict(Xva))[0]
            if best is None or v > best[0]:
                best = (v, c2, crf)
        pred = best[2].predict(Xte)
        print("   %-46s %.1f  (%.1f, %.1f)   [validation %.1f]" % (label, *f1(gold, pred), best[0]), flush=True)
        return pred

    print("\nEntity-level micro F1 on the test set (precision, recall)")
    print("   %-46s %.1f  (%.1f, %.1f)" % ("dictionary", *f1(gold, dte)))
    plain = fit([feats(s) for s, _ in tr], [feats(s) for s, _ in va], [feats(s) for s, _ in te], "CRF")
    print("   %-46s %.1f  (%.1f, %.1f)" % ("union: dictionary + non-overlapping CRF", *f1(gold, [union(d, c) for d, c in zip(dte, plain)])))
    stacked = fit([stack_feats(s, d) for (s, _), d in zip(tr, dtr)], [stack_feats(s, d) for (s, _), d in zip(va, dva)], [stack_feats(s, d) for (s, _), d in zip(te, dte)], "CRF given the dictionary's tags (cross-fitted)")

    print("\nRecall by whether the phrase is an entity anywhere in training")
    for label, pred in (("dictionary", dte), ("CRF", plain), ("stacked CRF", stacked)):
        st = {True: [0, 0], False: [0, 0]}
        for (s, t), p in zip(te, pred):
            ps = set(spans(p))
            for a, b, ty in spans(t):
                g = tuple(x.lower() for x in s[a:b]) in full
                st[g][0] += (a, b, ty) in ps
                st[g][1] += 1
        print("   %-12s seen %.1f%% of %d, unseen %.1f%% of %d" % (label, 100 * st[True][0] / st[True][1], st[True][1], 100 * st[False][0] / st[False][1], st[False][1]))
    # paired bootstrap over sentences: stacked against dictionary
    import random
    rnd = random.Random(0)
    idx = list(range(len(te)))
    wins = 0
    diffs = []
    for _ in range(1000):
        b = [rnd.choice(idx) for _ in idx]
        d = f1([gold[i] for i in b], [stacked[i] for i in b])[0] - f1([gold[i] for i in b], [dte[i] for i in b])[0]
        diffs.append(d); wins += d > 0
    diffs.sort()
    print("\nstacked minus dictionary, 1000 bootstrap resamples of test sentences: median %+.2f, 95%% interval %+.2f to %+.2f, better in %.1f%%" % (diffs[500], diffs[25], diffs[975], wins / 10))


if __name__ == "__main__":
    main()
