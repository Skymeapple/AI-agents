"""FabNER on CPU: a classical baseline and an estimate of how consistent the labels are.

Experiment for HFM's "FabNER Material Manufacturing Entity Extraction" track, where
language models are fine-tuned to about 81 entity-level F1 and the best published
figure found is about 85. HFM's work needs a GPU. Two questions can be answered
without one, on the public dataset (DFKI-SLT/fabner on Hugging Face; Kumar and
Starly 2021):

  1. How far does a classical conditional random field get? This shows how much of
     the score is surface pattern and how much needs a large model.
  2. How consistent are the labels? If the same phrase is labelled one way in some
     sentences and another way (or not at all) in others, no model can score 100.
     The share of test entities affected bounds what any model can reach.

Score: entity-level micro F1, exact span and type match (the usual CoNLL scoring).

Usage: python fabner_ceiling.py <train.parquet> <validation.parquet> <test.parquet> <labels.json>
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict

import pandas as pd
import sklearn_crfsuite


def spans(tags):
    """Entity spans (start, end, type) from BIO or BIOES tags."""
    out, start, typ = [], None, None
    for i, t in enumerate(list(tags) + ["O"]):
        pre, _, ty = t.partition("-")
        if start is not None and (pre in ("O", "B", "S") or ty != typ):
            out.append((start, i, typ))
            start = None
        if pre in ("B", "S") or (pre in ("I", "E") and start is None):
            start, typ = i, ty
    return out


def f1(gold, pred):
    tp = fp = fn = 0
    for g, p in zip(gold, pred):
        g, p = set(spans(g)), set(spans(p))
        tp += len(g & p); fp += len(p - g); fn += len(g - p)
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return 100 * 2 * pr * rc / max(pr + rc, 1e-9), 100 * pr, 100 * rc


def shape(w):
    return "".join("A" if c.isupper() else "a" if c.islower() else "0" if c.isdigit() else "-" for c in w)[:8]


def feats(sent):
    rows = []
    for i, w in enumerate(sent):
        f = {"w": w.lower(), "s3": w[-3:].lower(), "s2": w[-2:].lower(), "p3": w[:3].lower(), "shape": shape(w), "up": w.isupper(), "title": w.istitle(), "digit": any(c.isdigit() for c in w), "hyph": "-" in w}
        for d in (-2, -1, 1, 2):
            j = i + d
            f[f"w{d}"] = sent[j].lower() if 0 <= j < len(sent) else "<pad>"
        if i > 0:
            f["bi-"] = sent[i - 1].lower() + "_" + w.lower()
        if i + 1 < len(sent):
            f["bi+"] = w.lower() + "_" + sent[i + 1].lower()
        rows.append(f)
    return rows


def main():
    names = json.load(open(sys.argv[4]))
    load = lambda p: [(list(map(str, r.tokens)), [names[t] for t in r.ner_tags]) for r in pd.read_parquet(p).itertuples()]
    tr, va, te = load(sys.argv[1]), load(sys.argv[2]), load(sys.argv[3])
    n_ent = lambda d: sum(len(spans(t)) for _, t in d)
    print(f"sentences: train {len(tr)}, validation {len(va)}, test {len(te)}; entities: {n_ent(tr)}, {n_ent(va)}, {n_ent(te)}; types: {sorted({t.split('-')[1] for t in names if '-' in t})}")

    # ---------- 1. classical baselines ----------
    # dictionary: label each phrase seen as an entity in training with its most common type (longest match first)
    seen = defaultdict(Counter)
    for s, t in tr:
        for a, b, ty in spans(t):
            seen[tuple(x.lower() for x in s[a:b])][ty] += 1
    maxlen = max(len(k) for k in seen)

    def dict_tag(s):
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

    gold = [t for _, t in te]
    print("\n1. Entity-level micro F1 on the test set (precision, recall)")
    print("   dictionary of training entities:  %.1f  (%.1f, %.1f)" % f1(gold, [dict_tag(s) for s, _ in te]))
    best = None
    for c2 in (0.01, 0.05, 0.2):
        crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.05, c2=c2, max_iterations=150, all_possible_transitions=True)
        crf.fit([feats(s) for s, _ in tr], [t for _, t in tr])
        v = f1([t for _, t in va], crf.predict([feats(s) for s, _ in va]))[0]
        if best is None or v > best[0]:
            best = (v, c2, crf)
    pred = best[2].predict([feats(s) for s, _ in te])
    print("   conditional random field:         %.1f  (%.1f, %.1f)   [setting chosen on validation, F1 there %.1f]" % (*f1(gold, pred), best[0]))
    print("   HFM's fine-tuned language model:  80.7 (HFM's own evaluation; shown for scale, not directly comparable)")

    # ---------- 2. label consistency ----------
    # for each phrase that is an entity somewhere in training, count how it is labelled every time it appears
    occ = defaultdict(Counter)                        # phrase -> Counter of labels over all appearances in train (type, or "none"/"part")
    index = defaultdict(list)
    for k in seen:
        index[k[0]].append(k)
    for s, t in tr:
        low = [x.lower() for x in s]
        sp = {(a, b): ty for a, b, ty in spans(t)}
        inside = [False] * len(s)
        for (a, b) in sp:
            for i in range(a, b):
                inside[i] = True
        for i, w in enumerate(low):
            for k in index.get(w, ()):
                n = len(k)
                if tuple(low[i:i + n]) == k:
                    if (i, i + n) in sp:
                        occ[k][sp[(i, i + n)]] += 1
                    elif any(inside[i:i + n]):
                        occ[k]["part of another span"] += 1
                    else:
                        occ[k]["not labelled"] += 1
    multi = {k: c for k, c in occ.items() if sum(c.values()) >= 5}
    tot = sum(sum(c.values()) for c in multi.values())
    agree = sum(c.most_common(1)[0][1] for c in multi.values())
    kinds = Counter()
    for c in multi.values():
        top = c.most_common(1)[0][0]
        for lab, n in c.items():
            if lab != top:
                kinds["labelled, majority unlabelled" if top in ("not labelled", "part of another span") else ("unlabelled or inside a longer span" if lab in ("not labelled", "part of another span") else "a different type")] += n
    print(f"\n2. Label consistency in the training set, phrases that are an entity at least once and appear 5+ times ({len(multi)} phrases, {tot} appearances)")
    print(f"   appearances labelled like the phrase's most common label: {100 * agree / tot:.1f}%")
    for k_, n in kinds.most_common():
        print(f"   disagreeing, {k_}: {100 * n / tot:.1f}%")
    # how does that play out on the test set: an oracle that knows each test phrase's majority training label
    hit = miss = unseen = 0
    for s, t in te:
        for a, b, ty in spans(t):
            k = tuple(x.lower() for x in s[a:b])
            if k not in occ:
                unseen += 1
            elif occ[k].most_common(1)[0][0] == ty:
                hit += 1
            else:
                miss += 1
    n = hit + miss + unseen
    print(f"\n   test entities: {n}. Phrase seen in training with the same majority label {100 * hit / n:.1f}%, seen but majority label differs {100 * miss / n:.1f}%, never seen as an entity {100 * unseen / n:.1f}%")
    # errors of the CRF split by whether the phrase is consistently labelled in training
    cons = {k for k, c in occ.items() if sum(c.values()) >= 3 and c.most_common(1)[0][1] / sum(c.values()) >= 0.9}
    incons = {k for k, c in occ.items() if sum(c.values()) >= 3 and c.most_common(1)[0][1] / sum(c.values()) < 0.7}
    stat = {"consistent phrases (90%+ agreement)": [0, 0], "inconsistent phrases (under 70%)": [0, 0], "phrases never seen as an entity": [0, 0]}
    for (s, t), p in zip(te, pred):
        ps = set(spans(p))
        for a, b, ty in spans(t):
            k = tuple(x.lower() for x in s[a:b])
            g = "phrases never seen as an entity" if k not in occ else "consistent phrases (90%+ agreement)" if k in cons else "inconsistent phrases (under 70%)" if k in incons else None
            if g:
                stat[g][0] += (a, b, ty) in ps
                stat[g][1] += 1
    print("\n   share of test entities the CRF finds exactly (recall), by how the phrase is labelled in training")
    for g, (h, m) in stat.items():
        print(f"   {g:<40} {100 * h / max(m, 1):.1f}%  of {m}")


if __name__ == "__main__":
    main()
