"""EmbSpatial-Bench without the image: how much is answerable from the options, and from object boxes?

HFM's track trains a 2B vision-language model to 0.785 accuracy on this benchmark
(3,640 four-choice questions, six spatial relations, three scene sources). That
needs a GPU. Two things can be measured on a CPU from the public records
(FlagEval/EmbSpatial-Bench on Hugging Face), which carry each question's options
and the names and bounding boxes of the objects it asks about:

  blind   answer from the question and options only (no image, no boxes)
  boxes   answer from the benchmark's own bounding boxes with plain geometry

"Blind" measures shortcuts in how the options were written. "Boxes" measures how far
a perfect 2D object detector plus rules would get, and so which relations need depth.
Learned variants are cross-validated with whole scenes held out.

Usage: python embspatial_no_image.py <embspatial.jsonl>
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

DIRS = {"left": ("left of", "left side"), "right": ("right of", "right side"), "above": ("above",), "under": ("under", "beneath", "below")}


def direction(option):
    for d, keys in DIRS.items():
        if any(k in option for k in keys):
            return d
    return None


def box(r, k):
    """Centre x, centre y, top, bottom, width, height as fractions of the image. ai2thor boxes are corner pairs, the others x, y, width, height."""
    a, b, c, d = r["objects"]["bbox"][k]
    if r["source"] == "ai2thor":
        x1, y1, x2, y2 = a, b, c, d
    else:
        x1, y1, x2, y2 = a, b, a + c, b + d
    W, H = r["width"], r["height"]
    return ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, y1 / H, y2 / H, (x2 - x1) / W, (y2 - y1) / H)


def cv_option_model(rows, feats, folds=5, seed=0):
    """Score each option with a logistic model; pick the highest. Whole scenes are held out."""
    scenes = sorted({r["scene"] for r in rows})
    random.Random(seed).shuffle(scenes)
    fold = {s: i % folds for i, s in enumerate(scenes)}
    pred = {}
    for k in range(folds):
        tr = [r for r in rows if fold[r["scene"]] != k]
        te = [r for r in rows if fold[r["scene"]] == k]
        X, y = [], []
        for r in tr:
            for i, f in enumerate(feats(r)):
                if f is not None:
                    X.append(f); y.append(int(i == r["answer"]))
        vec = DictVectorizer()
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(vec.fit_transform(X), y)
        for r in te:
            fs = feats(r)
            idx = [i for i, f in enumerate(fs) if f is not None]
            p = clf.predict_proba(vec.transform([fs[i] for i in idx]))[:, 1]
            pred[r["id"]] = idx[int(np.argmax(p))]
    return pred


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    pair = [r for r in rows if r["relation"] not in ("close", "far")]
    dist = [r for r in rows if r["relation"] in ("close", "far")]
    print(f"questions: {len(rows)}; two-object relation questions {len(pair)}, nearest/farthest questions {len(dist)}")

    # ---------- how the options were written ----------
    never = Counter(); total = Counter()
    for r in pair:
        for i, o in enumerate(r["options"]):
            kind = direction(o) or "other (touching, inside, blocking, connected, outside)"
            total[kind] += 1; never[kind] += i == r["answer"]
    print("\n1. Two-object questions: how often each kind of option is the correct one")
    for k in total:
        print(f"   {k:<58} offered {total[k]:>5}, correct {never[k]:>5}")
    ndir = Counter(sum(direction(o) is not None for o in r["options"]) for r in pair)
    print("   directional options per question:", dict(sorted(ndir.items())))

    pred = {}                                        # method -> {id: chosen option}
    rnd = random.Random(0)
    # blind rule: choose at random among the directional options
    exp = {r["id"]: 1 / sum(direction(o) is not None for o in r["options"]) for r in pair}
    # blind learned: option direction, object names, and their combinations
    def blind_pair(r):
        a, b = r["objects"]["name"][:2]
        return [None if direction(o) is None else {"d": direction(o), "a|d": a + "|" + direction(o), "b|d": b + "|" + direction(o), "ab|d": a + ">" + b + "|" + direction(o), "src|d": r["source"] + "|" + direction(o)} for o in r["options"]]
    def blind_dist(r):
        return [{"n|rel": o + "|" + r["relation"], "rel": r["relation"], "pos": str(i)} for i, o in enumerate(r["options"])]
    pred["blind, learned"] = {**cv_option_model(pair, blind_pair), **cv_option_model(dist, blind_dist)}

    # boxes, rules
    def rule_pair(r):
        (ax, ay, *_), (bx, by, *_) = box(r, 0), box(r, 1)
        s = {"left": bx - ax, "right": ax - bx, "above": by - ay, "under": ay - by}
        c = [(s[direction(o)], i) for i, o in enumerate(r["options"]) if direction(o)]
        return max(c)[1]
    def lookup(r, name):
        return box(r, r["objects"]["name"].index(name)) if name in r["objects"]["name"] else None
    def rule_dist(r, key):
        c = [(key(lookup(r, o)), i) for i, o in enumerate(r["options"]) if lookup(r, o)]
        return (max(c) if r["relation"] == "close" else min(c))[1]
    pred["boxes, rules (lower in the image = nearer)"] = {**{r["id"]: rule_pair(r) for r in pair}, **{r["id"]: rule_dist(r, lambda b: b[3]) for r in dist}}
    pred["boxes, rules (larger box = nearer)"] = {**{r["id"]: rule_pair(r) for r in pair}, **{r["id"]: rule_dist(r, lambda b: b[4] * b[5]) for r in dist}}
    # boxes, learned for nearest/farthest: box geometry relative to the other options, plus the object's name
    def box_dist(r):
        bs = [lookup(r, o) for o in r["options"]]
        ok = [b for b in bs if b]
        out = []
        for o, b in zip(r["options"], bs):
            if b is None:
                out.append(None); continue
            sgn = 1 if r["relation"] == "close" else -1
            f = {"n|rel": o + "|" + r["relation"], "src": r["source"]}
            for name, j in (("bottom", 3), ("top", 2), ("cy", 1), ("h", 5), ("w", 4)):
                f[name] = sgn * (b[j] - np.mean([x[j] for x in ok]))
                f[name + "|" + r["source"]] = f[name]
                f[name + " rank"] = sgn * (sum(x[j] < b[j] for x in ok) / max(len(ok) - 1, 1) - 0.5)
            f["area"] = sgn * (b[4] * b[5] - np.mean([x[4] * x[5] for x in ok]))
            out.append(f)
        return out
    pred["boxes, rules for pairs, learned for nearest/farthest"] = {**{r["id"]: rule_pair(r) for r in pair}, **cv_option_model(dist, box_dist)}

    # ---------- table ----------
    rels = ["left", "right", "above", "under", "close", "far"]
    acc = lambda p, rs: 100 * np.mean([p[r["id"]] == r["answer"] for r in rs])
    print("\n2. Accuracy (%) without the image")
    print(f"   {'method':<56}{'all':>6}" + "".join(f"{x:>7}" for x in rels))
    e = lambda rs: 100 * np.mean([exp.get(r["id"], 0.25) for r in rs])
    print(f"   {'chance':<56}{25.0:>6.1f}" + "".join(f"{25.0:>7.1f}" for _ in rels))
    print(f"   {'blind, rule: any directional option at random':<56}{e(rows):>6.1f}" + "".join(f"{e([r for r in rows if r['relation'] == x]):>7.1f}" for x in rels))
    for m, p in pred.items():
        print(f"   {m:<56}{acc(p, rows):>6.1f}" + "".join(f"{acc(p, [r for r in rows if r['relation'] == x]):>7.1f}" for x in rels))
    print(f"   {'HFM best, trained 2B vision-language model':<56}{78.5:>6.1f}   (HFM's own evaluation; for scale)")
    print("\n3. By scene source, best box method")
    best = pred["boxes, rules for pairs, learned for nearest/farthest"]
    for s in ("ai2thor", "mp3d", "scannet"):
        rs = [r for r in rows if r["source"] == s]
        print(f"   {s:<10} all {acc(best, rs):.1f}   pairs {acc(best, [r for r in rs if r in pair]):.1f}   nearest/farthest {acc(best, [r for r in rs if r['relation'] in ('close', 'far')]):.1f}   (n {len(rs)})")
    # where the box rule fails on pairs
    wrong = [r for r in pair if best[r["id"]] != r["answer"]]
    ax = Counter()
    for r in wrong:
        ax[(r["relation"], direction(r["options"][best[r["id"]]]))] += 1
    print("\n4. Two-object questions the box rule gets wrong:", len(wrong), "of", len(pair), "; (true relation, chosen):", dict(ax.most_common(8)))
    # bootstrap interval for the headline
    ids = [r["id"] for r in rows]; ans = {r["id"]: r["answer"] for r in rows}
    b = sorted(100 * np.mean([best[i] == ans[i] for i in rnd.choices(ids, k=len(ids))]) for _ in range(1000))
    print(f"\n5. Best box method overall: {acc(best, rows):.1f}%, 95% interval {b[25]:.1f} to {b[975]:.1f}")


if __name__ == "__main__":
    main()
