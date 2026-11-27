"""TOPv2 on CPU: how much of compositional semantic parsing is recall of seen structures?

Experiment for HFM's "TOPv2 compositional semantic parsing" track, where a 7-billion-
parameter language model is fine-tuned on 12,288 examples to 83% exact match and 17
proposals produced no accepted improvement. That work needs a GPU. Without one, on
the public dataset (TOPv2, Chen et al. 2020), two things can be measured:

  1. Structure coverage. Strip the words from each parse to get its frame, for
     example [IN:GET_WEATHER [SL:LOCATION ] ]. What share of test parses use a frame
     already present in training?
  2. A classical parser. Each word is tagged with its full path in the tree and a
     linear classifier predicts each word's tag from the words around it and the
     words of the whole utterance; the tree is rebuilt from the tags. Exact match =
     the whole tree is right. (A conditional random field over the same tags was
     tried first; with thousands of distinct tags it did not finish in two hours.)

Results are split by whether the frame was seen in training, which is the
"compositional" part of the task.

Usage: python topv2_classical.py <train.parquet> <test.parquet> [training examples ...]
"""
from __future__ import annotations

import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier


def to_tags(parse):
    """Words and one tag per word: the path of open nodes, each marked ^ on the node's first word."""
    words, tags, stack = [], [], []
    for tok in parse.split():
        if tok.startswith("[") and len(tok) > 1:
            stack.append([tok[1:], True])
        elif tok == "]":
            if stack:
                stack.pop()
        else:
            words.append(tok)
            tags.append("|".join(lab + ("^" if first else "") for lab, first in stack) or "O")
            for s in stack:
                s[1] = False
    return words, tags


def to_parse(words, tags):
    out, stack = [], []
    for w, t in zip(words, tags):
        path = [] if t == "O" else t.split("|")
        keep = 0
        while keep < len(stack) and keep < len(path) and not path[keep].endswith("^") and path[keep] == stack[keep]:
            keep += 1
        out += ["]"] * (len(stack) - keep)
        stack = stack[:keep]
        for p in path[keep:]:
            lab = p.rstrip("^")
            out.append("[" + lab)
            stack.append(lab)
        out.append(w)
    return " ".join(out + ["]"] * len(stack))


def frame(parse):
    return " ".join(t for t in parse.split() if t == "]" or (t.startswith("[") and len(t) > 1))


def shape(w):
    return "".join("A" if c.isupper() else "a" if c.islower() else "0" if c.isdigit() else "-" for c in w)[:6]


def feats(sent):
    rows = []
    n = len(sent)
    for i, w in enumerate(sent):
        lw = w.lower()
        f = {"w": lw, "s3": lw[-3:], "p3": lw[:3], "shape": shape(w), "pos": min(i, 6), "rev": min(n - 1 - i, 6), "first": sent[0].lower(), "len": min(n, 15)}
        for d in (-3, -2, -1, 1, 2, 3):
            j = i + d
            f[f"w{d}"] = sent[j].lower() if 0 <= j < n else "<pad>"
        if i > 0:
            f["bi-"] = sent[i - 1].lower() + "_" + lw
        if i + 1 < n:
            f["bi+"] = lw + "_" + sent[i + 1].lower()
        for v in set(x.lower() for x in sent):                   # the whole utterance as a bag of words, to carry the intent
            f["has:" + v] = 1
        rows.append(f)
    return rows


def main():
    tr, te = pd.read_parquet(sys.argv[1]), pd.read_parquet(sys.argv[2])
    print("columns:", list(tr.columns), "train", len(tr), "test", len(te))
    col = "semantic_parse" if "semantic_parse" in tr.columns else [c for c in tr.columns if "parse" in c.lower()][0]
    print("example:", tr[col].iloc[0])
    norm = lambda s: " ".join(str(s).split())
    tr_p, te_p = [norm(x) for x in tr[col]], [norm(x) for x in te[col]]
    rt = np.mean([to_parse(*to_tags(p)) == p for p in te_p])
    print(f"tag encoding reproduces the gold tree for {100 * rt:.1f}% of test parses (the rest cannot be scored as correct)")
    sizes = [int(x) for x in sys.argv[3:]] or [12288]
    rng = np.random.default_rng(0)
    order = rng.permutation(len(tr_p))
    te_w = [to_tags(p)[0] for p in te_p]
    if len(te_p) > 8000:                                          # keep the test pass quick
        pick = rng.permutation(len(te_p))[:8000]
        te_p, te_w = [te_p[i] for i in pick], [te_w[i] for i in pick]
    for n in sizes:
        sub = [tr_p[i] for i in order[:n]]
        frames = Counter(frame(p) for p in sub)
        seen = np.array([frame(p) in frames for p in te_p])
        memo = {}
        for p in sub:
            memo.setdefault(" ".join(to_tags(p)[0]).lower(), p)
        exact_memo = np.array([memo.get(" ".join(w).lower()) == p for w, p in zip(te_w, te_p)])
        pairs = [(w, t) for w, t in (to_tags(p) for p in sub) if w]
        Y = [t for _, ts in pairs for t in ts]
        t0 = time.time()
        vec = DictVectorizer(dtype=np.float32)

        def matrix(rows, fit=False):
            m = (vec.fit_transform(rows) if fit else vec.transform(rows)).tocsr()
            m.indices, m.indptr = m.indices.astype(np.int32), m.indptr.astype(np.int32)
            return m

        X = matrix([f for w, _ in pairs for f in feats(w)], fit=True)
        clf = SGDClassifier(loss="modified_huber", alpha=2e-6, max_iter=12, tol=None, n_jobs=-1, random_state=0).fit(X, Y)
        flat = clf.predict(matrix([f for w in te_w for f in feats(w)]))
        pred, k = [], 0
        for w in te_w:
            pred.append(list(flat[k:k + len(w)]))
            k += len(w)
        ok = np.array([to_parse(w, t) == p for w, t, p in zip(te_w, pred, te_p)])
        fr_ok = np.array([frame(to_parse(w, t)) == frame(p) for w, t, p in zip(te_w, pred, te_p)])
        print(f"\ntraining examples: {n}   ({len(frames)} distinct frames, {len(set(Y))} distinct tags, fit {time.time() - t0:.0f}s, {len(te_p)} test parses)")
        print(f"  test parses whose frame was seen in training:        {100 * seen.mean():.1f}%")
        print(f"  identical utterance in training with the same parse: {100 * exact_memo.mean():.1f}%")
        print(f"  classical parser, exact match:                       {100 * ok.mean():.1f}%   (frame right {100 * fr_ok.mean():.1f}%)")
        print(f"     on seen frames   {100 * ok[seen].mean():.1f}% of {seen.sum()}")
        print(f"     on unseen frames {100 * ok[~seen].mean() if (~seen).any() else float('nan'):.1f}% of {(~seen).sum()}")
        depth = np.array([max((t.count('|') + 1 if t != 'O' else 0) for t in to_tags(p)[1]) if to_tags(p)[1] else 0 for p in te_p])
        for lo, hi, name in ((0, 2, "flat (one intent and its slots)"), (3, 99, "nested (an intent inside a slot)")):
            m = (depth >= lo) & (depth <= hi)
            print(f"     {name:<34} {100 * ok[m].mean():.1f}% of {m.sum()}")
    print("\nFor scale: HFM's fine-tuned 7B language model reaches 83% exact match on HFM's own frozen test with 12,288 training rows; not directly comparable.")


if __name__ == "__main__":
    main()
