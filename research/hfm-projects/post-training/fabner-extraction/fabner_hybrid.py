"""FabNER: can a small language model find the entities a dictionary cannot?

Follow-up to fabner_ceiling.py. That study found a dictionary of training phrases
scores 83.4 F1, and that the headroom is in the 14% of test entities whose phrase
never appears as an entity in training, of which a classical model finds 7%.
This asks whether a small open model (Qwen2.5-3B-Instruct, 4-bit, llama.cpp on a
free 4-core runner), shown the 6 most similar training sentences with their
entities, finds those unseen entities, and whether adding its proposals to the
dictionary's output helps.

Systems, on the first 300 test sentences
  dictionary        longest match of training entity phrases, most common type
  model             entities listed by the model
  hybrid            dictionary output, plus model entities that do not overlap it
  hybrid, unseen    as hybrid, but only model entities whose phrase is not in the
                    training dictionary at all (so the model cannot undo it)

Usage: python fabner_hybrid.py <model.gguf> <train.parquet> <test.parquet> <labels.json> <out.jsonl>
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fabner_ceiling import spans  # noqa: E402


def prf(gold, pred):
    tp = sum(len(set(g) & set(p)) for g, p in zip(gold, pred))
    npred, ngold = sum(len(set(p)) for p in pred), sum(len(set(g)) for g in gold)
    pr, rc = tp / max(npred, 1), tp / max(ngold, 1)
    return 100 * 2 * pr * rc / max(pr + rc, 1e-9), 100 * pr, 100 * rc


def locate(low, phrase, taken):
    """All non-overlapping places where the phrase's tokens occur in the sentence."""
    k, out = phrase.lower().split(), []
    for i in range(len(low) - len(k) + 1):
        if k and low[i:i + len(k)] == k and not any(taken[i:i + len(k)]):
            out.append((i, i + len(k)))
    return out


def main():
    model_path, trp, tep, labp, out = sys.argv[1:6]
    from sklearn.feature_extraction.text import TfidfVectorizer
    from llama_cpp import Llama
    names = json.load(open(labp))
    load = lambda p: [(list(map(str, r.tokens)), [names[t] for t in r.ner_tags]) for r in pd.read_parquet(p).itertuples()]
    tr, te = load(trp), load(tep)[:300]
    types = sorted({t.split("-")[1] for t in names if "-" in t})
    seen = defaultdict(Counter)
    for s, t in tr:
        for a, b, ty in spans(t):
            seen[tuple(x.lower() for x in s[a:b])][ty] += 1
    maxlen = max(len(k) for k in seen)

    def dict_spans(s):
        low, res, i = [x.lower() for x in s], [], 0
        while i < len(s):
            for n in range(min(maxlen, len(s) - i), 0, -1):
                k = tuple(low[i:i + n])
                if k in seen:
                    res.append((i, i + n, seen[k].most_common(1)[0][0]))
                    i += n - 1
                    break
            i += 1
        return res

    text = [" ".join(s) for s, _ in tr]
    vec = TfidfVectorizer(sublinear_tf=True).fit(text)
    P = vec.transform(text)
    listing = lambda s, t: "\n".join(f"{' '.join(s[a:b])} | {ty}" for a, b, ty in spans(t)) or "(none)"
    llm = Llama(model_path=model_path, n_ctx=4096, n_threads=os.cpu_count(), verbose=False)
    sys_msg = ("You extract entities from sentences about manufacturing. List every entity on its own line as `phrase | TYPE`, copying the phrase exactly from the sentence. "
               f"Types: {', '.join(types)}. Follow the examples' conventions. If there are none, write (none).")
    gold, D, M, H, HU, rows = [], [], [], [], [], []
    with open(out, "w") as fo:
        for n, (s, t) in enumerate(te):
            low = [x.lower() for x in s]
            near = np.argsort(-(P @ vec.transform([" ".join(s)]).T).toarray().ravel())[:6]
            shots = "\n\n".join(f"Sentence: {text[j]}\nEntities:\n{listing(*tr[j])}" for j in near[::-1])
            t0 = time.time()
            r = llm.create_chat_completion(messages=[{"role": "system", "content": sys_msg}, {"role": "user", "content": f"{shots}\n\nSentence: {' '.join(s)}\nEntities:"}], max_tokens=220, temperature=0.0)
            reply = r["choices"][0]["message"]["content"]
            m, taken = [], [False] * len(s)
            for line in reply.split("\n"):
                if "|" not in line:
                    continue
                ph, ty = (x.strip(" `") for x in line.rsplit("|", 1))
                if ty not in types:
                    continue
                for a, b in locate(low, ph, taken):
                    m.append((a, b, ty))
                    taken[a:b] = [True] * (b - a)
            d = dict_spans(s)
            covered = [False] * len(s)
            for a, b, _ in d:
                covered[a:b] = [True] * (b - a)
            extra = [x for x in m if not any(covered[x[0]:x[1]])]
            extra_unseen = [x for x in extra if tuple(low[x[0]:x[1]]) not in seen]
            g = spans(t)
            gold.append(g); D.append(d); M.append(m); H.append(d + extra); HU.append(d + extra_unseen)
            fo.write(json.dumps(dict(n=n, gold=g, dictionary=d, model=m, reply=reply[:500], secs=time.time() - t0)) + "\n"); fo.flush()
            if (n + 1) % 20 == 0:
                print(f"{n + 1} sentences; F1 so far: dictionary {prf(gold, D)[0]:.1f}, model {prf(gold, M)[0]:.1f}, hybrid {prf(gold, H)[0]:.1f}; {time.time() - t0:.0f}s last", flush=True)
    print(f"\nentity-level micro F1 (precision, recall) on {len(te)} test sentences, {sum(map(len, gold))} entities")
    for name, p in (("dictionary", D), ("model with 6 retrieved examples", M), ("hybrid: dictionary + non-overlapping model entities", H), ("hybrid: dictionary + model entities with unseen phrases", HU)):
        print("  %-56s %.1f  (%.1f, %.1f)" % (name, *prf(gold, p)))
    unseen = [[x for x, s_ in zip(g, [te[i][0]] * len(g)) if tuple(w.lower() for w in s_[x[0]:x[1]]) not in seen] for i, g in enumerate(gold)]
    nu = sum(map(len, unseen))
    print(f"\nrecall on the {nu} test entities whose phrase is never an entity in training ({100 * nu / sum(map(len, gold)):.1f}% of entities)")
    for name, p in (("dictionary", D), ("model", M), ("hybrid", H)):
        print(f"  {name:<12} {100 * sum(len(set(u) & set(q)) for u, q in zip(unseen, p)) / max(nu, 1):.1f}%")
    add = sum(len(h) - len(d) for h, d in zip(HU, D)); right = sum(len((set(h) - set(d)) & set(g)) for h, d, g in zip(HU, D, gold))
    print(f"\nentities the model adds with unseen phrases: {add}, of which correct: {right} ({100 * right / max(add, 1):.0f}%)")


if __name__ == "__main__":
    main()
