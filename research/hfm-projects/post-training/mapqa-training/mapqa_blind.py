"""MapQA without the map: how many answers follow from the question alone?

HFM's track trains a vision-language model on MapQA (questions about choropleth maps
of US states) and scores 0.83. That needs a GPU. On a CPU we can measure the floor:
what a model that never sees the map gets, from the wording of the question and the
habits of the answers. Data: the public copy nimapourjafar/mm_mapqa on Hugging Face
(first file: 5,346 maps with their question-answer turns; text only was fetched).

Whole maps are held out in every fold. Score: exact match after trimming and lower-casing.

Usage: python mapqa_blind.py <mapqa.jsonl>
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

STYLE = re.compile(r"\n.*$", re.S)   # the trailing style instruction ("Be succinct.")


def norm(a):
    return a.strip().rstrip(".").lower()


def template(q):
    """Replace state names and regions by placeholders to get the question type."""
    q = re.sub(r"\b(the )?(USA|Northeast|South|West|MidWest|Midwest)\b", "<region>", q)
    q = re.sub(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b(?<!Does)(?<!Which)(?<!What)(?<!Name)(?<!Among)(?<!Is)(?<!Are)(?<!How)(?<!In)", "<state>", q)
    return q


def main():
    qa = []
    for line in open(sys.argv[1]):
        r = json.loads(line)
        turns = [t for t in r["data"] if t["modality"] == "text"]
        for u, a in zip(turns[0::2], turns[1::2]):
            if u["role"] == "user" and a["role"] == "assistant":
                q = STYLE.sub("", u["data"]).strip()
                qa.append((r["row"], q, norm(a["data"])))
    maps = sorted({m for m, _, _ in qa})
    print(f"maps: {len(maps)}; question-answer pairs: {len(qa)}; per map {len(qa) / len(maps):.1f}")
    kind = lambda a: "yes/no" if a in ("yes", "no") else "a number or range" if re.search(r"\d", a) else "one or more states"
    kinds = Counter(kind(a) for _, _, a in qa)
    print("answer kinds:", {k: f"{100 * v / len(qa):.1f}%" for k, v in kinds.most_common()})
    tmpl = Counter(template(q) for _, q, _ in qa)
    print(f"question templates: {len(tmpl)}; the 12 most common cover {100 * sum(v for _, v in tmpl.most_common(12)) / len(qa):.1f}%")
    for t, n in tmpl.most_common(12):
        ans = Counter(a for _, q, a in qa if template(q) == t)
        top = ans.most_common(1)[0]
        print(f"   {n:>6}  {t[:74]:<74} most common answer '{top[0][:18]}' {100 * top[1] / n:.0f}%")

    random.Random(0).shuffle(maps)
    fold = {m: i % 5 for i, m in enumerate(maps)}
    res = defaultdict(list)   # method -> list of (kind, correct)
    for k in range(5):
        tr = [x for x in qa if fold[x[0]] != k]
        te = [x for x in qa if fold[x[0]] == k]
        overall = Counter(a for _, _, a in tr).most_common(1)[0][0]
        by_t = defaultdict(Counter)
        for _, q, a in tr:
            by_t[template(q)][a] += 1
        by_q = defaultdict(Counter)
        for _, q, a in tr:
            by_q[q][a] += 1
        # learned: words of the question -> answer, restricted to answers seen 20+ times
        common = {a for a, n in Counter(a for _, _, a in tr).items() if n >= 20}
        sub = [x for x in tr if x[2] in common]
        random.Random(k).shuffle(sub)
        sub = sub[:120000]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True)
        clf = LogisticRegression(C=5.0, max_iter=300).fit(vec.fit_transform([q for _, q, _ in sub]), [a for _, _, a in sub])
        pred = clf.predict(vec.transform([q for _, q, _ in te]))
        for (m, q, a), p in zip(te, pred):
            kd = kind(a)
            res["always the most common answer"].append((kd, overall == a))
            t = by_t.get(template(q))
            res["most common answer for the question type"].append((kd, (t.most_common(1)[0][0] if t else overall) == a))
            e = by_q.get(q)
            res["most common answer for the exact question, else its type"].append((kd, (e.most_common(1)[0][0] if e else (t.most_common(1)[0][0] if t else overall)) == a))
            res["learned from the question's words"].append((kd, p == a))
    print(f"\nExact-match accuracy (%) without the map, whole maps held out\n   {'method':<58}{'all':>6}{'yes/no':>8}{'states':>8}{'number':>8}")
    for m, v in res.items():
        f = lambda kd: 100 * np.mean([c for k_, c in v if kd is None or k_ == kd])
        print(f"   {m:<58}{f(None):>6.1f}{f('yes/no'):>8.1f}{f('one or more states'):>8.1f}{f('a number or range'):>8.1f}")
    print(f"   {'HFM best, trained vision-language model':<58}{82.8:>6.1f}   (HFM's own evaluation and question mix; for scale)")
    yes = Counter(a for _, _, a in qa if a in ("yes", "no"))
    print(f"\nyes/no balance: yes {100 * yes['yes'] / sum(yes.values()):.1f}%")


if __name__ == "__main__":
    main()
