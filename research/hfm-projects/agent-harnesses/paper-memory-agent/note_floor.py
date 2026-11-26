"""How much of what readers ask about survives in a 3,000-character note? A model-free floor.

Stand-in for HFM's "Building an Agent for Preserving Knowledge from Research Papers"
track, which needs a paid language-model API. HFM's task: turn an NLP paper into a
note of at most 3,000 characters, from which a frozen model later answers undisclosed
questions. Here no language model is used. Notes are built by selecting sentences, and
scored by whether the gold answer material of QASPER questions (Dasigi et al. 2021,
CC-BY 4.0) is still in the note. That measures the note's information content, which
bounds what any reader could answer from it; it is not an answer-F1 score.

Note builders (all cut at the character budget)
  lead               title, abstract, then the paper from the top
  abstract + firsts  abstract, then the first sentence of every paragraph
  central            sentences closest to the paper's TF-IDF centroid, with a redundancy penalty
  numbers first      abstract, then sentences containing numbers, in paper order
  learned            a logistic regression, trained on the training papers, predicting from
                     position, section type and words whether a sentence is answer evidence
                     for some question; highest scores first ("question anticipation")
  learned + dense    the same, ranked by score per character and with citations stripped
  oracle             knows the questions: packs gold evidence first (the budget's ceiling)

Scores, over answerable questions with text evidence
  evidence kept      share of questions with a gold evidence sentence in the note
  span kept          share of extractive questions whose full answer span is in the note
  answer words       mean share of the gold answer's content words present in the note

The builder is chosen on the validation papers and reported on the test papers.

Usage: python note_floor.py <train.parquet> <validation.parquet> <test.parquet>
"""
import re
import sys
from collections import Counter

import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

BUDGET = 3000
MIN_LEN, MAX_LEN = 4000, 86000
STOP = set("a an the of to in on for and or is are was were be been by with as at from that this it its we our their they which "
           "can not no yes do does did than then so such these those into over under between both each also more most".split())
SECTIONS = [("intro", "introduction"), ("related", "related|background|previous"), ("data", "data|corpus|corpora|annotation"),
            ("method", "model|method|approach|architecture|system|framework"), ("setup", "experiment|setup|setting|implementation|training"),
            ("result", "result|evaluation|performance"), ("analysis", "analysis|discussion|ablation|error"), ("end", "conclusion|future")]


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%. ]", " ", t.lower())).strip()


def words(t):
    return [w for w in re.findall(r"[a-z0-9%]+(?:\.[0-9]+)?", t.lower()) if w not in STOP]


def sentences(par):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(\[])", par) if len(s.strip()) > 15]


def section_type(name):
    name = (name or "").lower()
    for key, pat in SECTIONS:
        if re.search(pat, name):
            return key
    return "other"


def rows(obj):
    """A struct of lists or a list of structs, as a list of dicts."""
    if isinstance(obj, dict):
        keys = list(obj)
        n = len(obj[keys[0]]) if keys else 0
        return [{k: obj[k][i] for k in keys} for i in range(n)]
    return list(obj or [])


def load(path):
    papers = []
    for rec in pq.read_table(path).to_pylist():
        sents = [{"t": s, "sec": "abstract", "par": 0, "first": i == 0, "abs": True} for i, s in enumerate(sentences(rec.get("abstract") or ""))]
        ft = rec["full_text"]
        par_id = 0
        for sec in rows(ft):
            for par in sec.get("paragraphs") or []:
                par_id += 1
                for i, s in enumerate(sentences(par or "")):
                    sents.append({"t": s, "sec": section_type(sec.get("section_name")), "par": par_id, "first": i == 0, "abs": False})
        total = sum(len(s["t"]) for s in sents)
        if not (MIN_LEN <= total <= MAX_LEN) or len(sents) < 10:
            continue
        for i, s in enumerate(sents):
            s["pos"] = i / len(sents)
            s["n"] = norm(s["t"])
        qs = []
        for q in rows(rec["qas"]):
            anns = []
            for a in _answers(q["answers"]):
                if a.get("unanswerable"):
                    continue
                ev = [e for e in (a.get("evidence") or []) if e and not e.startswith("FLOAT SELECTED")]
                spans = [s for s in (a.get("extractive_spans") or []) if s]
                text = " ".join(spans) if spans else (a.get("free_form_answer") or "")
                if ev:
                    anns.append({"ev": [norm(e) for e in ev], "spans": [norm(s) for s in spans], "ans": words(text)})
            if anns:
                qs.append(anns)
        if qs:
            papers.append({"title": rec.get("title") or "", "sents": sents, "qs": qs, "chars": total})
    return papers


def _answers(obj):
    """All annotators' answer dicts for one question, whatever the nesting."""
    out = []
    if isinstance(obj, dict) and "answer" in obj:
        a = obj["answer"]
        out += rows(a) if isinstance(a, (dict, list)) else []
    else:
        for x in rows(obj):
            a = x.get("answer") if isinstance(x, dict) else None
            if isinstance(a, dict) and "evidence" in a and not isinstance(a.get("unanswerable"), list):
                out.append(a)
            else:
                out += rows(a)
    return out


def evidence_flags(p):
    """For each sentence, whether it lies inside some gold evidence paragraph."""
    ev = [e for q in p["qs"] for a in q for e in a["ev"]]
    return np.array([any(s["n"] and s["n"] in e for e in ev) for s in p["sents"]])


def strip_cites(t):
    t = re.sub(r"\s*\((?:[^()]*\b(?:19|20)\d\d[^()]*|BIBREF\d+|TABREF\d+|FIGREF\d+)\)", "", t)
    return re.sub(r"\s*(?:BIBREF|TABREF|FIGREF|SECREF|INLINEFORM|DISPLAYFORM)\d+", "", t)


def pack(p, order, budget, dense=False):
    """Take sentences in the given priority order until the budget is full; return the
    note in paper order. Sentences that do not fit are skipped."""
    used, chosen = 0, []
    for i in order:
        text = strip_cites(p["sents"][i]["t"]) if dense else p["sents"][i]["t"]
        if used + len(text) + 1 <= budget:
            chosen.append((i, text))
            used += len(text) + 1
    return " ".join(t for _, t in sorted(chosen))


def build_lead(p, budget, **_):
    return (p["title"] + ". " + " ".join(s["t"] for s in p["sents"]))[:budget]


def build_firsts(p, budget, **_):
    n = len(p["sents"])
    order = [i for i in range(n) if p["sents"][i]["abs"]] + [i for i in range(n) if p["sents"][i]["first"] and not p["sents"][i]["abs"]]
    return pack(p, order, budget)


def build_central(p, budget, **_):
    X = TfidfVectorizer(sublinear_tf=True).fit_transform([s["n"] for s in p["sents"]])
    c = np.asarray(X.mean(axis=0))
    c /= np.linalg.norm(c) + 1e-12
    rel = np.asarray(X @ c.T).ravel()
    chosen, used, cand = [], 0, set(range(len(rel)))
    red = np.zeros(len(rel))
    while cand:
        i = max(cand, key=lambda j: rel[j] - 0.5 * red[j])
        cand.discard(i)
        if used + len(p["sents"][i]["t"]) + 1 > budget:
            if used > budget - 60:
                break
            continue
        chosen.append(i)
        used += len(p["sents"][i]["t"]) + 1
        red = np.maximum(red, np.asarray((X @ X[i].T).todense()).ravel())
    return " ".join(p["sents"][i]["t"] for i in sorted(chosen))


def build_numbers(p, budget, **_):
    n = len(p["sents"])
    order = [i for i in range(n) if p["sents"][i]["abs"]] + [i for i in range(n) if not p["sents"][i]["abs"] and re.search(r"\d", p["sents"][i]["t"])]
    return pack(p, order, budget)


def structure(p):
    keys = ["abstract", "other"] + [k for k, _ in SECTIONS]
    F = []
    for s in p["sents"]:
        f = [float(s["sec"] == k) for k in keys]
        f += [float(s["first"]), s["pos"], s["pos"] ** 2, float(bool(re.search(r"\d", s["t"]))), float("%" in s["t"]),
              min(len(s["t"]), 400) / 400, float(len(re.findall(r"\d+(?:\.\d+)?", s["t"])) >= 3)]
        F.append(f)
    return np.array(F)


class Learned:
    def fit(self, papers):
        self.vec = TfidfVectorizer(ngram_range=(1, 2), min_df=20, max_features=30000, binary=True, use_idf=False, norm=None)
        text = [s["n"] for p in papers for s in p["sents"]]
        X = hstack([self.vec.fit_transform(text), csr_matrix(np.vstack([structure(p) for p in papers]))]).tocsr()
        y = np.concatenate([evidence_flags(p) for p in papers])
        self.model = LogisticRegression(C=0.3, max_iter=300).fit(X, y)
        print(f"learned: {len(y)} training sentences, {y.mean():.1%} are evidence", flush=True)
        names = list(self.vec.get_feature_names_out())
        w = self.model.coef_[0][:len(names)]
        top = np.argsort(w)
        print("  words that raise the score: " + ", ".join(names[i] for i in top[-25:][::-1]))
        print("  words that lower the score: " + ", ".join(names[i] for i in top[:25]))
        return self

    def score(self, p):
        X = hstack([self.vec.transform([s["n"] for s in p["sents"]]), csr_matrix(structure(p))]).tocsr()
        return self.model.predict_proba(X)[:, 1]


def build_learned(p, budget, model=None, dense=False, **_):
    s = model.score(p)
    if dense:
        s = s / np.sqrt(np.array([len(strip_cites(x["t"])) + 20 for x in p["sents"]]))
    return pack(p, list(np.argsort(-s)), budget, dense)


def build_oracle(p, budget, **_):
    flags = evidence_flags(p)
    return pack(p, [i for i in range(len(flags)) if flags[i]] + [i for i in range(len(flags)) if not flags[i]], budget)


def score(p, note):
    n = norm(note)
    note_words = set(words(note))
    kept = [s["n"] for s in p["sents"] if s["n"] and s["n"] in n]
    ev, span, aw = [], [], []
    for q in p["qs"]:
        ev.append(max(float(any(k in e for e in a["ev"] for k in kept)) for a in q))
        with_spans = [a for a in q if a["spans"]]
        if with_spans:
            span.append(max(float(all(s in n for s in a["spans"])) for a in with_spans))
        with_ans = [a for a in q if a["ans"]]
        if with_ans:
            aw.append(max(np.mean([w in note_words for w in a["ans"]]) for a in with_ans))
    return ev, span, aw


def evaluate(papers, build, budget=BUDGET, **kw):
    E, S, A = [], [], []
    for p in papers:
        e, s, a = score(p, build(p, budget, **kw))
        E += e
        S += s
        A += a
    return np.mean(E), np.mean(S), np.mean(A), len(E), len(S)


def main():
    train, valid, test = (load(f) for f in sys.argv[1:4])
    for name, d in (("train", train), ("validation", valid), ("test", test)):
        print(f"{name}: {len(d)} papers in the 4,000-86,000 character range, {sum(len(p['qs']) for p in d)} answerable text questions, "
              f"median length {int(np.median([p['chars'] for p in d]))} characters", flush=True)
    model = Learned().fit(train)
    builders = {"lead": (build_lead, {}), "abstract + firsts": (build_firsts, {}), "central": (build_central, {}),
                "numbers first": (build_numbers, {}), "learned": (build_learned, {"model": model}),
                "learned + dense": (build_learned, {"model": model, "dense": True}), "oracle": (build_oracle, {})}
    res = {}
    for split, d in (("validation", valid), ("test", test)):
        print(f"\n===== {split}: budget {BUDGET} characters =====")
        print(f"  {'builder':20s}{'evidence kept':>15s}{'span kept':>12s}{'answer words':>14s}")
        for name, (fn, kw) in builders.items():
            e, s, a, ne, ns = evaluate(d, fn, **kw)
            res[split, name] = (e, s, a)
            print(f"  {name:20s}{e:15.3f}{s:12.3f}{a:14.3f}", flush=True)
        print(f"  ({ne} questions; {ns} with extractive spans)")
    real = [n for n in builders if n != "oracle"]
    best = max(real, key=lambda n: res["validation", n][0])
    print(f"\nRESULT builder chosen on validation (by evidence kept): {best}")
    for name in ("lead", best, "oracle"):
        e, s, a = res["test", name]
        print(f"RESULT test, {name:18s} evidence kept {e:.3f}  span kept {s:.3f}  answer words {a:.3f}")
    print("\n===== test: how the budget changes the picture (evidence kept) =====")
    print(f"  {'budget':>8s}{'lead':>9s}{best:>20s}{'oracle':>9s}")
    for b in (1000, 2000, 3000, 6000, 12000, 24000):
        vals = [evaluate(test, builders[n][0], b, **builders[n][1])[0] for n in ("lead", best, "oracle")]
        print(f"  {b:8d}{vals[0]:9.3f}{vals[1]:20.3f}{vals[2]:9.3f}", flush=True)
    # where the evidence sits
    pos = np.concatenate([np.array([s["pos"] for s in p["sents"]])[evidence_flags(p)] for p in train])
    hist = np.histogram(pos, bins=10, range=(0, 1))[0] / len(pos)
    print("\nRESULT share of evidence sentences by tenth of the paper (training papers): " + " ".join(f"{h:.2f}" for h in hist))
    sec = Counter(s["sec"] for p in train for s, f in zip(p["sents"], evidence_flags(p)) if f)
    allsec = Counter(s["sec"] for p in train for s in p["sents"])
    print("RESULT evidence rate by section type: " + ", ".join(f"{k} {sec[k] / allsec[k]:.2f}" for k in sorted(allsec, key=lambda k: -sec[k] / allsec[k])))


if __name__ == "__main__":
    main()
