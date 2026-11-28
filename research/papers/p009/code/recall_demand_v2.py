"""Revision (round 1) of Experiment C: recall demand of real text with stronger, held-out definitions of
'context-specific', document-level bootstrap CIs, and a subword-tokenised variant.

Context-specific predicates for a bigram (a, b) (statistics from the OTHER fold of a 2-fold document split,
so the window being measured never contributes to its own parametric statistics; D0 is the original in-sample rule):
  D0  b != corpus-wide top-1 successor of a, estimated in-sample (original paper, for reproduction)
  D1  b != held-out top-1 successor of a
  D2  b not among the held-out top-10 successors of a
  D3  p_ho(b|a) < 0.01
  D4  p_ho(b|a) < 0.001
with p_ho(b|a) = (C(a,b) + p_uni(b)) / (C(a) + 1) (Dirichlet smoothing, add-one unigram), i.e. a held-out
smoothed bigram language model.  For each window of length T:
  N_ctx[D] = distinct bigrams occurring >= 2 times in the window that satisfy D
  rho[D]   = fraction of positions whose bigram occurred earlier in the window and satisfies D
Corpora: prose (NLTK Gutenberg, word tokens), prose_noKJV (same without the Bible), prose_bpe (same text,
byte-level BPE, vocabulary 8192, trained on the corpus with HuggingFace `tokenizers`), code (Go 1.25.1 stdlib).
Uncertainty: cluster bootstrap over documents (B=1000): documents are resampled with replacement and every
window drawn from a resampled document enters with its multiplicity; CIs for the per-T means and for the
log-log exponents (global fit over T=256..16384 and local fit over T=8192..32768).
Output: results/demand_v2.json
"""
import os, re, glob, json, collections, numpy as np
from recall_demand import load_prose, load_code, DATA

ROOT = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(ROOT, "..", "results")
rng = np.random.default_rng(1)
TS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
DEFS = ["D0", "D1", "D2", "D3", "D4"]


def load_prose_bpe(vocab=8192):
    from tokenizers import ByteLevelBPETokenizer
    files = sorted(glob.glob(os.path.join(DATA, "gutenberg", "*.txt")))
    texts = [open(f, encoding="latin-1").read() for f in files]
    tok = ByteLevelBPETokenizer()
    tok.train_from_iterator(texts, vocab_size=vocab, min_frequency=2, show_progress=False)
    return [tok.encode(t).ids for t in texts], files


class Stats:
    def __init__(s, docs):
        s.big = collections.Counter(); s.pred = collections.Counter(); s.uni = collections.Counter()
        succ = collections.defaultdict(collections.Counter)
        for d in docs:
            s.uni.update(d)
            for a, b in zip(d, d[1:]):
                succ[a][b] += 1
        for a, c in succ.items():
            s.pred[a] = sum(c.values())
            for b, n in c.items():
                s.big[(a, b)] = n
        s.top1 = {a: c.most_common(1)[0][0] for a, c in succ.items()}
        s.top10 = {a: set(x for x, _ in c.most_common(10)) for a, c in succ.items()}
        s.ntok = sum(s.uni.values())

    def p(s, a, b, Vtot):
        pu = (s.uni.get(b, 0) + 1) / (s.ntok + Vtot)
        return (s.big.get((a, b), 0) + pu) / (s.pred.get(a, 0) + 1)


def predicates(pairs, ho, ins, Vtot):
    out = {}
    for (a, b) in pairs:
        p = ho.p(a, b, Vtot)
        out[(a, b)] = (ins.top1.get(a) != b, ho.top1.get(a) != b, b not in ho.top10.get(a, ()), p < 0.01, p < 0.001)
    return out


def window_stats(w, ho, ins, Vtot):
    bg = list(zip(w, w[1:]))
    c = collections.Counter(bg)
    pr = predicates(c.keys(), ho, ins, Vtot)
    nctx = np.zeros(5); hits = np.zeros(5); seen = set()
    for k, v in c.items():
        if v >= 2:
            nctx += pr[k]
    for k in bg:
        if k in seen:
            hits += pr[k]
        seen.add(k)
    return len(c), nctx, hits / max(len(bg), 1)


def fit(T, Y):
    return float(np.polyfit(np.log(T), np.log(Y), 1)[0])


def measure(docs, name, n_small=100, n_large=60, fold_of=None):
    lens = np.array([len(d) for d in docs]); Vtot = len(set(t for d in docs for t in d))
    if fold_of is None:
        order = np.argsort(-lens); fold_of = np.zeros(len(docs), int)
        fold_of[order[1::2]] = 1                       # alternate by length so both folds hold long documents
    st = [Stats([d for d, f in zip(docs, fold_of) if f == k]) for k in (0, 1)]
    ins = Stats(docs)
    W = []                                             # (T, doc, N_bi, nctx[5], rho[5])
    for T in TS:
        elig = np.where(lens >= T)[0]
        if len(elig) == 0:
            continue
        for _ in range(n_small if T <= 8192 else n_large):
            i = int(rng.choice(elig)); d = docs[i]; s0 = int(rng.integers(0, len(d) - T + 1))
            nb, nc, rh = window_stats(d[s0:s0 + T], st[1 - fold_of[i]], ins, Vtot)
            W.append((T, i, nb, nc, rh))
        print(name, T, len(elig), flush=True)
    Tarr = np.array([w[0] for w in W]); Darr = np.array([w[1] for w in W])
    NB = np.array([w[2] for w in W]); NC = np.array([w[3] for w in W]); RH = np.array([w[4] for w in W])
    Tu = sorted(set(Tarr.tolist()))

    def summarise(weights):
        res = {"N_bi": [], "N_ctx": [], "rho": []}
        for T in Tu:
            m = (Tarr == T); wt = weights[m]
            if wt.sum() == 0:
                for k in res: res[k].append(np.nan if k == "N_bi" else np.full(5, np.nan))
                continue
            res["N_bi"].append(float((NB[m] * wt).sum() / wt.sum()))
            res["N_ctx"].append((NC[m] * wt[:, None]).sum(0) / wt.sum())
            res["rho"].append((RH[m] * wt[:, None]).sum(0) / wt.sum())
        return res

    def exps(res):
        T = np.array(Tu, float); g = (T >= 256) & (T <= 16384); l = T >= 8192
        o = {"N_bi_global": fit(T[g], np.array(res["N_bi"])[g])}
        NCm = np.array(res["N_ctx"])
        for j, D in enumerate(DEFS):
            y = NCm[:, j]
            o[f"N_ctx_{D}_global"] = fit(T[g], y[g]) if np.all(y[g] > 0) else np.nan
            o[f"N_ctx_{D}_local"] = fit(T[l], y[l]) if np.all(y[l] > 0) and np.isfinite(y[l]).all() else np.nan
        return o

    base = summarise(np.ones(len(W))); ex0 = exps(base)
    ndoc = len(docs); boots_mean = []; boots_exp = []
    for _ in range(1000):
        cnt = np.bincount(rng.integers(0, ndoc, ndoc), minlength=ndoc).astype(float)
        r = summarise(cnt[Darr]); boots_mean.append(r); boots_exp.append(exps(r))
    rows = []
    for t_i, T in enumerate(Tu):
        m = Tarr == T
        row = {"T": T, "n_windows": int(m.sum()), "n_docs_eligible": int((lens >= T).sum()),
               "n_docs_sampled": int(len(set(Darr[m].tolist()))), "N_bi": base["N_bi"][t_i]}
        for j, D in enumerate(DEFS):
            for key in ["N_ctx", "rho"]:
                bs = np.array([b[key][t_i][j] for b in boots_mean]); bs = bs[np.isfinite(bs)]
                row[f"{key}_{D}"] = float(base[key][t_i][j])
                row[f"{key}_{D}_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        rows.append(row)
    ex_ci = {}
    for k in ex0:
        bs = np.array([b[k] for b in boots_exp], float); bs = bs[np.isfinite(bs)]
        ex_ci[k] = {"est": ex0[k], "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))] if len(bs) else None}
    return {"rows": rows, "exponents": ex_ci, "n_docs": ndoc, "n_tokens": int(lens.sum()), "vocab": Vtot,
            "fold_sizes_tokens": [int(lens[fold_of == k].sum()) for k in (0, 1)]}


def main():
    prose = load_prose(); files = sorted(glob.glob(os.path.join(DATA, "gutenberg", "*.txt")))
    out = {}
    out["prose"] = measure(prose, "prose")
    keep = [i for i, f in enumerate(files) if "bible" not in f]
    out["prose_noKJV"] = measure([prose[i] for i in keep], "prose_noKJV")
    bpe, _ = load_prose_bpe()
    out["prose_bpe"] = measure(bpe, "prose_bpe")
    out["code"] = measure(load_code(), "code")
    json.dump(out, open(os.path.join(RES, "demand_v2.json"), "w"), indent=1)
    for k, v in out.items():
        print(k, v["n_docs"], v["n_tokens"], v["vocab"])
        for e, x in v["exponents"].items():
            print("   ", e, x)
        r = v["rows"][-1]
        print("   T=", r["T"], {D: (round(r[f"N_ctx_{D}"], 1), round(r[f"rho_{D}"], 3)) for D in DEFS})


if __name__ == "__main__":
    main()
