"""Experiment C: how does the in-context associative-recall demand of real text grow with context length?

Corpora (public, local):
  prose : NLTK Gutenberg selection (18 public-domain books), word tokens [a-z]+ (lower-cased)
  code  : Go 1.25 standard library sources (/usr/local/go1.25.1/src), concatenated per package directory,
          tokens = identifiers/numbers (\\w+) and single punctuation characters.
For random windows of length T inside one document:
  N_bi(T)  = distinct bigrams (a,b)
  N_rep(T) = distinct bigrams occurring >= 2 times in the window
  N_ctx(T) = distinct bigrams occurring >= 2 times whose successor b is NOT the corpus-wide top-1 successor
             of a (associations that must be read from context rather than from parametric statistics)
  rho(T)   = fraction of positions t in the window whose bigram (x_{t-1},x_t) already occurred earlier in the
             window AND is not the global top-1 continuation ("context-only recall opportunities")
Fits N(T) = c T^beta by least squares in log-log space over T in [256, 16384].
Output: results/demand.json, results/fig_demand.png
"""
import os, re, glob, json, collections, numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "..", "results"); DATA = os.path.join(ROOT, "..", "data")
rng = np.random.default_rng(0)
TS = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def load_prose():
    docs = []
    for f in sorted(glob.glob(os.path.join(DATA, "gutenberg", "*.txt"))):
        docs.append(re.findall(r"[a-z]+", open(f, encoding="latin-1").read().lower()))
    return docs


def load_code():
    base = "/usr/local/go1.25.1/src"
    docs = []
    for d in sorted({os.path.dirname(p) for p in glob.glob(base + "/**/*.go", recursive=True)}):
        if "testdata" in d or "/vendor/" in d:
            continue
        txt = "".join(open(f, encoding="utf-8", errors="ignore").read()
                      for f in sorted(glob.glob(d + "/*.go")) if not f.endswith("_test.go"))
        toks = re.findall(r"\w+|[^\s\w]", txt)
        if len(toks) >= 4096:
            docs.append(toks)
    return docs


def top1_successor(docs):
    cnt = collections.defaultdict(collections.Counter)
    for d in docs:
        for a, b in zip(d, d[1:]):
            cnt[a][b] += 1
    return {a: c.most_common(1)[0][0] for a, c in cnt.items()}


def window_stats(w, top1):
    bg = list(zip(w, w[1:]))
    c = collections.Counter(bg)
    n_rep = sum(1 for k, v in c.items() if v >= 2)
    n_ctx = sum(1 for (a, b), v in c.items() if v >= 2 and top1.get(a) != b)
    seen = set(); hits = 0
    for (a, b) in bg:
        if (a, b) in seen and top1.get(a) != b:
            hits += 1
        seen.add((a, b))
    return len(c), n_rep, n_ctx, hits / max(len(bg), 1)


def measure(docs, name, samples=60):
    top1 = top1_successor(docs)
    lens = np.array([len(d) for d in docs])
    rows = []
    for T in TS:
        elig = np.where(lens >= T)[0]
        if len(elig) == 0:
            continue
        vals = []
        for _ in range(samples if T <= 8192 else samples // 2):
            d = docs[rng.choice(elig)]; s = rng.integers(0, len(d) - T + 1)
            vals.append(window_stats(d[s:s + T], top1))
        v = np.array(vals)
        rows.append({"T": T, "n_docs_eligible": int(len(elig)), "N_bi": float(v[:, 0].mean()),
                     "N_rep": float(v[:, 1].mean()), "N_ctx": float(v[:, 2].mean()),
                     "N_ctx_sd": float(v[:, 2].std()), "rho": float(v[:, 3].mean())})
        print(name, rows[-1], flush=True)
    fits = {}
    for key in ["N_bi", "N_rep", "N_ctx"]:
        r = [x for x in rows if 256 <= x["T"] <= 16384]
        X = np.log([x["T"] for x in r]); Y = np.log([x[key] for x in r])
        beta, logc = np.polyfit(X, Y, 1)
        # bootstrap over T points is meaningless; report residual-based SE
        res = Y - (beta * X + logc); se = np.sqrt(res.var(ddof=2) / ((X - X.mean()) ** 2).sum())
        fits[key] = {"beta": float(beta), "c": float(np.exp(logc)), "beta_se": float(se)}
    return {"rows": rows, "fits": fits, "n_docs": len(docs), "n_tokens": int(lens.sum()),
            "vocab": len(set(t for d in docs for t in d))}


def main():
    out = {"prose": measure(load_prose(), "prose"), "code": measure(load_code(), "code")}
    json.dump(out, open(os.path.join(RES, "demand.json"), "w"), indent=1)
    for k in out:
        print(k, out[k]["fits"], out[k]["n_docs"], out[k]["n_tokens"], out[k]["vocab"])
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, k in enumerate(["prose", "code"]):
        r = out[k]["rows"]; T = [x["T"] for x in r]
        for key, mk in [("N_bi", "s"), ("N_rep", "o"), ("N_ctx", "^")]:
            f = out[k]["fits"][key]
            ax[0].loglog(T, [x[key] for x in r], mk, color=f"C{i}", ms=4)
            ax[0].loglog(T, [f["c"] * t ** f["beta"] for t in T], "-", color=f"C{i}", lw=.8,
                         label=f"{k} {key}: beta={f['beta']:.2f}")
        ax[1].semilogx(T, [x["rho"] for x in r], "o-", color=f"C{i}", label=k)
    ax[0].set_xlabel("context length T (tokens)"); ax[0].set_ylabel("distinct associations"); ax[0].legend(fontsize=6)
    ax[0].set_title("(a) recall demand vs context length")
    ax[1].set_xlabel("context length T"); ax[1].set_ylabel("fraction of tokens"); ax[1].legend()
    ax[1].set_title("(b) context-only recall opportunities rho(T)")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_demand.png"), dpi=130)


if __name__ == "__main__":
    main()
