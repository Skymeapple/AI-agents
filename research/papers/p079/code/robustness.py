"""Robustness: does the income gradient survive a control for writing system (Latin vs non-Latin script)?
Also language-level (deduplicated) gradient. Output: results/robustness.json"""
import csv, json, os, re, unicodedata
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = {r["lang"]: r for r in csv.DictReader(open(ROOT + "/results/premium.csv"))}
D = list(csv.DictReader(open(ROOT + "/results/country_decomposition.csv")))
def latin_share(lang):
    t = open(f"{ROOT}/data/udhr2/{lang}.txt", encoding="utf-8").read()
    letters = [c for c in t if c.isalpha()]
    return np.mean(["LATIN" in unicodedata.name(c, "") for c in letters])
rng = np.random.default_rng(7)
out = {}
for tok in ["cl100k_base", "o200k_base", "llama4", "gemma3", "tekken", "deepseek_v3"]:
    y = np.log([float(P[r["lang"]][tok]) for r in D]); x = np.log([float(r["gni"]) for r in D])
    nl = np.array([latin_share(r["lang"]) < 0.5 for r in D], float)
    X = np.column_stack([np.ones_like(x), x, nl]); b, *_ = np.linalg.lstsq(X, y, rcond=None)
    bs = [np.linalg.lstsq(X[k], y[k], rcond=None)[0][1] for k in (rng.integers(0, len(y), len(y)) for _ in range(4000))]
    # Latin-script-only subsample
    m = nl == 0; Xl = np.column_stack([np.ones(m.sum()), x[m]]); bl, *_ = np.linalg.lstsq(Xl, y[m], rcond=None)
    bsl = [np.linalg.lstsq(Xl[k], y[m][k], rcond=None)[0][1] for k in (rng.integers(0, m.sum(), m.sum()) for _ in range(4000))]
    # language-level (dedupe): mean ln GNI per language
    langs = sorted({r["lang"] for r in D})
    xl = np.array([np.mean([np.log(float(r["gni"])) for r in D if r["lang"] == l]) for l in langs]); yl = np.log([float(P[l][tok]) for l in langs])
    bll = np.polyfit(xl, yl, 1)[0]
    out[tok] = dict(beta_income_ctrl_script=float(b[1]), ci=[float(v) for v in np.percentile(bs, [2.5, 97.5])], beta_nonlatin=float(b[2]),
                    beta_income_latin_only=float(bl[1]), ci_latin_only=[float(v) for v in np.percentile(bsl, [2.5, 97.5])], n_latin=int(m.sum()),
                    beta_income_language_level=float(bll), n_languages=len(langs))
json.dump(out, open(ROOT + "/results/robustness.json", "w"), indent=1)
for k, v in out.items(): print(k, {a: (round(b, 3) if isinstance(b, float) else ([round(c, 3) for c in b] if isinstance(b, list) else b)) for a, b in v.items()})
