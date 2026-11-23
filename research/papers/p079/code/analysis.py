"""Affordability-lag decomposition of the AI divide into income, language (tokenization) and price terms.

Inputs : results/premium.csv (from tokenize_udhr.py), data/wb_*.json (World Bank WDI),
         code/country_language.csv (author-coded predominant language per country),
         code/external_flores_o200k.csv (premiums reported by an independent FLORES-based study).
Outputs: results/*.csv, results/*.json, results/fig*.png
"""
import json, os, csv
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA, RES, CODE = ROOT + "/data", ROOT + "/results", ROOT + "/code"
rng = np.random.default_rng(20261122)
OUT = {}

# ---------------- load premiums ----------------
with open(RES + "/premium.csv") as f:
    rd = csv.reader(f); hdr = next(rd)
    TOKS = hdr[2:]
    P = {r[0]: dict(zip(TOKS, map(float, r[2:]))) for r in rd}
    f.seek(0); next(rd)
GEN = {  # release order / generation pairs (old -> new) from the same developer
    "OpenAI": ["p50k_base", "cl100k_base", "o200k_base"],
    "Meta": ["llama3", "llama4"],
    "Google": ["gemma1", "gemma3"],
    "Mistral": ["mistral_v1", "tekken"],
}
CURRENT = ["o200k_base", "llama4", "gemma3", "tekken", "deepseek_v3"]  # current-generation tokenizers
LABEL = {"p50k_base": "OpenAI p50k (GPT-3)", "cl100k_base": "OpenAI cl100k (GPT-4)", "o200k_base": "OpenAI o200k (GPT-4o/5)",
         "llama3": "Llama 3", "llama4": "Llama 4", "tekken": "Mistral Tekken", "claude_legacy": "Claude (legacy, pre-3)",
         "deepseek_v3": "DeepSeek-V3", "mistral_v1": "Mistral v1 (SPM 32k)", "gemma1": "Gemma 1/2", "gemma3": "Gemma 3"}

# ---------------- 1. validation against independent FLORES measurements ----------------
ext = list(csv.DictReader(open(CODE + "/external_flores_o200k.csv")))
xs, ys, names = [], [], []
for r in ext:
    if r["udhr"] in P:
        xs.append(float(r["premium_o200k"])); ys.append(P[r["udhr"]]["o200k_base"]); names.append(r["language"])
xs, ys = np.array(xs), np.array(ys)
sp = stats.spearmanr(xs, ys); pr = stats.pearsonr(np.log(xs), np.log(ys))
OUT["validation"] = {"n": len(xs), "spearman": sp.statistic, "spearman_p": sp.pvalue, "pearson_log": pr.statistic,
                     "median_ratio_udhr_over_flores": float(np.median(ys / xs)),
                     "pairs": dict(zip(names, [[float(a), float(b)] for a, b in zip(xs, ys)]))}
mask = np.array([n != "Yoruba" for n in names])
sp2 = stats.spearmanr(xs[mask], ys[mask])
OUT["validation"]["spearman_excl_yoruba"] = sp2.statistic
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.scatter(xs, ys, s=22, color="#2a6fbb")
for a, b, n in zip(xs, ys, names):
    ax.annotate(n, (a, b), fontsize=7, xytext=(3, 2), textcoords="offset points")
lim = [1, max(xs.max(), ys.max()) * 1.1]
ax.plot(lim, lim, ls="--", color="grey", lw=0.8); ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("o200k premium, FLORES-200 (external study)"); ax.set_ylabel("o200k premium, UDHR (this paper)")
ax.set_title(f"Validation: Spearman rho={sp.statistic:.2f} (n={len(xs)})", fontsize=9)
fig.tight_layout(); fig.savefig(RES + "/fig1_validation.png", dpi=160); plt.close(fig)

# ---------------- 2. translation-variant noise ----------------
pairs = [("pes_1", "pes_2"), ("ron_1953", "ron_1993"), ("hau_NE", "hau_NG"), ("hat_kreyol", "hat_popular"),
         ("nya_chechewa", "nya_chinyanja"), ("aka_asante", "aka_akuapem"), ("por_BR", "por_PT")]
noise = []
for a, b in pairs:
    if a in P and b in P:
        d = [abs(np.log(P[a][t] / P[b][t])) for t in TOKS]
        noise.append((a, b, float(np.median(d))))
OUT["variant_noise"] = {"pairs": noise, "median_abs_log_diff": float(np.median([n[2] for n in noise]))}
# cross-tokenizer comparisons cancel content: the ratio pi_T1/pi_T2 for the same text is content-free.

# ---------------- 3. countries ----------------
def wb(ind):
    d = json.load(open(f"{DATA}/wb_{ind}.json"))[1]
    out = {}
    for x in d:  # keep most recent non-null observation
        if x["value"] is not None and (x["countryiso3code"] not in out or x["date"] > out[x["countryiso3code"]][1]):
            out[x["countryiso3code"]] = (x["value"], x["date"])
    return out
gni, gnippp, pop = wb("NY.GNP.PCAP.CD"), wb("NY.GNP.PCAP.PP.CD"), wb("SP.POP.TOTL")
meta = {c["id"]: c for c in json.load(open(DATA + "/wb_countries.json"))[1]}
C = []
for r in csv.DictReader(open(CODE + "/country_language.csv")):
    i = r["iso3"]
    if i in gni and i in pop and int(gni[i][1]) >= 2020:
        C.append(dict(iso3=i, lang=r["lang"], gni=gni[i][0], gni_year=gni[i][1], gnippp=gnippp.get(i, (np.nan,))[0],
                      pop=pop[i][0], income=meta[i]["incomeLevel"]["value"], name=meta[i]["name"]))
    else:
        print("excluded (no WB GNI/pop data since 2020):", i)
OUT["n_countries"] = len(C)
OUT["gni_years"] = sorted({c["gni_year"] for c in C})
lnY = np.log([c["gni"] for c in C]); W = np.array([c["pop"] for c in C], float)

# ---------------- 4. income gradient of the language premium, per tokenizer ----------------
def ols_slope(x, y, w=None):
    if w is None: w = np.ones_like(x)
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    return np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2)
grad = {}
for t in TOKS:
    y = np.log([P[c["lang"]][t] for c in C])
    b = ols_slope(lnY, y); bw = ols_slope(lnY, y, W)
    boots = []
    for _ in range(4000):
        k = rng.integers(0, len(C), len(C)); boots.append(ols_slope(lnY[k], y[k]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # population-weighted mean premium among low+lower-middle income vs high income countries
    lowmask = np.array([c["income"] in ("Low income", "Lower middle income") for c in C])
    himask = np.array([c["income"] == "High income" for c in C])
    gm = lambda m: float(np.exp(np.average(y[m], weights=W[m])))
    grad[t] = dict(slope=float(b), ci=[float(lo), float(hi)], slope_popw=float(bw),
                   geo_mean_premium_all=float(np.exp(y.mean())), popw_premium_LIC_LMIC=gm(lowmask), popw_premium_HIC=gm(himask),
                   share_countries_premium_gt2=float(np.mean(np.exp(y) > 2)),
                   pop_share_premium_gt2=float(W[np.exp(y) > 2].sum() / W.sum()))
OUT["income_gradient"] = grad

order = ["p50k_base", "claude_legacy", "mistral_v1", "cl100k_base", "llama3", "gemma1", "tekken", "o200k_base", "deepseek_v3", "llama4", "gemma3"]
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for k, t in enumerate(order):
    g = grad[t]; c = "#c0392b" if t in CURRENT else "#7f8c8d"
    ax.errorbar(k, g["slope"], yerr=[[g["slope"] - g["ci"][0]], [g["ci"][1] - g["slope"]]], fmt="o", color=c, capsize=3)
ax.axhline(0, color="k", lw=0.6)
ax.set_xticks(range(len(order))); ax.set_xticklabels([LABEL[t] for t in order], rotation=40, ha="right", fontsize=7)
ax.set_ylabel("d ln(premium) / d ln(GNI pc)")
ax.set_title(f"Income gradient of the tokenization premium ({len(C)} countries, 95% bootstrap CI)\nred = current-generation tokenizers", fontsize=8.5)
fig.tight_layout(); fig.savefig(RES + "/fig2_income_gradient.png", dpi=160); plt.close(fig)

# ---------------- 5. progressivity of tokenizer upgrades ----------------
prog = {}
langs = sorted({c["lang"] for c in C})
lang_inc = {l: np.log(np.average([c["gni"] for c in C if c["lang"] == l], weights=[c["pop"] for c in C if c["lang"] == l])) for l in langs}
for dev, seq in GEN.items():
    for a, b in zip(seq[:-1], seq[1:]):
        x0 = np.array([np.log(P[l][a]) for l in langs]); x1 = np.array([np.log(P[l][b]) for l in langs])
        inc = np.array([lang_inc[l] for l in langs]); d = x1 - x0
        X = np.column_stack([np.ones_like(inc), inc, x0])
        beta, *_ = np.linalg.lstsq(X, d, rcond=None)
        bs = []
        for _ in range(4000):
            k = rng.integers(0, len(langs), len(langs)); bb, *_ = np.linalg.lstsq(X[k], d[k], rcond=None); bs.append(bb)
        bs = np.array(bs)
        r_inc = stats.spearmanr(inc, d)
        # share of the cross-language log-premium dispersion removed
        prog[f"{a}->{b}"] = dict(developer=dev, mean_dlog=float(d.mean()), spearman_income_vs_change=float(r_inc.statistic),
                                spearman_p=float(r_inc.pvalue),
                                beta_income=float(beta[1]), beta_income_ci=[float(x) for x in np.percentile(bs[:, 1], [2.5, 97.5])],
                                beta_initial=float(beta[2]), beta_initial_ci=[float(x) for x in np.percentile(bs[:, 2], [2.5, 97.5])],
                                sd_log_before=float(x0.std()), sd_log_after=float(x1.std()),
                                n_languages=len(langs), n_worse=int((d > 0.01).sum()))
OUT["upgrade_progressivity"] = prog

# ---------------- 6. affordability-lag decomposition ----------------
ref = next(c for c in C if c["iso3"] == "USA")
G = {"2x/yr (frontier-tier, slow)": np.log(2), "9x/yr (Epoch lower bound)": np.log(9), "50x/yr (Epoch median)": np.log(50)}
dec = []
for c in C:
    pi_o = P[c["lang"]]["o200k_base"]; pi_best = min(P[c["lang"]][t] for t in CURRENT)
    best_t = min(CURRENT, key=lambda t: P[c["lang"]][t])
    pi_worst_cur = max(P[c["lang"]][t] for t in CURRENT)
    inc = np.log(ref["gni"] / c["gni"]); lang_o = np.log(pi_o); lang_b = np.log(pi_best)
    row = dict(iso3=c["iso3"], name=c["name"], lang=c["lang"], income_group=c["income"], gni=c["gni"], pop=c["pop"],
               premium_o200k=pi_o, premium_best_current=pi_best, best_tokenizer=best_t, premium_worst_current=pi_worst_cur,
               premium_cl100k=P[c["lang"]]["cl100k_base"],
               log_income_gap=inc, log_lang_o200k=lang_o, log_lang_best=lang_b,
               lang_share_o200k=lang_o / (inc + lang_o) if inc + lang_o > 0 else np.nan,
               lang_share_cl100k=np.log(P[c["lang"]]["cl100k_base"]) / (inc + np.log(P[c["lang"]]["cl100k_base"])) if inc > 0 else np.nan)
    for gname, g in G.items():
        row[f"lag_years_income@{gname.split()[0]}"] = inc / g
        row[f"lag_years_lang_o200k@{gname.split()[0]}"] = lang_o / g
    dec.append(row)
keys = list(dec[0].keys())
with open(RES + "/country_decomposition.csv", "w") as f:
    w = csv.DictWriter(f, keys); w.writeheader(); [w.writerow(r) for r in dec]
lic = [r for r in dec if r["income_group"] in ("Low income", "Lower middle income")]
OUT["decomposition_summary"] = {
    "median_lang_share_o200k_LIC_LMIC": float(np.median([r["lang_share_o200k"] for r in lic])),
    "median_lang_share_cl100k_LIC_LMIC": float(np.median([r["lang_share_cl100k"] for r in lic])),
    "median_lang_share_best_LIC_LMIC": float(np.median([r["log_lang_best"] / (r["log_income_gap"] + r["log_lang_best"]) for r in lic])),
    "top5_lang_share_o200k": sorted([(round(r["lang_share_o200k"], 3), r["name"], r["lang"]) for r in lic], reverse=True)[:5],
    "max_lang_share": max((r["lang_share_o200k"], r["name"]) for r in lic),
    "n_LIC_LMIC": len(lic),
    "median_income_lag_years": {g: float(np.median([r[f"lag_years_income@{g.split()[0]}"] for r in lic])) for g in G},
    "median_lang_lag_years_o200k": {g: float(np.median([r[f"lag_years_lang_o200k@{g.split()[0]}"] for r in lic])) for g in G},
    "median_log_gain_switch_o200k_to_best": float(np.median([np.log(r["premium_o200k"] / r["premium_best_current"]) for r in lic])),
    "median_log_gain_switch_worst_to_best_current": float(np.median([np.log(r["premium_worst_current"] / r["premium_best_current"]) for r in lic])),
    "median_log_gain_cl100k_to_best": float(np.median([np.log(r["premium_cl100k"] / r["premium_best_current"]) for r in lic])),
}
# figure: stacked log-gap bars for LIC/LMIC + selected others
sel = sorted([r for r in dec if r["iso3"] != "USA"], key=lambda r: r["log_income_gap"] + r["log_lang_o200k"], reverse=True)[:30]
fig, ax = plt.subplots(figsize=(6.6, 6.2))
yy = np.arange(len(sel))
ax.barh(yy, [r["log_income_gap"] for r in sel], color="#34495e", label="income gap ln(GNI_US/GNI_c)")
ax.barh(yy, [r["log_lang_o200k"] for r in sel], left=[r["log_income_gap"] for r in sel], color="#e67e22", label="language premium ln(pi), o200k")
ax.barh(yy, [r["log_lang_best"] - r["log_lang_o200k"] for r in sel], left=[r["log_income_gap"] + r["log_lang_o200k"] for r in sel],
        color="#27ae60", alpha=0.8, label="part of premium removed by best current tokenizer")
ax.set_yticks(yy); ax.set_yticklabels([f'{r["name"][:18]} ({r["lang"].split("_")[0]})' for r in sel], fontsize=7); ax.invert_yaxis()
sec = ax.secondary_xaxis("top", functions=(lambda x: x / np.log(9), lambda y: y * np.log(9)))
sec.set_xlabel("years of price decline at 9x/yr", fontsize=8)
ax.set_xlabel("log affordability gap vs. United States (natural log units)")
ax.legend(fontsize=7, loc="lower right"); fig.tight_layout(); fig.savefig(RES + "/fig3_decomposition.png", dpi=160); plt.close(fig)

# ---------------- 7. affordability frontier: population priced out vs bundle cost ----------------
costs = np.logspace(-1.5, 2, 300)  # monthly English-equivalent bundle cost, USD
thr = 0.02  # A4AI-style: <= 2% of monthly GNI per capita
popT = W.sum()
def priced_out(premkey):
    out = []
    for k in costs:
        m = np.array([k * (P[c["lang"]][premkey] if premkey else 1.0) > thr * c["gni"] / 12 for c in C])
        out.append(W[m].sum() / popT)
    return np.array(out)
po_eng, po_o, po_cl = priced_out(None), priced_out("o200k_base"), priced_out("cl100k_base")
best_prem = {c["lang"]: min(P[c["lang"]][t] for t in CURRENT) for c in C}
po_best = np.array([W[np.array([k * best_prem[c["lang"]] > thr * c["gni"] / 12 for c in C])].sum() / popT for k in costs])
fig, ax = plt.subplots(figsize=(5.8, 3.8))
ax.plot(costs, po_eng, label="no language premium (English pricing)", color="#34495e")
ax.plot(costs, po_cl, label="local language, cl100k (2023)", color="#c0392b")
ax.plot(costs, po_o, label="local language, o200k (2024-)", color="#e67e22")
ax.plot(costs, po_best, label="local language, best current tokenizer", color="#27ae60", ls="--")
ax.set_xscale("log"); ax.set_xlabel("monthly bundle cost in English (USD)"); ax.set_ylabel("population share priced out")
ax.set_title(f"Share of {len(C)}-country population for whom bundle > 2% of monthly GNI pc", fontsize=8.5)
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(RES + "/fig4_affordability.png", dpi=160); plt.close(fig)
def at(k, arr): return float(np.interp(np.log(k), np.log(costs), arr))
OUT["affordability"] = {f"bundle_${k}": {"eng": at(k, po_eng), "cl100k": at(k, po_cl), "o200k": at(k, po_o), "best": at(k, po_best)}
                        for k in [1, 2, 5, 10, 20]}
OUT["affordability"]["total_pop_billion"] = float(popT / 1e9)
# sensitivity: users in countries with an official colonial-era lingua franca write in it (pi of eng/fra/por/spa)
LF = {"IND": "eng", "PAK": "eng", "NGA": "eng", "PHL": "eng", "KEN": "eng", "TZA": "eng", "UGA": "eng", "GHA": "eng", "ZAF": "eng",
      "ZWE": "eng", "ZMB": "eng", "MWI": "eng", "RWA": "eng", "BWA": "eng", "LSO": "eng", "SEN": "fra", "MLI": "fra", "BFA": "fra",
      "TGO": "fra", "NER": "fra", "COD": "fra", "MDG": "fra", "HTI": "fra", "BDI": "fra", "MAR": "fra", "DZA": "fra"}
def priced_out_lf(premkey):
    out = []
    for k in costs:
        m = np.array([k * P[LF.get(c["iso3"], c["lang"])][premkey] > thr * c["gni"] / 12 for c in C])
        out.append(W[m].sum() / popT)
    return np.array(out)
po_lf_cl, po_lf_o = priced_out_lf("cl100k_base"), priced_out_lf("o200k_base")
OUT["affordability_lingua_franca"] = {f"bundle_${k}": {"eng": at(k, po_eng), "cl100k": at(k, po_lf_cl), "o200k": at(k, po_lf_o)} for k in [1, 2, 5, 10, 20]}
# horizontal distance between curves at the median-population point = language tax in log-price (years) units
def cost_at_share(arr, s):  # smallest bundle cost at which priced-out share reaches s
    i = np.argmax(arr >= s); return float(costs[i])
OUT["affordability_cost_at_25pct"] = {"eng": cost_at_share(po_eng, .25), "cl100k": cost_at_share(po_cl, .25), "o200k": cost_at_share(po_o, .25), "best": cost_at_share(po_best, .25)}

# ---------------- 8. reasoning-token dilution of the cost premium (analytical) ----------------
def pi_eff(pi, r): return (pi + r) / (1 + r)  # r = hidden reasoning tokens per visible token, reasoning assumed in a pivot language
dil = {}
for l in ["amh", "mya", "hin", "swh", "yor", "khm"]:
    dil[l] = {f"r={r}": pi_eff(P[l]["o200k_base"], r) for r in [0, 1, 3, 10]}
OUT["reasoning_dilution_o200k"] = dil

# ---------------- 9. within-country gap (India): English vs. major Indian languages ----------------
OUT["india_within"] = {l: {t: P[l][t] for t in ["cl100k_base", "o200k_base", "gemma3", "llama4"]} for l in ["eng", "hin", "ben", "mar", "tam", "guj", "kan", "mal", "pan", "urd"] if l in P}

# ---------------- 10. global distribution over all UDHR languages ----------------
allp = {t: np.array([P[l][t] for l in P]) for t in TOKS}
OUT["all_languages"] = {t: dict(n=len(v), median=float(np.median(v)), p90=float(np.percentile(v, 90)), max=float(v.max()),
                                share_gt2=float(np.mean(v > 2))) for t, v in allp.items()}

# heatmap figure: country languages x tokenizers
L = sorted(langs, key=lambda l: P[l]["o200k_base"])
M = np.log2(np.array([[P[l][t] for t in order] for l in L]))
fig, ax = plt.subplots(figsize=(6.4, 11))
im = ax.imshow(M, aspect="auto", cmap="magma_r", vmin=0, vmax=4)
ax.set_xticks(range(len(order))); ax.set_xticklabels([LABEL[t] for t in order], rotation=50, ha="right", fontsize=7)
ax.set_yticks(range(len(L))); ax.set_yticklabels(L, fontsize=6)
cb = fig.colorbar(im, ax=ax, shrink=0.5); cb.set_label("log2 premium vs English")
fig.tight_layout(); fig.savefig(RES + "/fig5_heatmap.png", dpi=160); plt.close(fig)

json.dump(OUT, open(RES + "/summary.json", "w"), indent=1, default=float)
print(json.dumps({k: OUT[k] for k in ["validation", "variant_noise", "n_countries", "gni_years", "decomposition_summary", "affordability"]}, indent=1, default=float))
for t in order:
    g = grad[t]; print(f"{t:14s} slope={g['slope']:+.3f} [{g['ci'][0]:+.3f},{g['ci'][1]:+.3f}] popw={g['slope_popw']:+.3f} gm={g['geo_mean_premium_all']:.2f} LIC/LMIC={g['popw_premium_LIC_LMIC']:.2f} HIC={g['popw_premium_HIC']:.2f} >2:{g['share_countries_premium_gt2']:.2f} pop>2:{g['pop_share_premium_gt2']:.2f}")
for k, v in prog.items():
    print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()})
print(json.dumps(OUT["reasoning_dilution_o200k"], indent=0))
