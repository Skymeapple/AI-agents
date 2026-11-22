"""Empirical module: US-PRC frontier gap on the LMArena (Chatbot Arena) leaderboard.

Input : data/scores.json  (daily-ish LMArena snapshots, 2023-05 .. 2025-05,
         from github.com/nakasyou/lmarena-history, a JSON conversion of the
         official lmarena-ai Hugging Face leaderboard pickles)
Output: results/arena_gap_series.csv, results/arena_lag_series.csv,
        results/arena_fit.json, results/fig_arena_gap.png

Every number reported in the paper as "computed" comes from this script.
"""
import json, re, os, datetime as dt
import numpy as np
from scipy.optimize import curve_fit

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "scores.json")
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

# ---------------------------------------------------------------- classification
CN = [r"^chatglm", r"^glm-", r"^qwen", r"^qwq", r"^deepseek", r"^yi-", r"^hunyuan",
      r"^step-", r"^internlm"]
US = [r"^gpt-", r"^chatgpt", r"^o1", r"^o3", r"^o4", r"^claude", r"^gemini", r"^gemma",
      r"^bard", r"^palm", r"^llama", r"^codellama", r"^grok", r"^early-grok", r"^athene",
      r"^nemotron", r"^llama2-70b-steerlm", r"^phi-", r"^amazon-nova", r"^granite",
      r"^dbrx", r"^mpt-", r"^vicuna", r"^koala", r"^alpaca", r"^olmo", r"^pplx",
      r"^snowflake", r"^starling", r"^stripedhyena", r"^gpt4all", r"^dolly",
      r"^fastchat", r"^guanaco", r"^smollm"]
# everything else (Mistral, Cohere, Reka, AI21, TII, Upstage, LAION, Stability,
# HF, WizardLM, OpenChat, anonymous codenames) is excluded from both groups.


def country(m):
    m = m.lower()
    if any(re.search(p, m) for p in CN):
        return "CN"
    if any(re.search(p, m) for p in US):
        return "US"
    return "OTHER"


def todate(s):
    return dt.date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def yearfrac(d):
    return d.year + (d.timetuple().tm_yday - 1) / 365.25


def main():
    d = json.load(open(DATA))
    dates = sorted(d)
    rows = []
    first_seen = {}
    for s in dates:
        txt = d[s]["text"]
        for cat in ("overall", "overall_style_control", "english"):
            if cat not in txt:
                continue
            sc = txt[cat]
            us = {m: v for m, v in sc.items() if country(m) == "US"}
            cn = {m: v for m, v in sc.items() if country(m) == "CN"}
            if not us or not cn:
                continue
            mu, mc = max(us, key=us.get), max(cn, key=cn.get)
            gap = us[mu] - sc[mc]
            rows.append(dict(date=s, cat=cat, top_us=mu, elo_us=us[mu], top_cn=mc,
                             elo_cn=cn[mc], gap=gap,
                             p_us=1 / (1 + 10 ** (-gap / 400))))
        if "overall" in txt:
            for m in txt["overall"]:
                first_seen.setdefault(m, todate(s))

    with open(os.path.join(RES, "arena_gap_series.csv"), "w") as f:
        f.write("date,category,top_us,elo_us,top_cn,elo_cn,gap_elo,p_us_beats_cn\n")
        for r in rows:
            f.write(f"{r['date']},{r['cat']},{r['top_us']},{r['elo_us']:.1f},{r['top_cn']},"
                    f"{r['elo_cn']:.1f},{r['gap']:.1f},{r['p_us']:.4f}\n")

    # ------------------------------------------- time-lag on a fixed rating scale
    final = d[dates[-1]]["text"]["overall"]
    models = [(first_seen[m], final[m], country(m)) for m in final if m in first_seen]
    snap = [todate(s) for s in dates if "overall" in d[s]["text"]]
    lag_rows = []
    for t in snap:
        cn_av = [r for (fs, r, c) in models if c == "CN" and fs <= t]
        us_av = [(fs, r) for (fs, r, c) in models if c == "US" and fs <= t]
        if not cn_av or not us_av:
            continue
        cbest = max(cn_av)
        ubest = max(r for _, r in us_av)
        hits = [fs for (fs, r) in us_av if r >= cbest]
        if hits:
            lag_m = (t - min(hits)).days / 30.44
        else:
            lag_m = 0.0  # CN at or above US frontier on this scale
        lag_rows.append((t, ubest, cbest, ubest - cbest, lag_m))
    with open(os.path.join(RES, "arena_lag_series.csv"), "w") as f:
        f.write("date,us_best_finalscale,cn_best_finalscale,gap_finalscale,lag_months\n")
        for t, u, c, g, l in lag_rows:
            f.write(f"{t},{u:.1f},{c:.1f},{g:.1f},{l:.2f}\n")

    # frontier slope (Elo/yr) on fixed scale: record-setting models (any country)
    recs = sorted([(fs, r) for (fs, r, c) in models if c in ("US", "CN")])
    best = -1e9
    rx, ry = [], []
    for fs, r in recs:
        if r > best:
            best = r
            rx.append(yearfrac(fs)); ry.append(r)
    rx, ry = np.array(rx), np.array(ry)
    slope, icpt = np.polyfit(rx, ry, 1)
    # bootstrap slope
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(2000):
        idx = rng.integers(0, len(rx), len(rx))
        if len(set(rx[idx])) < 3:
            continue
        bs.append(np.polyfit(rx[idx], ry[idx], 1)[0])
    slope_ci = np.percentile(bs, [5, 95])

    # ------------------------------------------- partial-adjustment fit G' = mu - delta G
    def path(t, G0, Gs, delta):
        return Gs + (G0 - Gs) * np.exp(-delta * t)

    fits = {}
    for cat in ("overall", "overall_style_control", "english", "overall_from2024"):
        if cat == "overall_from2024":
            rr = [r for r in rows if r["cat"] == "overall" and r["date"] >= "20240101"]
        else:
            rr = [r for r in rows if r["cat"] == cat]
        if len(rr) < 8:
            continue
        t = np.array([yearfrac(todate(r["date"])) for r in rr])
        g = np.array([r["gap"] for r in rr])
        t0 = t.min(); tt = t - t0
        try:
            p, cov = curve_fit(path, tt, g, p0=[g[0], max(g[-1], 1), 1.0],
                               bounds=([-500, -200, 0.01], [800, 400, 20]), maxfev=20000)
        except Exception as e:
            fits[cat] = {"error": str(e)}; continue
        res = g - path(tt, *p)
        # moving-block bootstrap on residuals (block = 6 snapshots)
        B, L = 1000, 6
        boot = []
        for _ in range(B):
            nb = int(np.ceil(len(res) / L))
            starts = rng.integers(0, len(res) - L + 1, nb)
            e = np.concatenate([res[s:s + L] for s in starts])[:len(res)]
            try:
                pb, _ = curve_fit(path, tt, path(tt, *p) + e, p0=p,
                                  bounds=([-500, -200, 0.01], [800, 400, 20]), maxfev=20000)
                boot.append(pb)
            except Exception:
                pass
        boot = np.array(boot)
        fits[cat] = dict(n=int(len(g)), t_start=float(t0), t_end=float(t.max()),
                         G0=float(p[0]), Gstar=float(p[1]), delta=float(p[2]),
                         halflife_yr=float(np.log(2) / p[2]),
                         delta_ci90=[float(x) for x in np.percentile(boot[:, 2], [5, 95])],
                         Gstar_ci90=[float(x) for x in np.percentile(boot[:, 1], [5, 95])],
                         rmse=float(np.sqrt(np.mean(res ** 2))),
                         first_gap=float(g[0]), last_gap=float(g[-1]),
                         mean_gap_last12m=float(np.mean(g[t >= t.max() - 1])))

    # lag summary
    lags = np.array([l for *_, l in lag_rows])
    lt = np.array([yearfrac(t) for t, *_ in lag_rows])
    # plateau test: linear trend in the within-snapshot gap after 2024-05-01
    plateau = {}
    for cat in ("overall", "english", "overall_style_control"):
        rr = [r for r in rows if r["cat"] == cat and r["date"] >= "20240501"]
        t = np.array([yearfrac(todate(r["date"])) for r in rr]); g = np.array([r["gap"] for r in rr])
        b = np.polyfit(t, g, 1)[0]
        L = 6; bs = []
        for _ in range(2000):
            nb = int(np.ceil(len(t) / L)); st = rng.integers(0, len(t) - L + 1, nb)
            ii = np.concatenate([np.arange(s0, s0 + L) for s0 in st])[:len(t)]
            bs.append(np.polyfit(t[ii], g[ii], 1)[0])
        plateau[cat] = dict(n=int(len(t)), slope_elo_per_yr=float(b),
                            slope_ci90=[float(x) for x in np.percentile(bs, [5, 95])],
                            mean=float(g.mean()), sd=float(g.std()), min=float(g.min()), max=float(g.max()))

    out = dict(plateau_since_2024_05=plateau, frontier_slope_elo_per_yr=float(slope), frontier_slope_ci90=[float(x) for x in slope_ci],
               n_records=int(len(rx)), fits=fits,
               lag_months_mean_all=float(lags.mean()),
               lag_months_mean_last12m=float(lags[lt >= lt.max() - 1].mean()),
               lag_months_last=float(lags[-1]), lag_months_max=float(lags.max()),
               n_snapshots=len(dates), first_snapshot=dates[0], last_snapshot=dates[-1])
    json.dump(out, open(os.path.join(RES, "arena_fit.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))

    # ------------------------------------------- figure
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for cat, col in (("overall", "#1f4e79"), ("overall_style_control", "#c55a11"), ("english", "#7f7f7f")):
        rr = [r for r in rows if r["cat"] == cat]
        if not rr:
            continue
        t = np.array([yearfrac(todate(r["date"])) for r in rr]); g = np.array([r["gap"] for r in rr])
        ax[0].plot(t, g, "o", ms=3, color=col, alpha=.6, label=cat)
        if cat in fits and "delta" in fits[cat]:
            f_ = fits[cat]; tt = np.linspace(t.min(), t.max(), 200)
            ax[0].plot(tt, path(tt - f_["t_start"], f_["G0"], f_["Gstar"], f_["delta"]), "-", color=col)
    ax[0].axhline(0, color="k", lw=.5)
    ax[0].set_xlabel("year"); ax[0].set_ylabel("top-US minus top-PRC Arena rating (within snapshot)")
    ax[0].set_title("(a) Within-snapshot Elo gap and partial-adjustment fit"); ax[0].legend(fontsize=8)
    ax[1].plot(lt, lags, "-o", ms=3, color="#1f4e79")
    ax[1].set_xlabel("year"); ax[1].set_ylabel("lag (months), fixed 2025-05 rating scale")
    ax[1].set_title("(b) Time lag of best PRC model behind US")
    for a in ax:
        a.set_xticks([2023.5, 2024.0, 2024.5, 2025.0, 2025.5])
        a.set_xticklabels(["2023-07", "2024-01", "2024-07", "2025-01", "2025-07"])
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_arena_gap.png"), dpi=150)


if __name__ == "__main__":
    main()
