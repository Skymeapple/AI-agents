"""Monte Carlo over parameter priors for both financial-channel variants.
Outputs: results/mc_summary.json, fig2_fan.png, fig3_binding.png, fig4_slowdown.png
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from params import sample
from model import simulate, summarize, CHANNELS

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
N = 20_000
CLAT_ALT = "--clat-low" in sys.argv     # sensitivity: Erdil & Schneider-Joseph's 1e28-1e31 range
NO_LAT = "--no-lat" in sys.argv         # sensitivity: remove latency ceiling entirely


def rev_sample(n, rng):
    return dict(
        gR0=np.log(2.5) + (np.log(4.0) - np.log(2.5)) * rng.random(n),  # [S] combined OpenAI+Anthropic revenue 3x (2024), >4x (2025), Epoch
        gRinf=0.05 + 0.15 * rng.random(n),                              # [A] mature growth 5-20%/yr
        tau=1.5 + 4.5 * rng.random(n),                                  # [A] e-folding time of revenue-growth decay
        lam_max=np.exp(np.log(1.0) + np.log(4.0) * rng.random(n)),      # [A] investment may outrun revenue by up to 4x
    )


def q(a, ps=(5, 25, 50, 75, 95)):
    a = np.asarray(a, float)
    return {str(x): float(np.percentile(a, x)) for x in ps}


def run(variant, seed):
    rng = np.random.default_rng(seed)
    p = sample(N, rng)
    if CLAT_ALT:
        p["log10_Clat"] = 28.0 + 3.0 * rng.random(N)
    if NO_LAT:
        p["log10_Clat"] = np.full(N, 60.0)
    rev = rev_sample(N, rng) if variant == "revenue" else None
    sim = simulate(p, variant=variant, rev=rev)
    s = summarize(sim)
    res = {}
    for yr in (2028, 2030, 2035, 2040, 2045):
        res[f"log10C_{yr}"] = q(s[f"log10C_{yr}"])
        b = s[f"bind_{yr}"]
        res[f"bind_{yr}"] = {c: float(np.mean(b == i)) for i, c in enumerate(CHANNELS)}
    for th in ("slow3", "slow2"):
        v = s[th]
        res[th] = q(np.where(np.isinf(v), 2100, v))
        res[th + "_never_by_2045"] = float(np.mean(np.isinf(v)))
    res["g12_median_path"] = {f"{tt:.2f}": float(np.exp(np.median(s["g12"][:, i])))
                              for i, tt in enumerate(s["tt"]) if abs(tt * 2 - round(tt * 2)) < 1e-6}
    # first binding constraint: channel that first caps growth (first time the binding channel's
    # growth falls under 2x, attribute to channel binding at that moment)
    k = 12
    idx = np.clip(np.searchsorted(s["tt"], np.where(np.isinf(s["slow2"]), 1e9, s["slow2"])), 0, len(s["tt"]) - 1)
    bind_at_slow = sim["bind"][np.arange(N), idx + k]
    res["bind_at_slow2"] = {c: float(np.mean(bind_at_slow[~np.isinf(s["slow2"])] == i)) for i, c in enumerate(CHANNELS)}
    # sanity: fraction of samples where 2025 point is power- or latency-limited (inconsistent with anchor)
    res["frac_not_financial_at_t0"] = float(np.mean(sim["bind"][:, 0] != 0))
    return sim, s, res


results, sims = {}, {}
for v, seed in (("exogenous", 11), ("revenue", 12)):
    sim, s, res = run(v, seed)
    results[v] = res; sims[v] = (sim, s)
suffix = "_clatlow" if CLAT_ALT else ("_nolat" if NO_LAT else "")
json.dump(results, open(os.path.join(OUT, f"mc_summary{suffix}.json"), "w"), indent=2)
for v in results:
    r = results[v]
    print(v, "C2030 median 1e%.2f" % r["log10C_2030"]["50"], "C2035 1e%.2f" % r["log10C_2035"]["50"],
          "slow2 median %.1f [%.1f,%.1f]" % (r["slow2"]["50"], r["slow2"]["5"], r["slow2"]["95"]),
          "bind@slow2", r["bind_at_slow2"], "bind2030", r["bind_2030"], "bad t0", r["frac_not_financial_at_t0"])
if CLAT_ALT or NO_LAT:
    sys.exit(0)

# ---------- figures ----------
colors = {"financial": "#DD8452", "power": "#55A868", "latency": "#8172B2"}
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, v in zip(axes, ("exogenous", "revenue")):
    sim, s = sims[v]
    t, L = sim["t"], sim["logC"] / np.log(10)
    for lo, hi, a in ((5, 95, .2), (25, 75, .35)):
        ax.fill_between(t, np.percentile(L, lo, 0), np.percentile(L, hi, 0), color="#4C72B0", alpha=a, lw=0)
    ax.plot(t, np.median(L, 0), color="#4C72B0", lw=2, label="median")
    ax.plot(t, np.log10(5e26) + (t - 2025.5) * np.log10(4.5), "k--", lw=1, label="4.5x/yr extrapolation")
    ax.plot(t, np.log10(5e26) + (t - 2025.5) * np.log10(1.38), "k:", lw=1, label="secular trend (1.38x/yr)")
    ax.set_title(f"variant: {v}", fontsize=10); ax.set_xlabel("year"); ax.set_ylim(26, 33)
    ax.grid(alpha=.3)
axes[0].set_ylabel("log10 largest-run training FLOP"); axes[0].legend(fontsize=8, loc="upper left")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig2_fan.png"), dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, v in zip(axes, ("exogenous", "revenue")):
    sim, s = sims[v]
    t = sim["t"]
    bottom = np.zeros_like(t)
    for i, c in enumerate(CHANNELS):
        frac = (sim["bind"] == i).mean(0)
        ax.fill_between(t, bottom, bottom + frac, color=colors[c], label=c, lw=0)
        bottom += frac
    ax.set_title(f"binding constraint share, variant: {v}", fontsize=10); ax.set_xlabel("year")
axes[0].set_ylabel("fraction of samples"); axes[0].legend(fontsize=8, loc="lower left")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig3_binding.png"), dpi=150)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
for v, ls in (("exogenous", "-"), ("revenue", "--")):
    sim, s = sims[v]
    axes[0].plot(s["tt"], np.exp(np.median(s["g12"], 0)), ls, color="#4C72B0", label=f"median, {v}")
    axes[0].fill_between(s["tt"], np.exp(np.percentile(s["g12"], 5, 0)), np.exp(np.percentile(s["g12"], 95, 0)),
                         color="#4C72B0", alpha=.12)
    sl = s["slow2"]; sl = sl[~np.isinf(sl)]
    axes[1].hist(sl, bins=np.arange(2026, 2046, .5), histtype="step", lw=2, ls=ls, label=v, density=True)
axes[0].set_yscale("log"); axes[0].set_yticks([1, 1.5, 2, 3, 4, 5]); axes[0].set_yticklabels(["1", "1.5", "2", "3", "4", "5"])
axes[0].set_ylabel("trailing 12-month growth factor"); axes[0].set_xlabel("year"); axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)
axes[1].set_xlabel("first year growth falls below 2x/yr"); axes[1].set_ylabel("density"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig4_slowdown.png"), dpi=150)
