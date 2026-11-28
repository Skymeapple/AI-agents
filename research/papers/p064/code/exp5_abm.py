"""Stochastic individual-level simulation: (i) checks the general-kernel shortfall formula when
time-to-seniority is heterogeneous; (ii) compares how quickly three observable indicators detect
an extensive (hiring) shock versus an intensive (learning) shock in finite occupations."""
import json
import numpy as np
from scipy.stats import gamma as gamma_dist
from common import *
from model import required_h

mu, g, p = SD_MU, SD_G, P
TAU_MEAN, SHAPE = 5.0, 6.0
dt = 1 / 12
lam = mu + g


def kernel_shortfall(t, xe):
    """General-kernel version of Prop. 2: G(t) = xe * int_0^t lam e^{-lam (t-v)} W(v) dv,
    W = CDF of the survival- and growth-tilted time-to-seniority density."""
    u = np.arange(0, 60, dt)
    k = gamma_dist.pdf(u, SHAPE, scale=TAU_MEAN / SHAPE)
    w = k * np.exp(-(g + mu) * u); w /= w.sum() * dt
    W = np.cumsum(w) * dt
    Wt = np.interp(t, u, W)
    G = np.zeros_like(t)
    for i in range(1, len(t)):
        G[i] = G[i - 1] + dt * lam * (xe * Wt[i - 1] - G[i - 1])
    return G


def run_abm(D0, x, d, seed, warm=100, pre=20, post=40):
    rng = np.random.default_rng(seed)
    # effective h so that the heterogeneous-tau system is balanced: h * p * E[e^{-(g+mu)tau}] = lam
    u = np.arange(0, 60, dt); k = gamma_dist.pdf(u, SHAPE, scale=TAU_MEAN / SHAPE)
    Ew = np.sum(k * np.exp(-(g + mu) * u)) * dt
    h0 = lam / (p * Ew)
    S = D0
    j_hire = np.empty(0); j_tau = np.empty(0); j_ai = np.empty(0, bool)
    n = int((warm + pre + post) / dt)
    t0 = warm + pre
    rec = dict(S=[], D=[], hires=[], promoted_by_cohort={})
    promo = {}   # hire-year -> [hired, promoted within 6y]
    for i in range(n):
        t = i * dt
        D = D0 * np.exp(g * (t - t0))
        ai = t >= t0
        hrate = h0 * (1 - x) if ai else h0
        nh = rng.poisson(hrate * D * dt)
        if nh:
            j_hire = np.concatenate([j_hire, np.full(nh, t)])
            j_tau = np.concatenate([j_tau, rng.gamma(SHAPE, TAU_MEAN / SHAPE, nh)])
            j_ai = np.concatenate([j_ai, np.full(nh, ai)])
            yr = int(np.floor(t)); promo.setdefault(yr, [0, 0]); promo[yr][0] += nh
        # junior exits
        keep = rng.random(len(j_hire)) > (1 - np.exp(-mu * dt))
        j_hire, j_tau, j_ai = j_hire[keep], j_tau[keep], j_ai[keep]
        # graduation
        due = (t - j_hire) >= j_tau
        if due.any():
            pp = np.where(j_ai[due], p * (1 - d), p)
            succ = rng.random(due.sum()) < pp
            hy = np.floor(j_hire[due][succ]).astype(int); ages = (t - j_hire[due][succ])
            for y_, a_ in zip(hy, ages):
                if a_ <= 6.0:
                    promo.setdefault(y_, [0, 0]); promo[y_][1] += 1
            S += succ.sum()
            j_hire, j_tau, j_ai = j_hire[~due], j_tau[~due], j_ai[~due]
        S -= rng.binomial(S, 1 - np.exp(-mu * dt))
        if i % 12 == 11:
            rec["S"].append(S); rec["D"].append(D)
        rec["hires"].append(nh)
    years = np.arange(len(rec["S"])) + 1 - t0       # year index relative to shock (end-of-year)
    S_ = np.array(rec["S"], float); D_ = np.array(rec["D"])
    hires = np.array(rec["hires"]).reshape(-1, 12).sum(1)
    entrant_rate = hires / np.concatenate([[D0], S_[:-1]])
    cyears = np.arange(int(warm + pre + post))
    comp = np.array([promo.get(int(y), [1, 0])[1] / max(promo.get(int(y), [1, 0])[0], 1) for y in cyears])
    # completion of cohort y is observable at end of year y+7 (6-year window + 1 reporting year)
    comp_obs_year = cyears + 7 - t0
    return dict(years=years, ratio=S_ / D_, entrant_rate=entrant_rate, comp=comp, comp_obs_year=comp_obs_year)


def ma3(v):
    """trailing 3-period mean (flows are noisy; analysts pool years)"""
    out = np.convolve(v, np.ones(3) / 3, mode="full")[:len(v)]
    out[:2] = v[:2]
    return out


def detect(series, obs_year, pre_mask, post_mask, k=3.0, consec=2):
    """first year (relative to shock) where series falls >k sd below its pre-shock mean for `consec` years"""
    m, s = series[pre_mask].mean(), series[pre_mask].std(ddof=1)
    flag = series < m - k * s
    idx = np.where(post_mask)[0]
    for a in range(len(idx) - consec + 1):
        if all(flag[idx[a:a + consec]]):
            return float(obs_year[idx[a]])
    return np.nan


results = {}
mean_paths = {}
for D0 in [500, 5000]:
    for (x, d, lab) in [(0.19, 0.0, "extensive x=0.19"), (0.0, 0.17, "intensive d=0.17")]:
        det = dict(senior_ratio=[], entrant_rate=[], cohort_completion=[])
        paths = []
        for rep in range(100):
            r = run_abm(D0, x, d, seed=1000 * D0 + rep + (0 if x > 0 else 7))
            y = r["years"]
            pre = (y <= 0) & (y > -20); post = y > 0
            # headcount ratio vs its pre-shock level
            det["senior_ratio"].append(detect(r["ratio"], y, pre, post))
            det["entrant_rate"].append(detect(ma3(r["entrant_rate"]), y, pre, post))
            cy = r["comp_obs_year"]; valid = cy <= y.max()
            comp = ma3(r["comp"])[valid]; cy = cy[valid]
            prec = (cy <= 0) & (cy > -12); postc = (cy > 0) & (np.arange(len(cy)) >= 0)
            # only cohorts fully observed and hired after the shock count as post
            hired_after = (cy - 7) >= 0
            det["cohort_completion"].append(detect(comp, cy, prec, postc & hired_after))
            paths.append(1 - r["ratio"] / r["ratio"][(y <= 0) & (y > -20)].mean())
        key = f"D0={D0}|{lab}"
        results[key] = {k: dict(median=float(np.nanmedian(v)), p10=float(np.nanpercentile(v, 10)) if np.isfinite(v).any() else None,
                                p90=float(np.nanpercentile(v, 90)) if np.isfinite(v).any() else None,
                                share_detected_within_30y=float(np.mean(np.isfinite(v))),
                                share_detected_within_10y=float(np.mean(np.nan_to_num(np.array(v), nan=1e9) <= 10)))
                        for k, v in det.items()}
        mean_paths[key] = (y, np.mean(paths, 0), np.percentile(paths, 5, 0), np.percentile(paths, 95, 0))
        print(key, json.dumps(results[key]))

# kernel formula check against the large-occupation mean path (extensive shock)
y, m, lo, hi = mean_paths["D0=5000|extensive x=0.19"]
tt = np.arange(0, 40, dt)
Gk = kernel_shortfall(tt, 0.19)
post = y > 0
Gk_y = np.interp(y[post], tt, Gk)
results["kernel_formula_max_abs_err_D5000"] = float(np.abs(m[post][:35] - Gk_y[:35]).max())
from model import shortfall_path
results["kernel_vs_fixedtau_max_diff"] = float(np.abs(Gk - shortfall_path(tt, 0.19, TAU_MEAN, mu, g)).max())
print("kernel err", results["kernel_formula_max_abs_err_D5000"], "kernel vs fixed", results["kernel_vs_fixedtau_max_diff"])
json.dump(results, open(os.path.join(RES, "exp5_abm.json"), "w"), indent=1)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
for j, key in enumerate(["D0=500|extensive x=0.19", "D0=5000|extensive x=0.19"]):
    y, m, lo, hi = mean_paths[key]
    ax[0].fill_between(y, 100 * lo, 100 * hi, color=C[j], alpha=0.18, lw=0)
    ax[0].plot(y, 100 * m, color=C[j], lw=1.6, label=f"ABM mean, {key.split('|')[0].replace('D0=', 'N=')} seniors (band: 5-95%)")
ax[0].plot(tt, 100 * Gk, color=INK, lw=1, ls=":", label="general-kernel formula")
ax[0].set_xlim(-10, 35); ax[0].set_ylim(-5, 25); ax[0].axvline(0, color=INK2, lw=0.8, ls="--")
ax[0].set_xlabel("years since hiring shock"); ax[0].set_ylabel("senior shortfall, % of demand")
ax[0].set_title("A. Stochastic occupations, heterogeneous training time", fontsize=9, loc="left")
ax[0].legend(fontsize=7, loc="upper left")
labels = ["senior_ratio", "entrant_rate", "cohort_completion"]
nice = ["senior headcount", "entrant rate", "cohort completion"]
groups = ["D0=500|extensive x=0.19", "D0=5000|extensive x=0.19", "D0=500|intensive d=0.17", "D0=5000|intensive d=0.17"]
gl = ["hiring shock\nN=500", "hiring shock\nN=5000", "learning shock\nN=500", "learning shock\nN=5000"]
w = 0.26
for k, lab in enumerate(labels):
    sh = [results[gk][lab]["share_detected_within_10y"] for gk in groups]
    xs = np.arange(4) + (k - 1) * w
    ax[1].bar(xs, [100 * v for v in sh], w * 0.9, color=C[k], label=nice[k])
    for xx, ss in zip(xs, sh):
        ax[1].text(xx, 100 * ss + 1.5, f"{100*ss:.0f}", ha="center", fontsize=7, color=INK2)
ax[1].set_xticks(np.arange(4)); ax[1].set_xticklabels(gl, fontsize=7.5)
ax[1].set_ylabel("% of 100 runs flagged within 10 years\n(3-sd rule, 2 consecutive years)")
ax[1].set_title("B. Which indicator sees the shock first?", fontsize=9, loc="left")
ax[1].legend(fontsize=7, loc="upper left"); ax[1].grid(axis="x", visible=False)
fig.tight_layout(); save(fig, "fig5_abm_indicators.png")
