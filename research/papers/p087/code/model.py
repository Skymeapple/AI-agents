"""
Part B: Make-or-buy model of custom AI accelerators versus a merchant GPU incumbent.

Per unit of delivered compute (one "NVIDIA-equivalent", NVE = throughput of one frontier
NVIDIA chip over its life T), a buyer pays
   buy :  P + T*o                                  with  P = C/(1-m)
   make:  [kappa*10^(g_usd*ell)*C*(1+mu) + T*o*10^(g_w*ell)/eta] / (sigma*D)  +  F/Q
where C = NVIDIA unit COGS, m = NVIDIA gross margin, o = opex per NVIDIA chip-year,
kappa = ASIC manufacturing cost per peak FLOP relative to NVIDIA COGS per peak FLOP,
mu = design-partner/foundry-services markup, eta = ASIC perf/W relative to NVIDIA (measured, Part A),
ell = ASIC time-to-market lag (yrs) vs the NVIDIA part it competes with,
g_usd, g_w = log10 growth rates of FLOP/$ and FLOP/W,
sigma = realised software efficiency relative to the CUDA stack,
D = drift-adjusted usefulness  = phi + (1-phi)*(1-exp(-lam*T))/(lam*T),
F = fixed cost per generation (NRE + compiler/runtime/kernels team), Q = NVE bought per generation.

Define tau = T*o/C and
   Gamma = [kappa*10^(g_usd*ell)*(1+mu) + tau*10^(g_w*ell)/eta] / (sigma*D) - tau.
Proposition 1:  make is optimal for some finite Q  iff  1/(1-m) > Gamma, i.e. m > m0 = 1 - 1/Gamma.
Proposition 2:  threshold  Q*(m) = F / (C*(1/(1-m) - Gamma));
                limit margin for a buyer of size Q:  m*(Q) = 1 - 1/(Gamma + F/(Q*C)), decreasing to m0.
"""
import json, os, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
rng = np.random.default_rng(87)

# ---------------- sourced anchors (see paper, Table 2) ----------------
NV_FY26_REVENUE = 215.938e9          # NVIDIA FY2026 press release (GAAP)
NV_GM_FY25, NV_GM_FY26 = 0.750, 0.711  # NVIDIA FY2026 press release (GAAP gross margin)
# Q3 FY2026 10-Q: direct customers A-D at 22%,15%,13%,11% of quarterly revenue
TOP_SHARES = [0.22, 0.15, 0.13, 0.11]
P_FRONTIER = 40000.0                 # between GB200 ($37k) and GB300 ($43k) list prices in Epoch dataset
G_USD = math.log10(2) / 2.1          # Epoch: FLOP/$ of ML GPUs doubles every 2.1 yrs
cre = json.load(open(os.path.join(RES, "cre_summary.json")))
G_W = cre["nvidia_frontier_growth"]["bf16_flops_per_w"]["g_oom_per_yr"]       # measured, Part A
ETA_MED = cre["group_stats"]["bf16_flops_per_w"]["hyperscaler_asic"]["median_CRE_F0_ci"]  # measured

BASE = dict(m=NV_GM_FY25, P=P_FRONTIER, o=2500.0, T=4.0, kappa=1.0, mu=0.5, eta=ETA_MED[0],
            ell=0.5, sigma=0.85, lam=0.15, phi=0.5, F=1.0e9, g_usd=G_USD, g_w=G_W)


def D(lam, T, phi):
    lam = np.maximum(lam, 1e-9)
    return phi + (1 - phi) * (1 - np.exp(-lam * T)) / (lam * T)


def gamma(p, C):
    tau = p["T"] * p["o"] / C
    num = p["kappa"] * 10 ** (p["g_usd"] * p["ell"]) * (1 + p["mu"]) + tau * 10 ** (p["g_w"] * p["ell"]) / p["eta"]
    return num / (p["sigma"] * D(p["lam"], p["T"], p["phi"])) - tau


def cogs(p):
    return p["P"] * (1 - p["m"])


def m0(p):
    g = gamma(p, cogs(p))
    return np.where(g > 1, 1 - 1 / g, 0.0)


def qstar(p, m=None):
    C = cogs(p)
    m = p["m"] if m is None else m
    s = C * (1 / (1 - m) - gamma(p, C))
    return np.where(s > 0, p["F"] / np.maximum(s, 1e-12), np.inf)


def limit_margin(p, Q):
    C = cogs(p)
    return 1 - 1 / (gamma(p, C) + p["F"] / (Q * C))


# ---------------- buyer population ----------------
def buyer_population(alpha_tail=1.16, n_tail=200):
    """Annual NVE demand per buyer. Top 4 from 10-Q shares; the residual 39% spread over a
    Zipf-like tail with exponent alpha_tail (assumption; varied in sensitivity analysis)."""
    nve_total = NV_FY26_REVENUE * 0.75 / P_FRONTIER  # ~75% of revenue is compute (Compute $162.4B / $215.9B)
    rest = 1 - sum(TOP_SHARES)
    ranks = np.arange(1, n_tail + 1)
    w = ranks ** (-alpha_tail)
    w = w / w.sum() * rest
    w = np.minimum(w, TOP_SHARES[-1])  # tail buyers cannot exceed the 4th largest
    w = w / w.sum() * rest
    shares = np.concatenate([TOP_SHARES, w])
    return shares * nve_total, shares


def contestable(p, shares_q, alpha_stable, lam_s, lam_f, phi_s, phi_f, gen_years=2.0, m=None):
    """Fraction of NVIDIA NVE demand where make beats buy, with each buyer's workload split into a
    stable segment (share alpha_stable; low drift) and a frontier segment (high drift).
    Q per generation = annual demand * gen_years. F is paid separately per segment the buyer enters
    (conservative: no shared NRE between segments)."""
    m = p["m"] if m is None else m
    tot = shares_q.sum()
    lost = 0.0
    for seg_share, lam, phi in ((alpha_stable, lam_s, phi_s), (1 - alpha_stable, lam_f, phi_f)):
        q = dict(p); q["lam"] = lam; q["phi"] = phi
        thr = qstar(q, m)
        Qseg = shares_q * seg_share * gen_years
        lost += Qseg[Qseg > thr].sum() / gen_years
    return lost / tot


def nvidia_uniform_opt(p, shares_q, alpha_stable, lam_s, lam_f, phi_s, phi_f, grid=None):
    """NVIDIA chooses one margin; profit = m/(1-m)*C * retained NVE (COGS C held fixed)."""
    C = cogs(p)
    grid = np.linspace(0.30, 0.85, 111) if grid is None else grid
    best = None
    out = []
    for m in grid:
        lost = contestable(p, shares_q, alpha_stable, lam_s, lam_f, phi_s, phi_f, m=m)
        prof = (m / (1 - m)) * C * (1 - lost) * shares_q.sum()
        out.append((m, lost, prof))
        if best is None or prof > best[2]:
            best = (m, lost, prof)
    return best, out


SEG = dict(alpha_stable=0.6, lam_s=0.10, lam_f=0.60, phi_s=0.6, phi_f=0.3)

RANGES = {  # (low, high) uniform sampling ranges for Monte Carlo; all are stated assumptions
    "m": (NV_GM_FY26, 0.78), "o": (1500, 4000), "T": (3, 6), "kappa": (0.7, 1.4), "mu": (0.2, 1.0),
    "eta": (ETA_MED[1], 1.12), "ell": (0.0, 1.5), "sigma": (0.6, 1.0), "F": (0.3e9, 3.0e9),
    "lam_s": (0.03, 0.25), "lam_f": (0.3, 1.0), "phi_s": (0.4, 0.8), "phi_f": (0.1, 0.5),
    "alpha_stable": (0.3, 0.8), "alpha_tail": (0.8, 1.6),
}


def main():
    out = {"anchors": {"P_frontier": P_FRONTIER, "m_FY25": NV_GM_FY25, "m_FY26": NV_GM_FY26,
                       "C_implied_FY25": P_FRONTIER * (1 - NV_GM_FY25), "g_usd": G_USD, "g_w": G_W,
                       "eta_median_ci": ETA_MED}}
    p = dict(BASE)
    C = cogs(p)
    out["base"] = {"Gamma": float(gamma(p, C)), "m0": float(m0(p)), "D": float(D(p["lam"], p["T"], p["phi"])),
                   "Qstar_NVE_at_m": float(qstar(p)), "Qstar_spend_usd": float(qstar(p) * p["P"]),
                   "Qstar_at_m_FY26": float(qstar(p, NV_GM_FY26))}
    qs, shares = buyer_population()
    out["population"] = {"nve_total_per_yr": float(qs.sum()), "top4_nve_per_yr": [float(x) for x in qs[:4]]}
    out["base"]["limit_margin_top_buyer"] = float(limit_margin(p, qs[0] * 2))
    out["base"]["limit_margin_4th_buyer"] = float(limit_margin(p, qs[3] * 2))
    out["base"]["contestable_share_m_FY25"] = float(contestable(p, qs, **SEG, m=NV_GM_FY25))
    out["base"]["contestable_share_m_FY26"] = float(contestable(p, qs, **SEG, m=NV_GM_FY26))
    best, curve = nvidia_uniform_opt(p, qs, **SEG)
    out["base"]["nvidia_opt_uniform_margin"] = float(best[0])
    out["base"]["nvidia_opt_lost_share"] = float(best[1])
    # segment-specific m0
    for nm, lam, phi in (("stable", SEG["lam_s"], SEG["phi_s"]), ("frontier", SEG["lam_f"], SEG["phi_f"])):
        q = dict(p); q["lam"] = lam; q["phi"] = phi
        out["base"][f"m0_{nm}"] = float(m0(q))
        out["base"][f"Qstar_spend_{nm}_usd"] = float(qstar(q) * p["P"])

    # ---------- one-at-a-time tornado on m0 ----------
    tornado = {}
    for k, (lo, hi) in RANGES.items():
        if k in ("lam_s", "lam_f", "phi_s", "phi_f", "alpha_stable", "alpha_tail", "m"):
            continue
        vals = []
        for v in (lo, hi):
            q = dict(p); q[k] = v
            vals.append(float(m0(q)))
        tornado[k] = vals
    out["tornado_m0"] = tornado

    # ---------- Monte Carlo ----------
    N = 4000
    samples = {k: rng.uniform(lo, hi, N) for k, (lo, hi) in RANGES.items()}
    res = {"m0_stable": [], "m0_frontier": [], "Qstar_stable_spend": [], "contest_now": [],
           "opt_margin": [], "opt_lost": []}
    pop_cache = {}
    for i in range(N):
        q = dict(BASE)
        for k in ("m", "o", "T", "kappa", "mu", "eta", "ell", "sigma", "F"):
            q[k] = samples[k][i]
        seg = {k: samples[k][i] for k in ("alpha_stable", "lam_s", "lam_f", "phi_s", "phi_f")}
        at = round(samples["alpha_tail"][i], 2)
        if at not in pop_cache:
            pop_cache[at] = buyer_population(at)[0]
        qsi = pop_cache[at]
        qa = dict(q); qa["lam"] = seg["lam_s"]; qa["phi"] = seg["phi_s"]
        qb = dict(q); qb["lam"] = seg["lam_f"]; qb["phi"] = seg["phi_f"]
        res["m0_stable"].append(float(m0(qa)))
        res["m0_frontier"].append(float(m0(qb)))
        res["Qstar_stable_spend"].append(float(qstar(qa) * q["P"]))
        res["contest_now"].append(contestable(q, qsi, **seg))
        b, _ = nvidia_uniform_opt(q, qsi, **seg, grid=np.linspace(0.30, 0.85, 56))
        res["opt_margin"].append(b[0]); res["opt_lost"].append(b[1])
    arr = {k: np.array(v) for k, v in res.items()}

    def pct(x):
        x = x[np.isfinite(x)]
        return {"p5": float(np.percentile(x, 5)), "p25": float(np.percentile(x, 25)), "median": float(np.median(x)),
                "p75": float(np.percentile(x, 75)), "p95": float(np.percentile(x, 95)), "n_finite": int(len(x))}
    mc = {k: pct(v) for k, v in arr.items()}
    mc["P(m_sample > m0_stable)"] = float(np.mean(samples["m"] > arr["m0_stable"]))
    mc["P(m_sample > m0_frontier)"] = float(np.mean(samples["m"] > arr["m0_frontier"]))
    mc["P(contest_now > 0)"] = float(np.mean(arr["contest_now"] > 0))
    mc["P(Qstar_stable_spend < 4th buyer 2yr spend)"] = float(np.mean(
        arr["Qstar_stable_spend"] < (out["population"]["top4_nve_per_yr"][3] * 2 * P_FRONTIER * np.array(samples["alpha_stable"]))))
    out["monte_carlo"] = mc

    # PRCC (partial rank correlation) for m0_stable and contest_now
    from scipy.stats import rankdata
    keys = list(RANGES.keys())
    X = np.column_stack([rankdata(samples[k]) for k in keys])

    def prcc(y):
        yr = rankdata(y)
        out_ = {}
        for j, k in enumerate(keys):
            others = np.delete(X, j, axis=1)
            A = np.column_stack([others, np.ones(len(yr))])
            rx = X[:, j] - A @ np.linalg.lstsq(A, X[:, j], rcond=None)[0]
            ry = yr - A @ np.linalg.lstsq(A, yr, rcond=None)[0]
            out_[k] = float(np.corrcoef(rx, ry)[0, 1])
        return out_
    out["prcc_m0_stable"] = prcc(arr["m0_stable"])
    out["prcc_contest_now"] = prcc(arr["contest_now"])
    out["prcc_opt_margin"] = prcc(arr["opt_margin"])

    json.dump(out, open(os.path.join(RES, "model_results.json"), "w"), indent=1)
    np.savez(os.path.join(RES, "mc_samples.npz"), **{k: samples[k] for k in samples}, **arr)

    # ---------------- figures ----------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Fig: limit margin vs buyer size
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    Qgrid = np.logspace(3.5, 7, 200)
    for sig, ls in ((0.7, ":"), (0.85, "-"), (1.0, "--")):
        q = dict(p); q["sigma"] = sig; q["lam"] = SEG["lam_s"]; q["phi"] = SEG["phi_s"]
        ax[0].plot(Qgrid * p["P"] / 1e9, limit_margin(q, Qgrid), ls, color="#1b7837", label=f"stable segment, sigma={sig}")
        q["lam"] = SEG["lam_f"]; q["phi"] = SEG["phi_f"]
        ax[0].plot(Qgrid * p["P"] / 1e9, limit_margin(q, Qgrid), ls, color="#b2182b", label=f"frontier segment, sigma={sig}")
    ax[0].axhline(NV_GM_FY25, color="k", lw=0.8); ax[0].text(0.2, NV_GM_FY25 + 0.01, "NVIDIA GAAP GM FY25 (75.0%)", fontsize=7)
    ax[0].axhline(NV_GM_FY26, color="gray", lw=0.8, ls="--"); ax[0].text(0.2, NV_GM_FY26 - 0.03, "FY26 (71.1%)", fontsize=7)
    for i, lab in enumerate(["A", "B", "C", "D"]):
        ax[0].axvline(qs[i] * 2 * p["P"] / 1e9, color="#999999", lw=0.5)
        ax[0].text(qs[i] * 2 * p["P"] / 1e9, 0.02, lab, fontsize=7, ha="center")
    ax[0].set_xscale("log"); ax[0].set_ylim(0, 0.95)
    ax[0].set_xlabel("Buyer's 2-year segment spend at NVIDIA prices ($B)")
    ax[0].set_ylabel("Limit margin m*(Q)")
    ax[0].set_title("Max NVIDIA margin that keeps a buyer from building", fontsize=9.5)
    ax[0].legend(fontsize=6.5, loc="upper right")
    ms = [c[0] for c in curve]; lost = [c[1] for c in curve]; prof = np.array([c[2] for c in curve])
    ax2 = ax[1]
    ax2.plot(ms, 1 - np.array(lost), color="#2166ac", label="NVIDIA retained share of NVE demand")
    ax2.plot(ms, prof / prof.max(), color="#555555", ls="--", label="NVIDIA gross profit (normalised)")
    ax2.axvline(best[0], color="k", lw=0.7)
    ax2.set_xlabel("NVIDIA uniform gross margin m"); ax2.set_ylim(0, 1.05)
    ax2.set_title("Margin-share frontier (base calibration)", fontsize=9.5)
    ax2.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_limit_margin.png"), dpi=170)

    # Fig: MC distributions + PRCC
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.0))
    ax[0].hist(arr["m0_stable"], bins=50, alpha=0.7, color="#1b7837", label="m0, stable segment")
    ax[0].hist(arr["m0_frontier"], bins=50, alpha=0.6, color="#b2182b", label="m0, frontier segment")
    ax[0].axvspan(NV_GM_FY26, 0.78, color="gray", alpha=0.2, label="observed GM range 71-78%")
    ax[0].set_xlabel("Viability margin floor m0"); ax[0].legend(fontsize=7)
    ax[0].set_title("Monte Carlo (N=4000) over Table-3 ranges", fontsize=9.5)
    pr = out["prcc_m0_stable"]
    ks = sorted([k for k in pr if k not in ("alpha_tail", "alpha_stable", "lam_f", "phi_f", "m")], key=lambda k: abs(pr[k]))
    ax[1].barh(ks, [pr[k] for k in ks], color=["#1b7837" if pr[k] < 0 else "#b2182b" for k in ks])
    ax[1].set_xlabel("PRCC with m0 (stable segment)"); ax[1].axvline(0, color="k", lw=0.6)
    ax[1].set_title("What drives the margin floor", fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_mc.png"), dpi=170)

    print(json.dumps({k: out[k] for k in ("anchors", "base", "population", "tornado_m0", "monte_carlo")}, indent=1))
    print("PRCC m0_stable", json.dumps(out["prcc_m0_stable"], indent=0))
    print("PRCC contest", json.dumps(out["prcc_contest_now"], indent=0))
    print("PRCC opt margin", json.dumps(out["prcc_opt_margin"], indent=0))


if __name__ == "__main__":
    main()
