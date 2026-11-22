"""Run all experiments for 'Per-request refusal rights under retries'.
Usage: python3 run_experiments.py   (writes to ../results/)
"""
import json, os, time
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from refusal_model import *

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(R, exist_ok=True)
OUT = {}
rng = np.random.default_rng(20261122)
C = ["#2b6cb0", "#c05621", "#2f855a", "#9b2c2c", "#6b46c1", "#4a5568"]
t0 = time.time()

# ======================================================== E1 Monte Carlo validation
def mc(p, t, K, policy="memoryless", m=3, n=400_000, qH=0.9, qB=0.05):
    res = {}
    for typ, mu in (("H", p.mu_H), ("B", p.mu_B)):
        a = mu + p.tau * rng.standard_normal(n)
        blocked = np.zeros(n, bool); done = np.zeros(n, bool); refusals = np.zeros(n, int)
        for j in range(K):
            act = ~done
            if not act.any():
                break
            if policy in ("strike", "escalate"):
                hit = act & (refusals >= m)
                if policy == "strike":
                    blocked[hit] = True
                else:
                    u = rng.random(n)
                    blocked[hit] = u[hit] < (qH if typ == "H" else qB)
                done[hit] = True
                act = ~done
            s = a + p.sigma * rng.standard_normal(n)
            ref = act & (s > t)
            served = act & ~ref
            done[served] = True
            refusals[ref] += 1
            if typ == "B":  # benign may quit after a refusal
                quit_ = ref & (rng.random(n) > p.beta)
                blocked[quit_] = True; done[quit_] = True
        # still active after K attempts: refused every time
        rem = ~done
        if policy == "escalate":
            hit = rem & (refusals >= m)
            u = rng.random(n)
            blocked[hit] = u[hit] < (qH if typ == "H" else qB)
            blocked[rem & ~hit] = True
        else:
            blocked[rem] = True
        res[typ] = blocked.mean()
    return res

val_rows = []
for rho in (0.0, 0.3, 0.7):
    p = Params(rho=rho, d=2.0, beta=0.5)
    for (pol, K) in (("memoryless", 1), ("memoryless", 5), ("memoryless", 20),
                     ("strike", 20), ("escalate", 20)):
        t = 1.5
        sim = mc(p, t, K, pol, m=3)
        if pol == "memoryless":
            PH = harmful_blocked(t, K, p.mu_H, p.tau, p.sigma, grid=p.gH)
            PB = benign_blocked(t, K, p.beta, p.mu_B, p.tau, p.sigma, grid=p.gB)
        elif pol == "strike":
            _, PH, PB = value_strike(p, t, 3, K)
        else:
            _, PH, PB = value_escalate(p, t, 3, K)
        val_rows.append(dict(rho=rho, policy=pol, K=K, PH_analytic=PH, PH_mc=sim["H"],
                             PB_analytic=PB, PB_mc=sim["B"]))
maxerr = max(max(abs(r["PH_analytic"] - r["PH_mc"]), abs(r["PB_analytic"] - r["PB_mc"]))
             for r in val_rows)
OUT["E1_validation"] = dict(rows=val_rows, max_abs_error=maxerr, n_mc=400_000)
print("E1 done, max abs err", maxerr, time.time() - t0)

# ======================================================== E2 power law of harmful blocking
Ks = np.unique(np.round(np.logspace(0, 9, 91)).astype(np.int64))
def ph_curve(rho, t, noise="gauss", latent="gauss", d=2.0):
    tau, sig = np.sqrt(rho), np.sqrt(1 - rho)
    span = 30.0 if latent == "gauss" else 1.0
    if latent == "gauss":
        x = np.linspace(d - span * tau, d + span * tau, 120001)
        lw = stats.norm.logpdf(x, d, tau)
    else:
        b = tau / np.sqrt(2)
        x = np.linspace(d - 120 * b, d + 120 * b, 120001)
        lw = stats.laplace.logpdf(x, d, b)
    lw = lw - np.log(np.sum(np.exp(lw - lw.max()))) - lw.max()
    lr, _ = log_refuse_prob(x, t, sig, noise)
    out = []
    for K in Ks:
        z = lw + K * lr
        mx = z.max()
        out.append(mx + np.log(np.sum(np.exp(z - mx))))
    return np.array(out)  # log P_H(blocked)

def local_slope(logp):
    lk = np.log(Ks.astype(float))
    return np.gradient(logp, lk)

pl = {}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for i, rho in enumerate((0.2, 0.33, 0.5, 0.67, 0.8)):
    lp = ph_curve(rho, t=1.0)
    sl = local_slope(lp)
    pred = (1 - rho) / rho
    idx = {int(k): j for j, k in enumerate(Ks)}
    def at(K):
        j = int(np.argmin(np.abs(np.log(Ks) - np.log(K))))
        return float(-sl[j])
    pl[str(rho)] = dict(predicted_alpha=pred, local_alpha={str(K): at(K) for K in (1e1, 1e2, 1e3, 1e4, 1e6, 1e9)},
                        PH_blocked={str(K): float(np.exp(lp[int(np.argmin(np.abs(np.log(Ks) - np.log(K))))]))
                                    for K in (1, 10, 100, 1e3, 1e4, 1e6)})
    ax[0].loglog(Ks, np.exp(lp), color=C[i], label=f"ρ={rho}")
    ax[1].semilogx(Ks, -sl, color=C[i])
    ax[1].axhline(pred, color=C[i], ls=":", lw=1)
ax[0].set_xlabel("retry budget K"); ax[0].set_ylabel("P(harmful user blocked)")
ax[0].set_title("Harmful blocking under memoryless refusal (t=1, d'=2)"); ax[0].legend(fontsize=8)
ax[0].set_ylim(1e-30, 1.5)
ax[1].set_xlabel("retry budget K"); ax[1].set_ylabel("local exponent  −dlogP/dlogK")
ax[1].set_title("Local exponent drifts toward (1−ρ)/ρ (dotted)"); ax[1].set_ylim(0, 6)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig1_powerlaw.png"), dpi=150); plt.close(fig)

# tail-matching variants at rho=0.5
tm = {}
fig, ax = plt.subplots(figsize=(5.6, 4.2))
for j, (noise, latent, lab) in enumerate((("gauss", "gauss", "Gauss latent / Gauss noise"),
                                          ("logistic", "gauss", "Gauss latent / logistic noise"),
                                          ("gauss", "laplace", "Laplace latent / Gauss noise"))):
    lp = ph_curve(0.5, 1.0, noise, latent)
    sl = -local_slope(lp)
    tm[lab] = {str(K): float(sl[int(np.argmin(np.abs(np.log(Ks) - np.log(K))))]) for K in (1e2, 1e4, 1e6, 1e9)}
    tm[lab]["PH_blocked_K1e4"] = float(np.exp(lp[int(np.argmin(np.abs(np.log(Ks) - np.log(1e4))))]))
    ax.loglog(Ks, np.exp(lp), color=C[j], label=lab)
ax.set_xlabel("retry budget K"); ax.set_ylabel("P(harmful user blocked)"); ax.set_ylim(1e-25, 1.5)
ax.set_title("Tail matching (ρ=0.5)"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig2_tailmatching.png"), dpi=150); plt.close(fig)
OUT["E2_powerlaw"] = dict(t=1.0, d=2.0, curves=pl, tail_matching_local_alpha=tm)
print("E2 done", time.time() - t0)

# ======================================================== E3 value decay & threshold path
Kgrid = [1, 2, 3, 5, 10, 20, 50, 100, 300, 1000, 3000, 10000]
e3 = {}
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
for hi, h in enumerate((30.0, 300.0)):
    for i, rho in enumerate((0.0, 0.3, 0.6, 0.9)):
        p = Params(pi=0.01, h=h, d=2.0, rho=rho, beta=0.5)
        rows = []
        for K in Kgrid:
            V, t, PH, PB = optimise(lambda tt: value_memoryless(p, tt, K))
            rows.append(dict(K=K, V=V, t=t if np.isfinite(t) else None, PH=PH, PB=PB,
                             fpr1=single_shot_fpr(p, t), tpr1=single_shot_tpr(p, t)))
        Vs = np.array([r["V"] for r in rows])
        # stateful 3-strike, L = 0 and L = 5
        Vs3 = optimise(lambda tt: value_strike(p, tt, 3, 10_000))[0]
        p5 = Params(pi=0.01, h=h, d=2.0, rho=rho, beta=0.5, L=5.0)
        Vs3L = optimise(lambda tt: value_strike(p5, tt, 3, 10_000))[0]
        Ves = optimise(lambda tt: value_escalate(p, tt, 3, 10_000))[0]
        e3[f"h={h},rho={rho}"] = dict(memoryless=rows, strike3_L0=Vs3, strike3_L5=Vs3L,
                                     escalate3=Ves, shutdown_value=max(0.0, p.pi * h - (1 - p.pi)))
        if hi == 0:
            axs[0].semilogx(Kgrid, Vs / Vs[0], "-o", ms=3, color=C[i], label=f"ρ={rho}")
            axs[1].semilogx(Kgrid, [r["fpr1"] for r in rows], "-o", ms=3, color=C[i], label=f"ρ={rho}")
        else:
            axs[2].semilogx(Kgrid, Vs, "-o", ms=3, color=C[i], label=f"memoryless ρ={rho}")
            axs[2].axhline(Vs3, color=C[i], ls="--", lw=1)
    if hi == 1:
        axs[2].axhline(max(0, 0.01 * h - 0.99), color="k", ls=":", lw=1, label="refuse-everything value")
axs[0].set_title("Value of refusal right, memoryless (h/v=30, π=0.01)"); axs[0].set_xlabel("adversary retry budget K"); axs[0].set_ylabel("V*(K) / V*(1)"); axs[0].legend(fontsize=8)
axs[1].set_title("Over-refusal at optimal threshold (h/v=30)"); axs[1].set_xlabel("adversary retry budget K"); axs[1].set_ylabel("benign single-shot refusal rate"); axs[1].legend(fontsize=8)
axs[2].set_title("h/v=300: memoryless (solid) vs 3-strike (dashed)"); axs[2].set_xlabel("adversary retry budget K"); axs[2].set_ylabel("V* (per user, units of v)"); axs[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(R, "fig3_value_decay.png"), dpi=150); plt.close(fig)
OUT["E3_value_decay"] = e3
print("E3 done", time.time() - t0)

# ======================================================== E4 retention map (rho x h)
rhos = np.round(np.linspace(0.0, 0.95, 20), 3)
hs = np.logspace(0.5, 3.5, 16)
ret_mem = np.zeros((len(hs), len(rhos))); ret_str = np.zeros_like(ret_mem)
tg = np.linspace(-4, 10, 281)
for i, h in enumerate(hs):
    for j, rho in enumerate(rhos):
        p = Params(pi=0.01, h=h, d=2.0, rho=rho, beta=0.5)
        V1 = optimise(lambda tt: value_memoryless(p, tt, 1), tg)[0]
        Vk = optimise(lambda tt: value_memoryless(p, tt, 100), tg)[0]
        Vs = optimise(lambda tt: value_strike(p, tt, 3, 100), tg)[0]
        sd = max(0.0, p.pi * h - (1 - p.pi))
        # retention of value *above* the deploy/no-deploy decision
        denom = V1 - sd
        ret_mem[i, j] = (Vk - sd) / denom if denom > 1e-9 else np.nan
        ret_str[i, j] = (Vs - sd) / denom if denom > 1e-9 else np.nan
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, M, ttl in ((axs[0], ret_mem, "memoryless, K=100"), (axs[1], ret_str, "3-strike, K=100")):
    im = ax.imshow(M, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis",
                   extent=[rhos[0], rhos[-1], np.log10(hs[0]), np.log10(hs[-1])])
    ax.set_title(f"Retained refusal value vs K=1 ({ttl})"); ax.set_xlabel("consistency ρ")
axs[0].set_ylabel("log10 harm ratio h/v")
fig.colorbar(im, ax=axs, shrink=0.85)
fig.savefig(os.path.join(R, "fig4_retention_map.png"), dpi=150); plt.close(fig)
OUT["E4_retention"] = dict(rhos=rhos.tolist(), hs=hs.tolist(), memoryless_K100=np.nan_to_num(ret_mem, nan=-1).tolist(),
                           strike3_K100=np.nan_to_num(ret_str, nan=-1).tolist(),
                           median_retention_memoryless=float(np.nanmedian(ret_mem)),
                           median_retention_strike=float(np.nanmedian(ret_str)))
print("E4 done", time.time() - t0)

# ======================================================== E5 calibration to a published flip rate
# Larsen (2025, arXiv:2512.12066) reports 24.8% of harmful prompts flip decision across 20
# sampling configurations (temperature x seed). Under our model with a deterministic latent
# per prompt and iid sampling noise, P(flip in n draws) = 1 - E[r^n + (1-r)^n].
# We invert for rho given an assumed mean single-shot refusal rate pbar (not taken from the paper).
from scipy.optimize import brentq
def flip_prob(rho, pbar, n=20):
    tau, sig = np.sqrt(rho), np.sqrt(1 - rho)
    t = -stats.norm.ppf(pbar)  # latent mean 0, total var 1 -> P(s>t)=pbar
    x, w = latent_grid(0.0, tau) if rho > 0 else (np.array([0.0]), np.array([1.0]))
    lr, l1 = log_refuse_prob(x, t, sig)
    return float(np.sum(w * (1 - np.exp(n * lr) - np.exp(n * l1))))
cal = {}
for pbar in (0.80, 0.90, 0.95):
    f = lambda r: flip_prob(r, pbar) - 0.248
    lo, hi = 0.01, 0.999
    if f(lo) * f(hi) < 0:
        rr = brentq(f, lo, hi)
        cal[str(pbar)] = dict(rho=rr, implied_alpha=(1 - rr) / rr)
    else:
        cal[str(pbar)] = dict(rho=None, note="no solution in range", f_lo=f(lo), f_hi=f(hi))
OUT["E5_calibration"] = dict(source_flip_rate=0.248, n_configs=20, results=cal)
print("E5", cal, time.time() - t0)

# ======================================================== E6 sensitivity
sens = []
base = dict(pi=0.01, h=30.0, d=2.0, beta=0.5)
for key, vals in (("beta", (0.0, 0.25, 0.5, 0.75, 0.9)), ("d", (1.0, 1.5, 2.0, 3.0)),
                  ("pi", (0.001, 0.01, 0.05)), ("h", (10.0, 30.0, 100.0))):
    for v in vals:
        for rho in (0.3, 0.8):
            kw = dict(base); kw[key] = v
            p = Params(rho=rho, **kw)
            V1 = optimise(lambda tt: value_memoryless(p, tt, 1), tg)[0]
            V100 = optimise(lambda tt: value_memoryless(p, tt, 100), tg)[0]
            S = optimise(lambda tt: value_strike(p, tt, 3, 100), tg)[0]
            sd = max(0.0, p.pi * p.h - (1 - p.pi))
            sens.append(dict(param=key, value=v, rho=rho, V1=V1, V100_memoryless=V100,
                             V100_strike3=S, shutdown=sd))
OUT["E6_sensitivity"] = sens
print("E6 done", time.time() - t0)

with open(os.path.join(R, "results.json"), "w") as fh:
    json.dump(OUT, fh, indent=1, default=float)
print("total seconds", time.time() - t0)
