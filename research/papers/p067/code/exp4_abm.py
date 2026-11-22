"""Exp 4: heterogeneous stochastic population (agent-based) simulation.
(a) AI quality improves then degrades: mean skill, bimodality, hysteresis;
(b) feedback vs no-feedback null model (bimodality as a discriminating signature);
(c) policies: practice floor, engagement design (higher kappa);
(d) early-warning signals (variance, lag-1 autocorrelation) before individual collapse."""
import json, numpy as np
from scipy.stats import kendalltau, skew, kurtosis
from scipy.ndimage import gaussian_filter1d
from model import *
from plotstyle import *

rng = np.random.default_rng(42)
N = 5000
dt = 0.1
base = Params(beta=100.0)

hl = np.exp(rng.normal(np.log(6.5), 0.35, N))         # forgetting half-life (months)
delta = LN2 / hl
kfac = np.exp(rng.normal(np.log(9.0), 0.35, N))       # alpha/delta -> s_H heterogeneity
alpha = delta * kfac
kappa = rng.uniform(0.05, 0.30, N)
dc = np.exp(rng.normal(np.log(0.03), 0.4, N))
sigma = 0.02

months = np.arange(0, 720 + dt, dt)
def eps_of_t(t):
    # 0-60: eps=0.55; 60-300 linear to 0.10; 300-360 hold; 360-600 linear back to 0.55; hold
    return np.interp(t, [0, 60, 300, 360, 600, 720], [0.55, 0.55, 0.10, 0.10, 0.55, 0.55])


def bimodality_coefficient(x):
    n = len(x)
    g, k = skew(x), kurtosis(x)
    return (g ** 2 + 1) / (k + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))


def run(policy="none", feedback=True, withdraw_at=None, seed=1):
    r_ = np.random.default_rng(seed)
    s = alpha / (alpha + delta)  # start at own skilled equilibrium
    sH_i = s.copy()
    kap = np.minimum(kappa + (0.30 if policy == "engage" else 0.0), 1.0)
    umin = 0.30 if policy == "floor" else 0.0
    rec = dict(t=[], mean=[], frac_low=[], bc=[], rel=[], snap={})
    snaps = {60, 200, 300, 450, 600, 720}
    for k, t in enumerate(months):
        e = eps_of_t(t)
        sref = s if feedback else sH_i
        Dv = base.h0 + base.h1 * (1 - sref) + dc - e * (1 - base.rho * sref)
        r = 1 / (1 + np.exp(-base.beta * Dv))
        r = np.minimum(r, 1 - umin)
        if withdraw_at is not None and t >= withdraw_at:
            r = np.zeros_like(r)
        p = (1 - r) + kap * r
        s = s + (alpha * p * (1 - s) - delta * s) * dt + sigma * np.sqrt(dt) * r_.standard_normal(N)
        s = np.clip(s, 0, 1)
        if k % 10 == 0:
            rel_s = s / sH_i
            rec["t"].append(float(t)); rec["mean"].append(float(rel_s.mean()))
            rec["frac_low"].append(float((rel_s < 0.8).mean()))
            rec["bc"].append(float(bimodality_coefficient(rel_s)))
            rec["rel"].append(float(r.mean()))
            if int(round(t)) in snaps and abs(t - round(t)) < 1e-9:
                rec["snap"][int(round(t))] = rel_s.copy()
    return rec


runs = {
    "feedback": run(),
    "no_feedback": run(feedback=False),
    "floor30": run(policy="floor"),
    "engage+0.3": run(policy="engage"),
}
withdraw = run(withdraw_at=300)

summary = {}
for name, rec in runs.items():
    t = np.array(rec["t"]); m = np.array(rec["mean"]); fl = np.array(rec["frac_low"]); bc = np.array(rec["bc"])
    epst = eps_of_t(t)
    # hysteresis: at the same eps=0.30, compare descending (t~180) and ascending (t~467)
    i_down = np.argmin(np.abs(t - 193.3)); i_up = np.argmin(np.abs(t - 466.7))
    summary[name] = dict(min_mean_rel_skill=float(m.min()), max_frac_low=float(fl.max()),
                         max_bimodality_coef=float(bc.max()),
                         t_at_max_bc=float(t[bc.argmax()]), eps_at_max_bc=float(epst[bc.argmax()]),
                         eps_at_hyst_check=[float(epst[i_down]), float(epst[i_up])],
                         mean_rel_skill_down_vs_up=[float(m[i_down]), float(m[i_up])],
                         frac_low_down_vs_up=[float(fl[i_down]), float(fl[i_up])],
                         final_mean_rel_skill=float(m[-1]))
# recovery after withdrawal
t = np.array(withdraw["t"]); m = np.array(withdraw["mean"])
pre = m[np.argmin(np.abs(t - 299))]
target = pre + 0.9 * (1.0 - pre)
after = np.where((t >= 300) & (m >= target))[0]
summary["withdrawal_at_300"] = dict(mean_rel_skill_before=float(pre),
                                    months_to_recover_90pct=float(t[after[0]] - 300) if len(after) else None)

# ---------------- early warning signals ----------------
def ews_run(seed, ramp=True, T=600.0, sig=0.01):
    r_ = np.random.default_rng(seed)
    P = base
    n = int(T / dt)
    e_path = np.linspace(0.45, 0.22, n) if ramp else np.full(n, 0.45)
    s = 0.9
    xs = np.empty(n)
    for k in range(n):
        s = s + drift(s, e_path[k], P) * dt + sig * np.sqrt(dt) * r_.standard_normal()
        s = min(max(s, 0), 1)
        xs[k] = s
    monthly = xs[::int(1 / dt)]
    return monthly


def indicators(x, win=120):
    resid = x - gaussian_filter1d(x, 20)
    var, ac = [], []
    for i in range(win, len(x) + 1):
        w = resid[i - win:i]
        var.append(w.var()); ac.append(np.corrcoef(w[:-1], w[1:])[0, 1])
    tt = np.arange(len(var))
    return kendalltau(tt, var)[0], kendalltau(tt, ac)[0]


R = 200
tau_ramp, tau_ctrl, ct = [], [], []
for i in range(R):
    x = ews_run(1000 + i, ramp=True)
    hit = np.where(x < 0.75)[0]
    if len(hit) == 0 or hit[0] < 200:
        continue
    cut = hit[0] - 12  # stop 1 year before collapse
    ct.append(int(hit[0]))
    tau_ramp.append(indicators(x[:cut]))
    y = ews_run(5000 + i, ramp=False)
    tau_ctrl.append(indicators(y[:cut]))
tau_ramp, tau_ctrl = np.array(tau_ramp), np.array(tau_ctrl)


def auc(a, b):
    return float(np.mean(a[:, None] > b[None, :]) + 0.5 * np.mean(a[:, None] == b[None, :]))

summary["ews"] = dict(n_runs=len(tau_ramp), median_collapse_month=float(np.median(ct)),
                      median_tau_var_ramp=float(np.median(tau_ramp[:, 0])),
                      median_tau_var_ctrl=float(np.median(tau_ctrl[:, 0])),
                      median_tau_ac1_ramp=float(np.median(tau_ramp[:, 1])),
                      median_tau_ac1_ctrl=float(np.median(tau_ctrl[:, 1])),
                      auc_var=auc(tau_ramp[:, 0], tau_ctrl[:, 0]),
                      auc_ac1=auc(tau_ramp[:, 1], tau_ctrl[:, 1]))
json.dump(summary, open("../results/exp4_abm.json", "w"), indent=2)
print(json.dumps(summary, indent=1))

# ---------------- figures ----------------
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
cols = {"feedback": C1, "no_feedback": INK2, "floor30": C3, "engage+0.3": C2}
labs = {"feedback": "endogenous reliance (model)", "no_feedback": "null: no skill feedback",
        "floor30": "practice floor 30%", "engage+0.3": "engagement design (kappa+0.3)"}
for name, rec in runs.items():
    ax[0].plot(rec["t"], rec["mean"], color=cols[name], label=labs[name])
    ax[1].plot(rec["t"], rec["bc"], color=cols[name], label=labs[name])
axe = ax[0].twinx(); axe.plot(runs["feedback"]["t"], eps_of_t(np.array(runs["feedback"]["t"])), color=INK, lw=0.8, ls=":")
axe.set_ylabel("AI error eps (dotted)", color=INK2); axe.grid(False)
ax[0].set_xlabel("month"); ax[0].set_ylabel("mean skill / own pre-AI skill"); ax[0].set_title("(a) Population skill")
ax[0].legend(fontsize=7, loc="lower left")
ax[1].axhline(5 / 9, color=INK2, lw=0.8, ls="--"); ax[1].text(5, 5 / 9 + 0.01, "BC = 5/9 (uniform)", fontsize=7, color=INK2)
ax[1].set_xlabel("month"); ax[1].set_ylabel("bimodality coefficient"); ax[1].set_title("(b) Bimodality signature")
for name in ["feedback", "no_feedback", "floor30"]:
    rec = runs[name]; tt = np.array(rec["t"]); mm = np.array(rec["mean"])
    d = (tt >= 60) & (tt <= 300); u = (tt >= 360) & (tt <= 600)
    ax[2].plot(eps_of_t(tt[d]), mm[d], color=cols[name], label=labs[name] + " (AI improving)")
    ax[2].plot(eps_of_t(tt[u]), mm[u], color=cols[name], ls="--", label="  ... (AI degrading)")
ax[2].invert_xaxis(); ax[2].set_xlabel("AI error rate eps"); ax[2].set_ylabel("mean relative skill")
ax[2].set_title("(c) Hysteresis loop: model vs null"); ax[2].legend(fontsize=6.5, loc="lower left")
plt.tight_layout(); plt.savefig("../results/fig5_population.png")

fig, ax = plt.subplots(figsize=(5, 3.6))
ax.scatter(tau_ctrl[:, 1], tau_ctrl[:, 0], s=10, color=INK2, alpha=0.6, label="control (fixed eps)")
ax.scatter(tau_ramp[:, 1], tau_ramp[:, 0], s=10, color=C2, alpha=0.7, label="pre-collapse (eps falling)")
ax.set_xlabel("Kendall tau of lag-1 autocorrelation"); ax.set_ylabel("Kendall tau of variance")
ax.set_title("Early-warning signals before deskilling"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("../results/fig6_ews.png")
