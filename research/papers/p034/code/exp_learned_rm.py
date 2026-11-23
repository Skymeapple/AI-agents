"""Exp 4: learned reward models over a finite output pool.

Outputs x = (u, h): u in R^6 carries genuine quality, h in R^6 are nuisance features.
gold g(x) = f(u) - c * sum_j relu(|h_j| - 1)^2   (h outside the training range ruins quality)
RM training data: in dataset 'A', nuisance dims 0-2 are a shortcut (spuriously track f(u));
dims 3-5 are small noise. The policy is exponential tilting over a pool that contains a
small share of off-support 'exploit' outputs (large |h|) which no RM ever saw.
Held-out RMs differ from the optimised RM1 by seed / data draw / data distribution / model
class; the extent to which they share RM1's exploit overestimation is measured, not set.
Run: python3 exp_learned_rm.py   (about 3-6 minutes)
"""
import json, os, sys, time, warnings
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from audit import cov_estimate

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
rng = np.random.default_rng(1)
DU, DH = 6, 6
W = rng.normal(0, 1, DU)
t0 = time.time()


def f_u(u):
    return 0.6 * u @ W / np.sqrt(DU) + np.sin(2 * u[:, 0]) * np.cos(u[:, 1]) + 0.5 * np.sin(u[:, 2] * u[:, 3]) + 0.3 * u[:, 4] * u[:, 5]

F_SD = np.std(f_u(rng.normal(0, 1, (100000, DU))))


def gold(X):
    u, h = X[:, :DU], X[:, DU:]
    return f_u(u) / F_SD - 1.0 * np.sum(np.maximum(np.abs(h) - 1.0, 0) ** 2, axis=1)


def sample_train(n, kind, rng):
    u = rng.normal(0, 1, (n, DU))
    fu = f_u(u) / F_SD
    h = 0.25 * rng.normal(0, 1, (n, DH))
    if kind == "A":  # shortcut: dims 0-2 track quality
        h[:, :3] = 0.6 * np.tanh(fu)[:, None] + 0.15 * rng.normal(0, 1, (n, 3))
    elif kind == "B":  # decorrelated, broader nuisance coverage (still within |h|<=1 mostly)
        h = rng.uniform(-1, 1, (n, DH))
    X = np.hstack([u, h])
    y = gold(X) + 0.3 * rng.normal(0, 1, n)
    return X, y


def mlp(seed):
    return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 64), alpha=1e-3, max_iter=400,
                                                        early_stopping=True, random_state=seed))

# ---------------- reward models ----------------
XA, yA = sample_train(6000, "A", np.random.default_rng(10))
XA2, yA2 = sample_train(6000, "A", np.random.default_rng(11))
XB, yB = sample_train(6000, "B", np.random.default_rng(12))
models = {
    "RM1 (optimised; data A, seed 0)": mlp(0).fit(XA, yA),
    "H1: same data A, new init seed": mlp(1).fit(XA, yA),
    "H2: fresh draw of data A": mlp(2).fit(XA2, yA2),
    "H3: quadratic ridge on data A": make_pipeline(StandardScaler(), PolynomialFeatures(2), Ridge(alpha=1.0)).fit(XA, yA),
    "H4: decorrelated data B": mlp(3).fit(XB, yB),
}
names = list(models)
print(f"trained RMs [{time.time()-t0:.0f}s]")

# ---------------- output pool (base policy) ----------------
NP, P_EX = 200_000, 0.01
Xg, _ = sample_train(NP, "A", np.random.default_rng(20))  # genuine outputs follow the natural distribution
n_ex = int(NP * P_EX)
r_ = np.random.default_rng(21)
dirs = r_.normal(0, 1, (n_ex, DH)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
Xe = np.hstack([r_.normal(0, 1, (n_ex, DU)), dirs * r_.uniform(2.0, 3.5, (n_ex, 1))])
Xpool = np.vstack([Xg[: NP - n_ex], Xe])
Zpool = np.r_[np.zeros(NP - n_ex, bool), np.ones(n_ex, bool)]
Gpool = gold(Xpool)
R = {k: m.predict(Xpool) for k, m in models.items()}
# standardise every proxy on genuine pool items (common scale with gold)
for k in R:
    gk = R[k][~Zpool]
    R[k] = (R[k] - gk.mean()) / gk.std() * Gpool[~Zpool].std() + Gpool[~Zpool].mean()
res = {"in_dist_corr_with_gold": {k: float(np.corrcoef(R[k][~Zpool], Gpool[~Zpool])[0, 1]) for k in names},
       "exploit_gold_mean": float(Gpool[Zpool].mean()), "genuine_gold_mean": float(Gpool[~Zpool].mean())}
print(res)

r1 = R[names[0]]
lams = np.linspace(0, 12, 241)

def tilt_w(score, lam):
    w = np.exp(lam * (score - score.max())); return w / w.sum()

def curves(score):
    out = {k: [] for k in ["gold", "pi", "cov", "kl"] + [f"gap|{n}" for n in names[1:]] + [f"qhat|{n}" for n in names[1:]]}
    for lam in lams:
        w = tilt_w(score, lam)
        Eg = w @ Gpool
        out["gold"].append(Eg); out["pi"].append(w @ Zpool)
        out["cov"].append(w @ (Gpool * (r1 - w @ r1)))
        out["kl"].append(float(np.sum(w * np.log(np.maximum(w * NP, 1e-300)))))
        for n in names[1:]:
            out[f"gap|{n}"].append(w @ (r1 - R[n]))
            over1 = w @ ((r1 - Gpool) * Zpool); over2 = w @ ((R[n] - Gpool) * Zpool)
            out[f"qhat|{n}"].append(over2 / over1 if abs(over1) > 1e-9 else np.nan)
    return {k: np.array(v) for k, v in out.items()}

C = curves(r1)
i_s = int(np.argmax(C["gold"])); lam_s = lams[i_s]
res["single_RM"] = dict(lam_star=float(lam_s), gold_star=float(C["gold"][i_s]), pi_at_peak=float(C["pi"][i_s]),
                        kl_at_peak=float(C["kl"][i_s]), gold_at_lammax=float(C["gold"][-1]), pi_at_lammax=float(C["pi"][-1]))
# derivative check: d gold/d lam vs on-policy covariance
dg = np.gradient(C["gold"], lams)
res["cov_identity_max_abs_err"] = float(np.max(np.abs(dg[2:-2] - C["cov"][2:-2])))
res["qhat_at_peak"] = {n: float(C[f"qhat|{n}"][i_s]) for n in names[1:]}
res["qhat_at_lam8"] = {n: float(C[f"qhat|{n}"][np.argmin(abs(lams - 8))]) for n in names[1:]}
print("single RM", res["single_RM"], "qhat@peak", res["qhat_at_peak"])

if "--probe" in sys.argv:
    print([(round(l,2), round(g,3), round(p,3)) for l,g,p in zip(lams[::10], C["gold"][::10], C["pi"][::10])]); sys.exit()
# ---------------- ensembles as mitigation ----------------
ens = {}
for n in names[1:]:
    for agg in ["mean", "min"]:
        s = (r1 + R[n]) / 2 if agg == "mean" else np.minimum(r1, R[n])
        g_curve = np.array([tilt_w(s, l) @ Gpool for l in lams])
        ens[f"{agg}(RM1,{n})"] = dict(peak_gold=float(g_curve.max()), gold_at_lammax=float(g_curve[-1]))
res["ensembles"] = ens

# ---------------- stopping rules with audits ----------------
SCHED = lams[lams <= 8.0001][::5]  # step 0.25, up to lambda=8
NB, REPS = 2000, 150
def run_stop(n_ho, rule, m, rng, tau=None):
    D = []
    for lam in SCHED:
        w = tilt_w(r1, lam)
        idx = np.minimum(np.searchsorted(np.cumsum(w), rng.random(NB)), NP - 1)
        g, a, b = Gpool[idx], r1[idx], R[n_ho][idx]
        if rule.startswith("cov"):
            design = rule.split("-")[1]
            est, se, _ = cov_estimate(g, a, b, m, design, False, rng)
            if est < 0:
                return lam
        elif rule == "firsthack":
            if Zpool[idx[rng.choice(NB, m, replace=False)]].any():
                return lam
        elif rule == "gap":
            D.append(np.mean(a - b))
            if len(D) > 4:
                x = SCHED[:4]; slope = np.sum(x * np.array(D[:4])) / np.sum(x * x)
                if D[-1] - slope * lam > tau:
                    return lam
    return SCHED[-1]

def regret(lst):
    lst = np.array(lst); gi = np.array([tilt_w(r1, l) @ Gpool for l in lst])
    return dict(mean_lam=float(lst.mean()), regret=float(C["gold"][i_s] - gi.mean()), p10_gold=float(np.percentile(gi, 10)),
                frac_overshoot=float(np.mean(lst > lam_s + 1e-9)),
                se=float(gi.std(ddof=1) / np.sqrt(len(gi))))

stop = {}
# tune proxy-only gap alarm on the held-out RM that shares the least (lowest qhat at peak), then apply to all
best_ho = min(names[1:], key=lambda n: res["qhat_at_peak"][n])
taus = np.linspace(0.05, 2.0, 40)
tune_rng = np.random.default_rng(5)
tr = [(t, regret([run_stop(best_ho, "gap", 0, tune_rng, t) for _ in range(40)])["regret"]) for t in taus]
tau_star = float(min(tr, key=lambda x: x[1])[0])
res["gap_alarm_tuned_on"] = best_ho; res["gap_alarm_tau"] = tau_star
for n in names[1:]:
    stop[f"{n}|gap-alarm(proxy-only)"] = regret([run_stop(n, "gap", 0, rng, tau_star) for _ in range(REPS)])
    stop[f"{n}|cov-disagree m=20"] = regret([run_stop(n, "cov-disagree", 20, rng) for _ in range(REPS)])
for rule in ["cov-uniform", "cov-leverage"]:
    stop[f"(any)|{rule} m=20"] = regret([run_stop(names[1], rule, 20, rng) for _ in range(REPS)])
stop["(any)|firsthack m=20"] = regret([run_stop(names[1], "firsthack", 20, rng) for _ in range(REPS)])
res["stopping"] = stop
for k, v in stop.items():
    print(k, {a: round(b, 3) for a, b in v.items()})

# ---------------- figure ----------------
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
ax[0].plot(lams, C["gold"], "k", label="gold"); ax[0].plot(lams, C["pi"], "m:", label="exploit share")
ax[0].axvline(lam_s, color="grey", ls=":"); ax[0].set_xlabel("lambda"); ax[0].legend(); ax[0].set_title("Optimising RM1 (learned MLP)")
for n in names[1:]:
    ax[1].plot(lams, C[f"gap|{n}"], label=n.split(":")[0] + f" (q^={res['qhat_at_peak'][n]:.2f})")
    ax[2].plot(lams, C[f"qhat|{n}"], label=n)
ax[1].set_xlabel("lambda"); ax[1].set_title("held-out gap E[r1 - r_k]"); ax[1].legend(fontsize=7)
ax[2].set_xlabel("lambda"); ax[2].set_title("measured shared vulnerability q^"); ax[2].set_ylim(-0.5, 1.5); ax[2].legend(fontsize=6)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig4_learned_rm.png"), dpi=150); plt.close()
json.dump(res, open(os.path.join(OUT, "learned_rm_results.json"), "w"), indent=1)
print(f"done in {time.time()-t0:.0f}s")
