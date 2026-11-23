"""Revision (round 1) analysis: multi-seed capacities, overwrite task, revised demand budget.

Inputs : results/train_seeds/*.json, results/train_overwrite/*.json, results/sim_capacity_ci.json,
         results/demand_v2.json, results/replay_memory.json
Outputs: results/analysis_v2.json, results/fig_seeds.png, results/fig_overwrite.png, results/fig_replay.png
"""
import os, json, glob, collections, numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
rng = np.random.default_rng(11)
out = {}


def nstar(Ns, accs):
    pts = sorted(zip(Ns, accs))
    for (n0, a0), (n1, a1) in zip(pts, pts[1:]):
        if a0 >= 0.5 > a1:
            f = (a0 - 0.5) / (a0 - a1)
            return float(np.exp(np.log(n0) + f * (np.log(n1) - np.log(n0))))
    return float("inf") if pts[-1][1] >= 0.5 else None


# ---------------- 1. multi-seed capacities
sim = json.load(open(os.path.join(R, "sim_capacity_ci.json")))
simN = {(c["dk"], c["dv"]): (c["Nstar_sim"], c["Nstar_sim_ci95"], c["Nstar_eq1"]) for c in sim["V512"]}
runs = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(R, "train_seeds", "*.json"))):
    d = json.load(open(f)); a = d["args"]
    Ns = [int(k) for k in d["acc"]]; acc = list(d["acc"].values())
    runs[(a["arch"], a["dk"])].append({"seed": a["seed"], "Nstar": nstar(Ns, acc), "acc": d["acc"],
                                       "gates": d.get("gates_on_pair_tokens"), "final_loss": d["final_loss"]})
seedtab = []
for (arch, dk), rr in sorted(runs.items()):
    ns = np.array([r["Nstar"] for r in rr], float)
    sN, sci, e1 = simN[(dk, 64)]
    row = {"arch": arch, "dk": dk, "dv": 64, "seeds": [r["seed"] for r in rr], "Nstar_each": ns.tolist(),
           "Nstar_mean": float(ns.mean()), "Nstar_sd": float(ns.std(ddof=1)) if len(ns) > 1 else None,
           "Nstar_sim": sN, "Nstar_sim_ci95": sci, "Nstar_eq1": e1,
           "eta_sim_each": (ns / sN).tolist(), "eta_sim_mean": float((ns / sN).mean()),
           "eta_eq1_mean": float((ns / e1).mean()),
           "beta_each": [r["gates"]["beta_mean"] for r in rr] if arch in ("delta", "gdelta") else None,
           "decay_each": [r["gates"]["decay_mean"] for r in rr] if arch in ("gla", "gdelta") else None}
    seedtab.append(row)
    print(arch, dk, np.round(ns, 1), "eta_sim %.3f" % row["eta_sim_mean"], row["beta_each"], row["decay_each"])
out["seeds"] = seedtab
# family-level summaries: log-log slope in d_k (per seed, then mean/sd), spread across families
fam = {}
for arch in sorted({k[0] for k in runs}):
    dks = sorted(dk for (a_, dk) in runs if a_ == arch)
    seeds = sorted({r["seed"] for dk in dks for r in runs[(arch, dk)]})
    slopes = []
    for s in seeds:
        pts = [(dk, r["Nstar"]) for dk in dks for r in runs[(arch, dk)] if r["seed"] == s]
        if len(pts) >= 2:
            slopes.append(float(np.polyfit(np.log([p[0] for p in pts]), np.log([p[1] for p in pts]), 1)[0]))
    fam[arch] = {"dk": dks, "slope_each_seed": slopes,
                 "slope_mean": float(np.mean(slopes)) if slopes else None,
                 "slope_sd": float(np.std(slopes, ddof=1)) if len(slopes) > 1 else None}
out["family_slopes"] = fam
# two-way decomposition of variance in eta_sim at each dk: between families vs between seeds
var = {}
for dk in sorted({k[1] for k in runs}):
    groups = [np.array([r["Nstar"] for r in runs[(a_, dk)]]) / simN[(dk, 64)][0] for a_ in ("linear", "gla", "delta", "gdelta") if (a_, dk) in runs]
    if all(len(g) >= 2 for g in groups) and len(groups) >= 2:
        allv = np.concatenate(groups); fm = np.array([g.mean() for g in groups])
        within = float(np.mean([g.var(ddof=1) for g in groups]))
        from scipy.stats import f_oneway
        F, p = f_oneway(*groups)
        var[dk] = {"eta_sim_grand_mean": float(allv.mean()), "family_means": fm.tolist(),
                   "sd_between_family_means": float(fm.std(ddof=1)), "sd_within_family_seeds": float(np.sqrt(within)),
                   "anova_F": float(F), "anova_p": float(p)}
out["eta_variance"] = var
print(json.dumps(var, indent=1))

# ---------------- 2. overwrite task (trained) + ideal memories
ow = {}
for f in sorted(glob.glob(os.path.join(R, "train_overwrite", "*.json"))):
    d = json.load(open(f)); a = d["args"]
    ow.setdefault(a["arch"], []).append({"seed": a["seed"], "acc": d["acc"], "gates": d["gates_on_pair_tokens"],
                                         "Nstar": nstar([int(k) for k in d["acc"]], list(d["acc"].values()))})


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def ideal_overwrite(kind, P, dk=16, dv=64, V=512, rep=3, trials=30, beta=1.0, g=0.9):
    accs = []
    for _ in range(trials):
        U = unit(rng.standard_normal((V, dv))); Kp = unit(rng.standard_normal((P, dk)))
        order = rng.permutation(P * rep) % P; vals = rng.integers(0, V, P * rep)
        M = np.zeros((dv, dk)); last = {}
        for t, (ki, v) in enumerate(zip(order, vals)):
            k = Kp[ki]; u = U[v]
            if kind == "additive":
                M += np.outer(u, k)
            elif kind == "decay":
                M = g * M + np.outer(u, k)
            elif kind == "delta":
                M += beta * np.outer(u - M @ k, k)
            last[ki] = v
        pred = np.argmax((Kp @ M.T) @ U.T, axis=1)
        accs.append(np.mean([pred[i] == last[i] for i in range(P)]))
    return float(np.mean(accs))


Ps = [4, 8, 16, 32, 64, 128]
ideal = {}
for name, kw in [("additive", {"kind": "additive"}), ("delta_beta1", {"kind": "delta", "beta": 1.0}),
                 ("delta_beta0.5", {"kind": "delta", "beta": 0.5}), ("decay_0.9", {"kind": "decay", "g": 0.9}),
                 ("decay_0.97", {"kind": "decay", "g": 0.97})]:
    ideal[name] = {P: ideal_overwrite(P=P, **kw) for P in Ps}
    print("ideal overwrite", name, {k: round(v, 3) for k, v in ideal[name].items()})
out["overwrite"] = {"trained": ow, "ideal": ideal}

# ---------------- 3. revised demand budget
dem = json.load(open(os.path.join(R, "demand_v2.json")))
rep = json.load(open(os.path.join(R, "replay_memory.json"))) if os.path.exists(os.path.join(R, "replay_memory.json")) else {}
budget = {}
for corpus in ["prose", "prose_bpe", "code", "prose_noKJV"]:
    D = dem[corpus]; lnV = np.log(D["vocab"]); rows = D["rows"]; T = np.array([r["T"] for r in rows], float)
    b = {"vocab": D["vocab"], "lnV": float(lnV)}
    for Dn in ["D0", "D3", "D4"]:
        N = np.array([r[f"N_ctx_{Dn}"] for r in rows]); lo = np.array([r[f"N_ctx_{Dn}_ci95"][0] for r in rows]); hi = np.array([r[f"N_ctx_{Dn}_ci95"][1] for r in rows])
        r32 = rows[-1]
        b[Dn] = {"N_ctx_32k": float(N[-1]), "N_ctx_32k_ci95": [float(lo[-1]), float(hi[-1])],
                 "rho_32k": r32[f"rho_{Dn}"], "rho_32k_ci95": r32[f"rho_{Dn}_ci95"],
                 "S_req_32k_eq4": float(2 * lnV * N[-1]),
                 "KV_over_Sreq_32k_eq4": float(2 * 4096 * 32768 / (2 * lnV * N[-1]))}

        def inv(Ncap, Nk):
            if Ncap <= Nk[0]:
                return float(T[0] * Ncap / Nk[0])
            if Ncap <= Nk[-1]:
                return float(np.exp(np.interp(np.log(Ncap), np.log(Nk), np.log(T))))
            bb = np.polyfit(np.log(T[-3:]), np.log(Nk[-3:]), 1)[0]
            return float(T[-1] * (Ncap / Nk[-1]) ** (1 / bb))
        b[Dn]["Tstar_oracle"] = {S: {"est": inv(S / (2 * lnV), N), "range_ci": [inv(S / (2 * lnV), hi), inv(S / (2 * lnV), lo)],
                                     "extrapolated": bool(S / (2 * lnV) > N[-1])} for S in [2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20]}
    b["Tstar_every_token"] = {S: S / (2 * lnV) for S in [2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20]}
    budget[corpus] = b
# fan-out correction from the replay experiment
corr = {}
for corpus, v in rep.items():
    for s in v["summary"]:
        corr.setdefault(corpus, []).append({k: s[k] for k in ["T", "shape", "def", "policy", "acc_max", "mean_n_ctx", "S_half_measured",
                                                            "S_req_predicted_eq4", "ratio_measured_over_predicted", "mean_abs_err_theory"]})
out["budget"] = budget; out["replay_summary"] = corr
for c, rr in corr.items():
    for s in rr:
        print(c, s)
json.dump(out, open(os.path.join(R, "analysis_v2.json"), "w"), indent=1, default=float)

# ---------------- figures
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for i, arch in enumerate(["linear", "gla", "delta", "gdelta"]):
    rows_ = [r for r in seedtab if r["arch"] == arch]
    ax[0].errorbar([r["dk"] * (1 + 0.03 * i) for r in rows_], [r["Nstar_mean"] for r in rows_],
                   yerr=[r["Nstar_sd"] or 0 for r in rows_], fmt="o", color=f"C{i}", label=arch, capsize=3)
dks = sorted(simN_ for simN_ in {k[0] for k in simN if k[1] == 64})
ax[0].plot(dks, [simN[(d, 64)][0] for d in dks], "k^:", label="simulated ideal additive")
ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("d_k (d_v=64)"); ax[0].set_ylabel("N* (mean ± sd over seeds)")
ax[0].legend(fontsize=7); ax[0].set_title("(a) trained capacity, 3 seeds")
for i, arch in enumerate(["linear", "gla", "delta", "gdelta"]):
    for r in ow.get(arch, []):
        ax[1].plot([int(k) for k in r["acc"]], list(r["acc"].values()), "o-" if r["seed"] == 0 else "s--", color=f"C{i}", ms=3,
                   label=f"{arch} (trained)" if r["seed"] == 0 else None)
for j, (name, v) in enumerate(ideal.items()):
    ax[1].plot(list(v.keys()), list(v.values()), ":", color="k", alpha=0.3 + 0.14 * j, label=f"ideal {name}")
ax[1].set_xscale("log"); ax[1].set_xlabel("live keys P (each written 3x)"); ax[1].set_ylabel("accuracy (latest value)")
ax[1].set_title("(b) overwrite task, d_k=16, d_v=64"); ax[1].legend(fontsize=6)
if "prose" in rep:
    for i, (corpus, ls) in enumerate([("prose", "-"), ("code", "--")]):
        if corpus not in rep: continue
        for s in rep[corpus]["summary"]:
            if s["def"] == "D3" and s["T"] == 4096 and s["shape"] == "square":
                Ss = [int(k) for k in s["acc_by_S"]]
                col = "C0" if s["policy"] == "every" else "C3"
                ax[2].plot(Ss, list(s["acc_by_S"].values()), "o" + ls, color=col, ms=4, label=f"{corpus} {s['policy']} (replay)")
                ax[2].plot(Ss, list(s["theory_by_S"].values()), ls, color="k", lw=0.8)
                ax[2].axvline(s["S_req_predicted_eq4"], color=col, ls=ls, lw=0.6)
    ax[2].set_xscale("log"); ax[2].set_xlabel("state S = d_k d_v (square)"); ax[2].set_ylabel("recall of N_ctx set (D3)")
    ax[2].set_title("(c) replay T=4096 (o); load-aware Eq.(1) black;\nnaive Eq.(4) S_req vertical; prose solid, code dashed", fontsize=9); ax[2].legend(fontsize=6)
plt.tight_layout(); plt.savefig(os.path.join(R, "fig_revision.png"), dpi=130)
