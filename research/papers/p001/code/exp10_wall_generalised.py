"""Exp 10 (revision 1): robustness of the data-wall model.  Responds to reviewer W5, W6, W7, M8, M9.

(a) Generalised self-similarity (Prop. 3'): with alpha != beta, define U_N as the compute-optimal model size for
    U_D unique tokens, U_N = (alpha A / (beta B))^(1/alpha) U_D^(beta/alpha)  (this is how Muennighoff et al.
    define U_N; k_N U_D is its alpha = beta special case).  Then every threshold scales as U^((alpha+beta)/alpha).
    We compare this 'consistent' rule with the 'linear' rule k_N U_D used in v0 (Exp. 5), over U = 1e8 ... 1e15.
(b) Uncertainty of the transfer exponent (alpha+beta)/alpha from the Exp. 1 bootstrap.
(c) Alternative functional forms: the Lovelace et al. (2026) additive-penalty laws and their own Muennighoff-D
    refit, with the parameters published in their repository (fits/standard and fits/wd_sweep/wd1.0).
    Thresholds in multiples of C_wall for U = 1e8, 1e9, 1e10 (inside their 5e7-6e9 token range) and 1e12, 1e14
    (extrapolated).  The frontier uses free disposal: L~_U(C) = min_{C' <= C} L*_U(C').
(d) Economic framing of the compute-efficiency loss (months of compute growth), and calendar with the
    Epoch AI 4.6e26 FLOP Grok-3 anchor.
"""
import json
import numpy as np
from common import RES

fit = json.load(open(f"{RES}/exp1_fit.json"))["fit240"]


def make_loss(base, form, prm, kN_rule="consistent", kN=0.051):
    E, A, a, B, b = base["E"], base["A"], base["alpha"], base["B"], base["beta"]
    nopt_c = (a * A / (b * B)) ** (1.0 / a)

    def loss(N, D, U):
        if np.isinf(U):
            return E + A * N ** -a + B * D ** -b
        UD = np.minimum(D, U); RD = np.maximum(0.0, D / UD - 1.0)
        if form == "muennighoff_dn" or form == "muennighoff_d":
            Rs = prm["RDs"]
            Dp = UD * (1 + Rs * (1 - np.exp(-RD / Rs)))
            if form == "muennighoff_dn":
                UN = np.minimum(N, nopt_c * UD ** (b / a) if kN_rule == "consistent" else kN * UD)
                RN = np.maximum(0.0, N / UN - 1.0)
                Np = UN * (1 + prm["RNs"] * (1 - np.exp(-RN / prm["RNs"])))
            else:
                Np = N
            return E + A * Np ** -a + B * Dp ** -b
        fresh = E + A * N ** -a + B * (UD * (1 + RD)) ** -b
        if form == "additive_1p":
            return fresh + prm["P"] * RD * N / UD
        if form == "additive_4p":
            pen = prm["P"] * np.maximum(RD, 1e-12) ** prm["delta"] * (N / UD ** prm["gamma"]) ** prm["kappa"]
            return fresh + np.where(RD > 0, pen, 0.0)
        raise ValueError(form)
    return loss


def thresholds(base, form, prm, U, kN_rule="consistent", npts=601, span=(-3, 5)):
    loss = make_loss(base, form, prm, kN_rule)
    E, a, b = base["E"], base["alpha"], base["beta"]
    # uncapped optimum: D*(C) ~ C^(a/(a+b)); C_wall where D* = U
    A_, B_ = base["A"], base["B"]
    G = (a * A_ / (b * B_)) ** (1.0 / (a + b))
    # D* = (C/6)/N*,  N* = G (C/6)^(b/(a+b))  => D* = (C/6)^(a/(a+b)) / G
    Cwall = 6.0 * (U * G) ** ((a + b) / a)
    Cs = Cwall * np.logspace(span[0], span[1], npts)
    lC = np.log(Cs)
    Nopt = G * (Cs / 6) ** (b / (a + b))
    lNg = np.log(Nopt)[:, None] + np.linspace(-9, 9, 3000)[None, :]

    def front(Uc):
        Ng = np.exp(lNg); Dg = Cs[:, None] / (6 * Ng)
        Lg = loss(Ng, Dg, Uc)
        Lmin = Lg.min(1)
        return np.minimum.accumulate(Lmin)  # free disposal
    Linf = front(np.inf); Lu = front(U); L2 = front(2 * U)

    def inv(Lc, t):
        return np.inf if t < Lc[-1] - 1e-15 else float(np.exp(np.interp(-t, -Lc, lC)))
    eta = np.array([inv(Linf, Lu[i]) / c for i, c in enumerate(Cs)])
    km = np.array([inv(Lu, L2[i]) / c for i, c in enumerate(Cs)])
    ginf = -np.gradient(np.log(Linf - E), lC); geff = -np.gradient(np.log(np.maximum(Lu - E, 1e-300)), lC)

    def first(cond):
        return float(Cs[np.argmax(cond)] / Cwall) if cond.any() else None
    return dict(U=U, C_wall=float(Cwall), C_wall_over_U2=float(Cwall / U ** 2), eff50=first(eta < 0.5),
                eff10=first(eta < 0.1), shadow_2x=first(km > 2.0), shadow_inf=first(np.isinf(km)),
                gamma_half=first(geff < 0.5 * ginf),
                eta_at_10x=float(np.interp(np.log(10.0), lC - np.log(Cwall), eta)),
                eta_at_100x=float(np.interp(np.log(100.0), lC - np.log(Cwall), eta)))


out = {}
EP = dict(E=fit["E"], A=fit["A"], alpha=fit["alpha"], B=fit["B"], beta=fit["beta"])
MU = dict(E=1.87, A=521.0, alpha=0.35, B=1488.0, beta=0.35)
MUR = dict(RDs=15.4, RNs=5.3)
Us = [1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]

# (a)
out["selfsim_epoch_base"] = {}
for rule in ["linear", "consistent"]:
    out["selfsim_epoch_base"][rule] = [thresholds(EP, "muennighoff_dn", MUR, U, rule) for U in Us]
    print(rule, [(f"{r['U']:.0e}", r["eff50"], r["shadow_inf"]) for r in out["selfsim_epoch_base"][rule]], flush=True)
out["muennighoff_base_check"] = [thresholds(MU, "muennighoff_dn", MUR, U, "linear") for U in [1e8, 1e11, 1e14]]
out["muennighoff_base_consistent"] = [thresholds(MU, "muennighoff_dn", MUR, U, "consistent") for U in [1e8, 1e11, 1e14]]
out["muennighoff_beta_0353"] = [thresholds(dict(MU, beta=0.353), "muennighoff_dn", MUR, U, rule)
                                for rule in ["linear", "consistent"] for U in [1e8, 1e14]]
print("MU check", [(r["eff50"], r["shadow_inf"]) for r in out["muennighoff_base_check"]])
print("MU beta .353", [(r["U"], r["eff50"]) for r in out["muennighoff_beta_0353"]])

# (b) transfer exponent uncertainty
bo = np.load(f"{RES}/exp1_boots.npy")
print("boots shape", bo.shape)
al, be = bo[:, 3], bo[:, 4]
tex = (al + be) / al
out["transfer_exponent"] = dict(point=float((fit["alpha"] + fit["beta"]) / fit["alpha"]),
                                p10=float(np.percentile(tex, 10)), p90=float(np.percentile(tex, 90)),
                                sd=float(tex.std()),
                                factor_over_6_decades_p10_p90=[float(10 ** (6 * (np.percentile(tex, 10) - (fit["alpha"] + fit["beta"]) / fit["alpha"]))),
                                                               float(10 ** (6 * (np.percentile(tex, 90) - (fit["alpha"] + fit["beta"]) / fit["alpha"])))])
print(out["transfer_exponent"])

# (c) Lovelace et al. forms (parameters copied from github.com/justinlovelace/data-constrained-scaling,
#     dcscaling/data/fits/{standard, wd_sweep/wd1.0}/{chinchilla_params, repetition_results}.json)
LOV = {
    "standard": dict(base=dict(E=1.8382623195648193, A=216.57994079589844, alpha=0.29993095993995667,
                               B=4964.4248046875, beta=0.42743393778800964),
                     muennighoff_d=dict(RDs=7.752259731292725), additive_1p=dict(P=0.023053960874676704),
                     additive_4p=dict(P=3.270826312018471e-07, delta=1.6745405197143555, kappa=1.3447744846343994,
                                      gamma=0.6355141401290894)),
    "wd1.0": dict(base=dict(E=2.0457119941711426, A=232.71905517578125, alpha=0.2973250150680542,
                            B=25509.951171875, beta=0.5254247784614563),
                  muennighoff_d=dict(RDs=13.968727111816406), additive_1p=dict(P=0.007071443367749453),
                  additive_4p=dict(P=0.003793648211285472, delta=1.5705057382583618, kappa=1.3711652755737305,
                                   gamma=1.0394004583358765)),
}
out["lovelace_forms"] = {}
for arm, d in LOV.items():
    out["lovelace_forms"][arm] = {}
    for form in ["muennighoff_d", "additive_1p", "additive_4p"]:
        rows = [thresholds(d["base"], form, d[form], U) for U in [1e8, 1e9, 1e10, 1e12, 1e14]]
        out["lovelace_forms"][arm][form] = rows
        print(arm, form, [(f"{r['U']:.0e}", r["eff50"], r["shadow_2x"], r["shadow_inf"]) for r in rows], flush=True)

# (d) economics and calendar
w4 = json.load(open(f"{RES}/exp4_data_wall.json"))
tab = {r["C"]: r for r in w4["caps"]["1e+14"]["table"]}
econ = []
for c in [1e28, 1e29, 1e30]:
    eta = tab[c]["compute_efficiency"]
    econ.append(dict(C=c, eta=eta, compute_penalty=1 / eta,
                     months_of_growth_at_g4=12 * np.log(1 / eta) / np.log(4), months_of_growth_at_g5=12 * np.log(1 / eta) / np.log(5)))
out["economics_U1e14"] = econ
cal = {}
for C0 in [1e26, 4.6e26]:
    for g in [3.0, 5.0]:
        key = f"C0={C0:.1e},g={g}"
        cal[key] = {}
        for U in ["1e+14", "3e+14"]:
            dd = w4["caps"][U]
            cal[key][U] = {k: float(2025 + np.log(dd[k] / C0) / np.log(g)) for k in ["C_wall", "C_efficiency_below_50pct"]}
out["calendar"] = cal
print(json.dumps(econ, indent=0)); print(json.dumps(cal, indent=0))
json.dump(out, open(f"{RES}/exp10_wall_generalised.json", "w"), indent=2)
