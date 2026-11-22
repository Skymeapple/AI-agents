"""Experiment 2: equilibrium analysis of the training-data commons.

Produces: results/exp2_equilibrium.json, fig2_statics.png, fig3_freeriding.png,
          fig4_regimes.png
Baseline parameters (illustrative, not estimated): M=1000, pi0=0.6, uniform costs on
[0,1], sigma=0.3, learner phi=0.95, s2=1000 (same world as Experiment 1).
A is reported as a = A / R0 where R0 = pi0*M*F(pi0) is creators' market revenue
absent AI.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from commons import Commons

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
BASE = dict(M=1000.0, pi0=0.6, sigma=0.3, phi=0.95, s2=1000.0)
R0 = BASE["pi0"] * BASE["M"] * min(BASE["pi0"], 1.0)


def model(a=20.0, **kw):
    p = dict(BASE); p.update(kw)
    return Commons(A=a * R0, **p)


def main():
    res = {"base": BASE, "R0": R0}
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---------------- (A) comparative statics & dilution multiplier (Prop. 1)
    sigmas = np.linspace(0, 0.6, 13)
    st = {"sigma": sigmas.tolist(), "N0": [], "q0": [], "price0": [], "mult_analytic": [],
          "mult_numeric": [], "attenuation": []}
    for s in sigmas:
        m = model(sigma=s)
        N0 = m.N_star(0.0)
        mult, _ = m.multiplier(0.0)
        st["N0"].append(N0); st["q0"].append(float(m.H(N0))); st["price0"].append(float(m.price(N0, 0)))
        st["mult_analytic"].append(float(mult)); st["mult_numeric"].append(float(m.dN_dT(0.0)))
        st["attenuation"].append(float(mult / m.M))  # relative to no-dilution pass-through M f = M
    res["statics"] = st
    res["noAI_N"] = BASE["M"] * BASE["pi0"]
    print("statics: N0 at sigma=0.3:", st["N0"][6], "q0:", st["q0"][6])

    # ---------------- (B) industry profit vs T, voluntary threshold A1 (Prop. 2)
    m = model()
    res["A1_over_R0"] = m.A1() / R0
    Ts = np.linspace(0, 2 * m.dist_scale(), 60)
    prof = {}
    for a in [3, 8, 20, 40]:
        mm = model(a=a)
        prof[str(a)] = [mm.industry_profit(T) - mm.industry_profit(0.0) for T in Ts]
    res["profit_curves"] = {"T": Ts.tolist(), "delta_profit": prof}

    # ---------------- (C) free riding: voluntary Nash vs K (Prop. 3)
    fr = {"K": list(range(1, 11)), "a": [8, 20, 40], "T_nash": {}, "Kbar": {}, "T_joint": {}}
    for a in fr["a"]:
        mm = model(a=a)
        fr["Kbar"][str(a)] = mm.Kbar()
        fr["T_joint"][str(a)] = mm.T_joint()
        fr["T_nash"][str(a)] = [mm.T_nash(K) for K in fr["K"]]
        print("a=", a, "Kbar=", round(mm.Kbar(), 2), "Tnash=", np.round(fr["T_nash"][str(a)], 3))
    # check: threshold prediction (T_nash>0 iff K<Kbar)
    mism = 0
    for a in fr["a"]:
        for K, T in zip(fr["K"], fr["T_nash"][str(a)]):
            pred = K < fr["Kbar"][str(a)]
            mism += int(pred != (T > 1e-4))
    fr["threshold_mismatches"] = mism
    res["freeriding"] = fr

    # Kbar map over (sigma, a)
    agrid = np.geomspace(2, 100, 25); sgrid = np.linspace(0.0, 0.6, 13)
    Kmap = np.zeros((len(sgrid), len(agrid)))
    A1map = np.zeros(len(sgrid))
    for i, s in enumerate(sgrid):
        mm = model(sigma=s)
        A1map[i] = mm.A1() / R0
        Kmap[i] = agrid / A1map[i]
    res["Kbar_map"] = {"a": agrid.tolist(), "sigma": sgrid.tolist(), "A1_over_R0": A1map.tolist()}

    # ---------------- (D) regimes at baseline a=20, K=5 (Prop. 4, 5)
    K = 5
    regimes = {}
    mm = model()
    def summarize(name, T, share=1.0, firms_pay=True):
        N = mm.N_star(T, share); q = float(mm.H(share * N))
        CS = mm.creator_surplus(T, share)
        ind = mm.A * q - (T * N if firms_pay else 0.0)
        W = mm.welfare(T, share)
        regimes[name] = dict(T=T, data_share=share, N=N, q=q, creator_surplus=CS,
                             industry_profit=ind, per_firm_profit=ind / K, welfare=W)
    summarize("no_AI_benchmark", 0.0, share=0.0)
    regimes["no_AI_benchmark"]["note"] = "AI trained on no human works (q=H(0)=0)"
    summarize("fair_use", 0.0)
    Tn = mm.T_nash(K); summarize("voluntary_K5", Tn)
    Tj = mm.T_joint(); summarize("industry_levy_TJ", Tj)
    Tw = mm.T_welfare(); summarize("welfare_levy_TW", Tw)
    # individual opt-out: share omega opts out for non-pecuniary reasons
    omega = 0.3
    summarize("individual_optout_30pct", 0.0, share=1 - omega)
    # pecuniary value of a single opt-out (atomistic): income gain of one creator
    N0 = mm.N_star(0.0)
    single_gain = mm.pi0 * mm.sigma * float(mm.H(N0) - mm.H(N0 - 1))
    res["single_optout_income_gain"] = single_gain
    res["single_optout_gain_rel_income"] = single_gain / float(mm.price(N0, 0.0))
    # collective bargaining (CMO withholds all works; AI falls back to N_pd public-domain works)
    N_pd = 25.0
    q_d = float(mm.H(N_pd))
    Pi_d = mm.A * q_d
    # creators' disagreement surplus: market without inclusion but AI at quality q_d
    def CS_given_q(qfix, T=0.0):
        g = lambda p: p
        p = mm.pi0 * (1 - mm.sigma * qfix) + T
        return mm.M * (p * mm.dist.F(p) - mm.dist.partial_mean(p))
    CS_d = CS_given_q(q_d)
    Tg = np.linspace(0, 2 * mm.dist_scale(), 400)
    best = (-np.inf, None)
    feas = []
    for T in Tg:
        g1 = mm.industry_profit(T) - Pi_d
        g2 = mm.creator_surplus(T) - CS_d
        if g1 > 0 and g2 > 0:
            feas.append(T)
            if g1 * g2 > best[0]:
                best = (g1 * g2, T)
    res["bargaining"] = dict(N_pd=N_pd, q_disagree=q_d, feasible_T_min=min(feas) if feas else None,
                             feasible_T_max=max(feas) if feas else None, T_nash_bargain=best[1])
    if best[1] is not None:
        summarize("collective_bargain", best[1])
    res["regimes_a20_K5"] = regimes
    # Pareto set relative to voluntary NE: T such that every firm and creators gain
    base_firm = regimes["voluntary_K5"]["per_firm_profit"]; base_cs = regimes["voluntary_K5"]["creator_surplus"]
    par = [T for T in Tg if (mm.industry_profit(T) / K > base_firm + 1e-9) and (mm.creator_surplus(T) > base_cs + 1e-9)]
    res["pareto_vs_voluntary"] = dict(T_min=min(par) if par else None, T_max=max(par) if par else None)
    print(json.dumps(regimes, indent=1, default=float))
    print("bargaining", res["bargaining"], "pareto", res["pareto_vs_voluntary"])

    # regimes across a for summary figure
    reg_a = {"a": [3, 8, 20, 40], "rows": {}}
    for a in reg_a["a"]:
        m2 = model(a=a)
        rows = {}
        for nm, T in [("fair_use", 0.0), ("voluntary_K5", m2.T_nash(5)), ("industry_levy_TJ", m2.T_joint()),
                      ("welfare_levy_TW", m2.T_welfare())]:
            rows[nm] = dict(T=T, CS=m2.creator_surplus(T), Pi=m2.industry_profit(T), W=m2.welfare(T),
                            q=float(m2.H(m2.N_star(T))))
        reg_a["rows"][str(a)] = rows
    res["regimes_by_a"] = reg_a
    json.dump(res, open(os.path.join(OUT, "exp2_equilibrium.json"), "w"), indent=1, default=float)

    # ---------------- figures
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(sigmas, st["N0"], "o-", label="fresh works N* (fair use)")
    ax[0].axhline(res["noAI_N"], ls=":", c="gray", label="no-AI benchmark")
    ax[0].set_xlabel("dilution σ"); ax[0].set_ylabel("works / period"); ax[0].legend(fontsize=8)
    ax[0].set_title("(a) commons depletion under fair use")
    ax2 = ax[0].twinx(); ax2.plot(sigmas, st["q0"], "s--", c="C3"); ax2.set_ylabel("AI quality q*", color="C3")
    ax[1].plot(sigmas, st["attenuation"], "o-", label="analytic Λ = 1/(1+Mfπ0σH')")
    ax[1].plot(sigmas, np.array(st["mult_numeric"]) / BASE["M"], "x", ms=8, label="numerical dN*/dT ÷ Mf")
    ax[1].set_xlabel("dilution σ"); ax[1].set_ylabel("pass-through of payments to supply")
    ax[1].legend(fontsize=8); ax[1].set_title("(b) dilution attenuates payments")
    for a, y in prof.items():
        ax[2].plot(Ts, np.array(y) / R0, label=f"a = A/R0 = {a}")
    ax[2].axhline(0, c="k", lw=0.5); ax[2].set_xlabel("total per-work payment T")
    ax[2].set_ylabel("Δ industry profit / R0"); ax[2].legend(fontsize=8)
    ax[2].set_title("(c) AI industry gains from paying (if a > a1)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_statics.png"), dpi=150)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a in fr["a"]:
        l, = ax[0].plot(fr["K"], fr["T_nash"][str(a)], "o-", label=f"a={a}: voluntary NE")
        ax[0].axhline(fr["T_joint"][str(a)], ls="--", c=l.get_color(), lw=1)
        ax[0].axvline(fr["Kbar"][str(a)], ls=":", c=l.get_color(), lw=1)
    ax[0].set_xlabel("number of frontier AI firms K"); ax[0].set_ylabel("total per-work payment T")
    ax[0].set_title("(a) voluntary payments vanish for K ≥ K̄ (dotted);\n dashed = industry-optimal T^J", fontsize=9)
    ax[0].legend(fontsize=8)
    for s, A1 in zip(sgrid, A1map):
        pass
    cs = ax[1].contourf(agrid, sgrid, np.log10(np.maximum(Kmap, 1e-2)), levels=np.linspace(-1, 2, 13), cmap="viridis")
    cl = ax[1].contour(agrid, sgrid, Kmap, levels=[1, 3, 5, 10], colors="w", linewidths=1)
    ax[1].clabel(cl, fmt=lambda v: f"K̄={v:g}", fontsize=8)
    ax[1].set_xscale("log"); ax[1].set_xlabel("AI-to-creator revenue ratio a"); ax[1].set_ylabel("dilution σ")
    fig.colorbar(cs, ax=ax[1], label="log10 K̄"); ax[1].set_title("(b) free-riding threshold K̄(a, σ)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_freeriding.png"), dpi=150)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    names = ["fair_use", "voluntary_K5", "industry_levy_TJ", "welfare_levy_TW"]
    lab = ["fair use", "voluntary (K=5)", "industry levy T^J", "welfare levy T^W"]
    x = np.arange(len(reg_a["a"])); w = 0.2
    for j, (nm, lb) in enumerate(zip(names, lab)):
        ax[0].bar(x + j * w, [reg_a["rows"][str(a)][nm]["CS"] / R0 for a in reg_a["a"]], w, label=lb)
        ax[1].bar(x + j * w, [(reg_a["rows"][str(a)][nm]["Pi"] - reg_a["rows"][str(a)]["fair_use"]["Pi"]) / R0 for a in reg_a["a"]], w, label=lb)
        ax[2].bar(x + j * w, [(reg_a["rows"][str(a)][nm]["W"] - reg_a["rows"][str(a)]["fair_use"]["W"]) / R0 for a in reg_a["a"]], w, label=lb)
    for k, t in enumerate(["creator surplus / R0", "Δ AI industry profit vs fair use / R0", "Δ welfare vs fair use / R0"]):
        ax[k].set_xticks(x + 1.5 * w); ax[k].set_xticklabels([f"a={a}" for a in reg_a["a"]]); ax[k].set_title(t, fontsize=9)
        ax[k].axhline(0, c="k", lw=0.5)
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_regimes.png"), dpi=150)


if __name__ == "__main__":
    main()
