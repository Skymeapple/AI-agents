"""Exp 4: A genuine break -- the data wall -- under the data-constrained scaling law of
Muennighoff et al. (2023, NeurIPS), eq. (17) of arXiv:2305.16264:
  L = 521/N'^0.35 + 1488/D'^0.35 + 1.87
  D' = U_D + U_D R_D*(1-exp(-R_D/R_D*)),  R_D = max(0, D/U_D - 1),  R_D* = 15.4
  N' = U_N + U_N R_N*(1-exp(-R_N/R_N*)),  R_N = max(0, N/U_N - 1),  R_N* = 5.3,  U_N = min(N, 0.051 U_D)
We compute the compute-optimal frontier with a cap U on unique tokens, the excess loss vs. the
uncapped law, the effective local exponent of reducible loss, and the *compute-equivalent value*
of doubling unique data (shadow price)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RES

A, B, E, a, b = 521.0, 1488.0, 1.87, 0.35, 0.35
RDs, RNs, kN = 15.4, 5.3, 0.051


def loss(N, D, U):
    if np.isinf(U):
        return E + A * N ** -a + B * D ** -b
    UD = np.minimum(D, U)
    RD = np.maximum(0.0, D / UD - 1.0)
    Dp = UD + UD * RDs * (1 - np.exp(-RD / RDs))
    UN = np.minimum(N, kN * UD)
    RN = np.maximum(0.0, N / UN - 1.0)
    Np = UN + UN * RNs * (1 - np.exp(-RN / RNs))
    return E + A * Np ** -a + B * Dp ** -b


lNgrid = np.linspace(np.log(1e7), np.log(1e17), 6000)


def frontier(C, U):
    Ng = np.exp(lNgrid)[None, :]
    Dg = C[:, None] / (6 * Ng)
    Lg = loss(Ng, Dg, U)
    j = Lg.argmin(1)
    return Lg[np.arange(len(C)), j], Ng[0, j], Dg[np.arange(len(C)), j]


if __name__ == "__main__":
    Cs = np.logspace(20, 33, 521)
    L_inf, N_inf, D_inf = frontier(Cs, np.inf)
    red_inf = L_inf - E
    g_inf = -np.gradient(np.log(red_inf), np.log(Cs))
    Us = [3e13, 1e14, 3e14, 1e15]
    out = dict(law="Muennighoff et al. 2023 eq.17", gamma_uncapped=float(np.median(g_inf)), caps={})
    curves = {}
    for U in Us:
        Lu, Nu, Du = frontier(Cs, U)
        curves[U] = (Lu, Nu, Du)
        exc = Lu - L_inf
        frac = exc / red_inf
        g_eff = -np.gradient(np.log(Lu - E), np.log(Cs))
        # floor as C -> inf: N,D -> inf, D' -> U(1+R_D*), N' -> kN U (1+R_N*)
        floor = E + A * (kN * U * (1 + RNs)) ** -a + B * (U * (1 + RDs)) ** -b
        # compute at which uncapped optimal D equals U (the "wall")
        Cwall = float(np.exp(np.interp(np.log(U), np.log(D_inf), np.log(Cs))))
        def first(cond):
            i = np.argmax(cond)
            return float(Cs[i]) if cond.any() else None
        # shadow price: k such that L*_U(kC) = L*_{2U}(C)
        L2, _, _ = frontier(Cs, 2 * U)
        # Lu is monotone decreasing in C: invert by log-interpolation; unattainable if target < min over grid
        lCs = np.log(Cs)
        def inv(Lcurve, target):
            # returns C with Lcurve(C)=target (log-interp), inf if target below curve's range
            if target < Lcurve[-1]:
                return np.inf
            return float(np.exp(np.interp(-target, -Lcurve, lCs)))
        kmult = np.array([inv(Lu, L2[i]) / c for i, c in enumerate(Cs)])
        # compute efficiency: uncapped compute that achieves the capped loss, divided by actual compute
        eta = np.array([inv(L_inf, Lu[i]) / c for i, c in enumerate(Cs)])
        tab = []
        for c in [1e24, 1e25, 1e26, 1e27, 1e28, 1e29, 1e30]:
            i = int(np.argmin(np.abs(np.log(Cs) - np.log(c))))
            tab.append(dict(C=c, L_capped=float(Lu[i]), L_uncapped=float(L_inf[i]), excess=float(exc[i]),
                            excess_frac_of_reducible=float(frac[i]), gamma_eff=float(g_eff[i]),
                            epochs=float(Du[i] / U), tokens_per_param=float(Du[i] / Nu[i]),
                            compute_equiv_of_doubling_U=(None if np.isinf(kmult[i]) else float(kmult[i])),
                            compute_efficiency=float(eta[i])))
        out["caps"][f"{U:.0e}"] = dict(
            U=U, C_wall=Cwall, floor=float(floor), floor_minus_E=float(floor - E),
            C_excess_1pct_reducible=first(frac > 0.01), C_excess_5pct_reducible=first(frac > 0.05),
            C_excess_20pct_reducible=first(frac > 0.20),
            C_gamma_eff_half=first(g_eff < 0.5 * np.median(g_inf)),
            C_efficiency_below_50pct=first(eta < 0.5), C_efficiency_below_10pct=first(eta < 0.1),
            C_shadow_price_infinite=first(np.isinf(kmult)), table=tab)
        curves[U] = (Lu, g_eff, kmult, eta)

    # calendar mapping (illustrative): C(t) = C0 * g^(t-2025)
    cal = {}
    for C0 in [1e26, 5e26]:
        for gr in [3.0, 5.0]:
            key = f"C0={C0:.0e},g={gr}"
            cal[key] = {}
            for U in Us:
                d = out["caps"][f"{U:.0e}"]
                cal[key][f"{U:.0e}"] = {kk: (None if d[kk] is None else float(2025 + np.log(d[kk] / C0) / np.log(gr)))
                                        for kk in ["C_wall", "C_efficiency_below_50pct", "C_gamma_eff_half", "C_shadow_price_infinite"]}
    out["calendar_illustration"] = cal
    json.dump(out, open(f"{RES}/exp4_data_wall.json", "w"), indent=2)
    print(json.dumps({k: {kk: v for kk, v in d.items() if kk != 'table'} for k, d in out["caps"].items()}, indent=1))
    print(json.dumps(cal, indent=1))

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    ax[0].loglog(Cs, L_inf - E, "k", label="no cap")
    for U in Us:
        ax[0].loglog(Cs, curves[U][0] - E, label=f"U={U:.0e} tokens")
    ax[0].set_xlabel("training FLOP"); ax[0].set_ylabel("reducible loss  $L^*-E$"); ax[0].legend(fontsize=7)
    ax[1].semilogx(Cs, g_inf, "k")
    for U in Us:
        ax[1].semilogx(Cs, curves[U][1])
    ax[1].set_xlabel("training FLOP"); ax[1].set_ylabel(r"local exponent of reducible loss $\gamma_{\rm eff}$")
    ax[1].set_title("genuine break: exponent collapses past the wall", fontsize=9)
    for i_, U in enumerate(Us):
        km = curves[U][2].copy(); km[np.isinf(km)] = np.nan
        ax[2].loglog(Cs, km, color=f"C{i_}", label=f"U={U:.0e}: value of 2x data")
        ax[2].loglog(Cs, 1 / curves[U][3], color=f"C{i_}", ls=":")
    ax[2].set_xlabel("training FLOP"); ax[2].set_ylabel("compute multiplier")
    ax[2].set_title("solid: compute-equiv. of doubling U (gap=infinite)\ndotted: 1/compute efficiency", fontsize=8); ax[2].legend(fontsize=6)
    plt.tight_layout(); plt.savefig(f"{RES}/fig4_data_wall.png", dpi=150)
