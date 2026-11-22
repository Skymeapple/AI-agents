"""Islanded AI-campus supply model: hourly LP capacity expansion with endogenous compute curtailment.

Normalized to a flat facility load L = 1 MW (results scale linearly). Decision variables:
capacities S (PV, MWac), W (wind), Pb/Eb (Li-ion power/energy), Gct, Gcc (gas CT / CC), N (firm clean, e.g. SMR);
hourly dispatch gct, gcc, c (charge), d (discharge), e (state of charge), u (curtailed compute load), n (firm output).
Constraints: hourly balance, cyclic storage, capacity limits, carbon-free-energy (CFE) share of *served* energy >= phi.
Objective: annualized capex + FOM + fuel/VOM + v * curtailed MWh (v = value of lost compute, $/MWh).
All cost inputs come from NREL ATB 2024 (v3, Moderate, Market case, 2030 cost year) except where noted (see params())."""
import numpy as np, scipy.sparse as sp
from scipy.optimize import linprog

def crf(r, n):
    return r * (1 + r) ** n / ((1 + r) ** n - 1)

def params(wacc=0.07, gas_price=3.5, firm_capex=9650.0, firm_fom=136.0, firm_var=13.6, overrides=None):
    # ATB 2024 v3 Moderate 2030 CAPEX ($/kW) and FOM ($/kW-yr); battery power/energy split derived from 2h & 4h entries
    b2, b4 = 941.975603, 1451.315621          # $/kW for 2h and 4h Li-ion
    e_cost = (b4 - b2) / 2.0                     # $/kWh   -> 254.7
    p_cost = b2 - 2 * e_cost                     # $/kW    -> 432.6
    f2, f4 = 20.219209, 32.503818
    e_fom = (f4 - f2) / 2.0; p_fom = f2 - 2 * e_fom
    P = dict(
        pv=1193.480944e3 * crf(wacc, 30) + 17.990681e3,
        wind=1407.953224e3 * crf(wacc, 30) + 29.263773e3,
        bat_p=p_cost * 1e3 * crf(wacc, 15) + p_fom * 1e3,
        bat_e=e_cost * 1e3 * crf(wacc, 15) + e_fom * 1e3,
        ct=1270.475989e3 * crf(wacc, 30) + 25.0e3,
        cc=1668.857856e3 * crf(wacc, 30) + 37.6e3,
        firm=firm_capex * 1e3 * crf(wacc, 40) + firm_fom * 1e3,
        ct_var=9.717 * gas_price + 6.94, cc_var=6.040 * gas_price + 2.42,
        firm_var=firm_var, ct_hr=9.717, cc_hr=6.040,
        eta_c=np.sqrt(0.85), eta_d=np.sqrt(0.85), firm_avail=0.93,
        co2_per_mmbtu=0.05306,  # tCO2/MMBtu natural gas (EPA emission factor)
    )
    if overrides: P.update(overrides)
    return P

def solve(cs, cw, P, phi=0.0, v=1000.0, L=1.0, allow=("pv", "wind", "bat", "ct", "cc", "firm"),
          fixed_caps=None, max_curtail_frac=None, carbon_price=0.0, return_series=False, dt=1):
    if dt > 1:  # time aggregation: average capacity factors over dt-hour blocks
        cs = cs[: len(cs) // dt * dt].reshape(-1, dt).mean(1); cw = cw[: len(cw) // dt * dt].reshape(-1, dt).mean(1)
    T = len(cs); nb = 7
    iS, iW, iPb, iEb, iCT, iCC, iN = range(7)
    H = lambda k: nb + k * T + np.arange(T)  # hourly blocks: 0 gct,1 gcc,2 c,3 d,4 e,5 u,6 n
    nv = nb + 7 * T
    cost = np.zeros(nv)
    cost[[iS, iW, iPb, iEb, iCT, iCC, iN]] = [P["pv"], P["wind"], P["bat_p"], P["bat_e"], P["ct"], P["cc"], P["firm"]]
    cost[H(0)] = P["ct_var"] + carbon_price * P["co2_per_mmbtu"] * P["ct_hr"]
    cost[H(1)] = P["cc_var"] + carbon_price * P["co2_per_mmbtu"] * P["cc_hr"]
    cost[H(2)] = 1e-4  # tiny charge cost to break degeneracy
    cost[H(5)] = v; cost[H(6)] = P["firm_var"]
    cost[nb:] *= dt  # hourly energy costs scale with step length
    bounds = [(0, None)] * nv
    tech_idx = {"pv": [iS], "wind": [iW], "bat": [iPb, iEb], "ct": [iCT], "cc": [iCC], "firm": [iN]}
    for k, ids in tech_idx.items():
        if k not in allow:
            for i in ids: bounds[i] = (0, 0)
    if fixed_caps is not None:
        for i, val in enumerate(fixed_caps): bounds[i] = (val, val)
    for t in H(5): bounds[t] = (0, L)
    rows, cols, vals, rhs_ub = [], [], [], []
    r = 0; ar = np.arange(T)
    # balance: -(S cs + W cw + n + gct + gcc + d - c + u) <= -L
    for blk, sgn in [(0, -1), (1, -1), (3, -1), (2, 1), (5, -1), (6, -1)]:
        rows.append(r + ar); cols.append(H(blk)); vals.append(np.full(T, sgn))
    rows += [r + ar, r + ar]; cols += [np.full(T, iS), np.full(T, iW)]; vals += [-cs, -cw]
    rhs_ub.append(np.full(T, -L)); r += T
    # capacity limits
    fa = P.get("firm_avail_series")  # optional hourly availability of the firm fleet (lumpy module outages)
    if fa is not None:
        fa = np.asarray(fa, float); fa = fa[: len(fa) // dt * dt].reshape(-1, dt).mean(1) if dt > 1 else fa
    else:
        fa = np.full(T, P["firm_avail"])
    for blk, cap, coef in [(0, iCT, 1.0), (1, iCC, 1.0), (2, iPb, 1.0), (3, iPb, 1.0), (4, iEb, 1.0), (6, iN, fa)]:
        rows += [r + ar, r + ar]; cols += [H(blk), np.full(T, cap)]; vals += [np.ones(T), -np.broadcast_to(coef, (T,)).astype(float)]
        rhs_ub.append(np.zeros(T)); r += T
    # CFE share: sum g + (1-phi) sum u <= (1-phi) T L
    rows += [np.full(T, r), np.full(T, r), np.full(T, r)]; cols += [H(0), H(1), H(5)]
    vals += [np.ones(T), np.ones(T), np.full(T, 1 - phi)]; rhs_ub.append([(1 - phi) * T * L]); r_cfe = r; r += 1
    if max_curtail_frac is not None:
        rows.append(np.full(T, r)); cols.append(H(5)); vals.append(np.ones(T)); rhs_ub.append([max_curtail_frac * T * L]); r += 1
    A_ub = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(r, nv))
    b_ub = np.concatenate([np.atleast_1d(np.asarray(x, float)) for x in rhs_ub])
    # storage dynamics: e_t - e_{t-1} - eta_c c_t + d_t/eta_d = 0 (cyclic)
    prev = np.roll(ar, 1)
    er = [ar, ar, ar, ar]; ec = [H(4), H(4)[prev], H(2), H(3)]
    ev = [np.ones(T), -np.ones(T), np.full(T, -P["eta_c"] * dt), np.full(T, dt / P["eta_d"])]
    A_eq = sp.csr_matrix((np.concatenate(ev), (np.concatenate(er), np.concatenate(ec))), shape=(T, nv))
    res = linprog(cost, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=np.zeros(T), bounds=bounds, method="highs")
    if res.status != 0:
        return {"status": res.status, "message": res.message}
    x = res.x; caps = x[:nb]
    g = x[H(0)] + x[H(1)]; u = x[H(5)]; served = (T * L - u.sum()) * dt
    fuel_mmbtu = (x[H(0)] * P["ct_hr"] + x[H(1)] * P["cc_hr"]).sum() * dt
    supply_cost = res.fun - v * u.sum() * dt - carbon_price * P["co2_per_mmbtu"] * fuel_mmbtu
    duals = res.ineqlin.marginals
    out = dict(status=0, total_cost=res.fun, supply_cost=supply_cost, caps=dict(zip(["pv", "wind", "bat_p", "bat_e", "ct", "cc", "firm"], caps.tolist())),
               served_frac=served / (T * L * dt), curtail_frac=u.sum() / (T * L), curtail_hours=int((u > 1e-6).sum()) * dt,
               gas_share=g.sum() * dt / served, cfe=1 - g.sum() * dt / served,
               lcoe_served=supply_cost / served,  # $/MWh of served compute energy
               co2_t_per_mwh=P["co2_per_mmbtu"] * fuel_mmbtu / served,
               cfe_dual=-duals[r_cfe] / dt, max_price=float((-duals[:T]).max()) / dt,
               spill_frac=float((caps[iS] * cs + caps[iW] * cw + x[H(6)] + g + x[H(3)] - x[H(2)] + u - L).sum() / (T * L)))
    if return_series:
        out["series"] = dict(u=u, g=g, d=x[H(3)], c=x[H(2)], e=x[H(4)], n=x[H(6)], price=-duals[:T] / dt)
    return out

if __name__ == "__main__":
    import time
    d = np.load(__import__("os").path.join(__import__("os").path.dirname(__file__), "..", "data", "cf_abilene_tx_2012.npz"))
    P = params()
    for k in ["pv", "wind", "bat_p", "bat_e", "ct", "cc", "firm"]: print(k, round(P[k]))
    for phi, v in [(0.0, 1000), (0.9, 1000), (1.0, 1000), (1.0, 300)]:
        t0 = time.time(); o = solve(d["solar"], d["wind"], P, phi=phi, v=v)
        print(phi, v, "%.1fs" % (time.time() - t0), {k: (round(val, 3) if isinstance(val, float) else val) for k, val in o.items() if k != "caps"}, {k: round(val, 3) for k, val in o["caps"].items()})
