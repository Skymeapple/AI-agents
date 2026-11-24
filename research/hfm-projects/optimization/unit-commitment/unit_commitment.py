"""Day-ahead unit commitment under wind uncertainty, on public German data.

Public proxy for HFM's "Rolling Unit-Commitment and Dispatch Controller under Real
OPSD Load and Wind Sequences" track. Data: Open Power System Data time series
(2020-10-06 release), German hourly load, the ENTSO-E day-ahead load forecast and
actual wind generation:
https://data.open-power-system-data.org/time_series/2020-10-06/time_series_60min_singleindex.csv

It tests the idea in research/human-frontier-mission.md section 8: commit units
against wind *scenarios* drawn from past forecast errors (two-stage stochastic unit
commitment) instead of a fixed reserve rule.

Setup, each day:
  * at midnight the controller fixes the on/off schedule of 17 slow thermal units
    for the next 24 hours, knowing only forecasts;
  * a real-time linear programme then dispatches those units, a fast peaker,
    storage, wind curtailment and (at 3000 EUR/MWh) load shedding against what
    actually happened. Real cost is what that programme reports.
The fleet, costs and storage are invented but German-scale, and are the same for
every controller. The wind forecast is a simple causal one (last observation
decaying to the 30-day mean) because this OPSD release has no wind forecast.

Controllers:
  reserve rule   deterministic commitment on the forecast, with a reserve of
                 rho * forecast wind + 3% of load; rho chosen on 2018 dev days
  stochastic     commitment minimising expected cost over wind scenarios built
                 from the previous 60 days of forecast errors, either drawn at
                 random or spread evenly from the worst to the best past error
  foresight      commitment with the actual wind and load known (lower bound)

HFM's unit data, verifier and reward are private; numbers are NOT comparable to
HFM's reward. Regret = cost above the perfect-foresight cost.

Usage: python research/hfm-experiments/unit_commitment.py <time_series_60min_singleindex.csv>
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

T = 24
# (count, size GW, min stable share, cost kEUR/GWh, start-up cost kEUR per unit)
FLEET = [(2, 4.0, 0.6, 10.0, 400.0), (4, 4.0, 0.5, 25.0, 240.0), (5, 4.0, 0.4, 40.0, 200.0), (6, 4.0, 0.4, 55.0, 120.0)]
UNITS = [(size, share * size, cost, su) for n, size, share, cost, su in FLEET for _ in range(n)]
N = len(UNITS)
PEAK_GW, PEAK_COST = 6.0, 150.0
STO_P, STO_E, ETA = 6.0, 36.0, 0.9
VOLL = 3000.0
TIME_LIMIT = 20.0


class Index:
    """Variable layout: commitment u, start-ups s, then one dispatch block per scenario."""

    def __init__(self, S):
        self.S = S
        self.u = lambda i, t: i * T + t
        self.s = lambda i, t: N * T + i * T + t
        base = 2 * N * T
        self.block = N * T + 6 * T  # p, peaker, charge, discharge, soc, curtail, shed
        self.p = lambda k, i, t: base + k * self.block + i * T + t
        self.x = lambda k, j, t: base + k * self.block + N * T + j * T + t  # j: 0 peak,1 chg,2 dis,3 soc,4 curt,5 shed
        self.n = base + S * self.block


def solve(load, wind, u_prev, reserve=None, fixed_u=None, relax=False):
    """load, wind: arrays (S, T). Returns (commitment N x T, objective)."""
    S = load.shape[0]
    ix = Index(S)
    c = np.zeros(ix.n)
    lb, ub = np.zeros(ix.n), np.full(ix.n, np.inf)
    integrality = np.zeros(ix.n)
    for i, (pmax, pmin, cost, su) in enumerate(UNITS):
        for t in range(T):
            ub[ix.u(i, t)] = 1
            ub[ix.s(i, t)] = 1
            integrality[ix.u(i, t)] = 0 if relax else 1
            c[ix.s(i, t)] = su
            if fixed_u is not None:
                lb[ix.u(i, t)] = ub[ix.u(i, t)] = fixed_u[i, t]
            for k in range(S):
                c[ix.p(k, i, t)] = cost / S
    for k in range(S):
        for t in range(T):
            ub[ix.x(k, 0, t)] = PEAK_GW
            c[ix.x(k, 0, t)] = PEAK_COST / S
            ub[ix.x(k, 1, t)] = ub[ix.x(k, 2, t)] = STO_P
            ub[ix.x(k, 3, t)] = STO_E
            c[ix.x(k, 5, t)] = VOLL / S
    rows = N * T + S * (2 * N * T + 2 * T + 1) + (T if reserve is not None else 0)
    A = lil_matrix((rows, ix.n))
    lo, hi = np.full(rows, -np.inf), np.full(rows, np.inf)
    r = 0
    for i in range(N):  # start-up definition: s >= u_t - u_{t-1}
        for t in range(T):
            A[r, ix.s(i, t)] = 1
            A[r, ix.u(i, t)] = -1
            if t:
                A[r, ix.u(i, t - 1)] = 1
                lo[r] = 0
            else:
                lo[r] = -u_prev[i]
            r += 1
    for k in range(S):
        for i, (pmax, pmin, _, _) in enumerate(UNITS):
            for t in range(T):
                A[r, ix.p(k, i, t)] = 1; A[r, ix.u(i, t)] = -pmax; hi[r] = 0; r += 1
                A[r, ix.p(k, i, t)] = 1; A[r, ix.u(i, t)] = -pmin; lo[r] = 0; r += 1
        for t in range(T):  # power balance
            for i in range(N):
                A[r, ix.p(k, i, t)] = 1
            A[r, ix.x(k, 0, t)] = 1; A[r, ix.x(k, 2, t)] = 1; A[r, ix.x(k, 1, t)] = -1
            A[r, ix.x(k, 4, t)] = -1; A[r, ix.x(k, 5, t)] = 1
            lo[r] = hi[r] = load[k, t] - wind[k, t]
            r += 1
        for t in range(T):  # storage: soc_t = soc_{t-1} + eta*chg - dis/eta, starting half full
            A[r, ix.x(k, 3, t)] = 1; A[r, ix.x(k, 1, t)] = -ETA; A[r, ix.x(k, 2, t)] = 1 / ETA
            if t:
                A[r, ix.x(k, 3, t - 1)] = -1
                lo[r] = hi[r] = 0
            else:
                lo[r] = hi[r] = STO_E / 2
            r += 1
        A[r, ix.x(k, 3, T - 1)] = 1; lo[r] = STO_E / 2; r += 1  # end the day half full
    if reserve is not None:
        for t in range(T):
            for i, (pmax, _, _, _) in enumerate(UNITS):
                A[r, ix.u(i, t)] = pmax
            lo[r] = min(load[0, t] - wind[0, t] + reserve[t], sum(u[0] for u in UNITS))
            r += 1
    res = milp(c, constraints=LinearConstraint(A.tocsr()[:r], lo[:r], hi[:r]), integrality=integrality,
               bounds=Bounds(lb, ub), options={"time_limit": TIME_LIMIT, "mip_rel_gap": 2e-3})
    if res.x is None:
        raise RuntimeError(res.message)
    u = np.round(res.x[:N * T]).reshape(N, T)
    return u, res.fun


def real_cost(u, load, wind, u_prev):
    """Real-time dispatch with the commitment fixed and actual load and wind."""
    _, cost = solve(load[None], wind[None], u_prev, fixed_u=u)
    return cost


def wind_forecast(wind_hist, clim, phi=0.97):
    """Causal forecast made at midnight: last observation decaying to the 30-day mean."""
    return clim + phi ** np.arange(1, T + 1) * (wind_hist[-1] - clim)


def run(days, data, rho=None, mode="reserve", n_scen=10, seed=0):
    rng = np.random.default_rng(seed)
    load_a, load_f, wind_a = data
    u_prev = np.ones(N)
    costs = []
    for d in days:
        sl = slice(d * T, (d + 1) * T)
        la, lf, wa = load_a[sl], load_f[sl], wind_a[sl]
        clim = wind_a[(d - 30) * T:d * T].mean()
        wf = wind_forecast(wind_a[:d * T], clim)
        if mode == "foresight":
            u, _ = solve(la[None], wa[None], u_prev)
        elif mode == "reserve":
            u, _ = solve(lf[None], wf[None], u_prev, reserve=rho * wf + 0.03 * lf)
        else:
            errs = []
            for e in range(d - 60, d):  # past errors of the same forecast rule
                c_e = wind_a[(e - 30) * T:e * T].mean()
                errs.append(wind_a[e * T:(e + 1) * T] - wind_forecast(wind_a[:e * T], c_e))
            if mode == "stochastic-spread":  # evenly spaced by mean error, extremes always included
                order = np.argsort(np.mean(errs, axis=1))
                pick = order[np.linspace(0, len(errs) - 1, n_scen).round().astype(int)]
            else:
                pick = rng.choice(len(errs), size=n_scen, replace=False)
            ws = np.clip(wf[None] + np.array(errs)[pick], 0, None)
            u, _ = solve(np.repeat(lf[None], n_scen, 0), ws, u_prev)
        costs.append(real_cost(u, la, wa, u_prev))
        u_prev = u[:, -1]
    return np.array(costs)


def main():
    df = pd.read_csv(sys.argv[1], usecols=["utc_timestamp", "DE_load_actual_entsoe_transparency",
                                           "DE_load_forecast_entsoe_transparency", "DE_wind_generation_actual"],
                     parse_dates=["utc_timestamp"]).set_index("utc_timestamp")
    df = df["2017-10-01":"2019-12-31"].interpolate(limit=6).dropna() / 1000.0  # GW
    t0 = df.index[0]
    data = (0.9 * df.iloc[:, 0].values, 0.9 * df.iloc[:, 1].values, df.iloc[:, 2].values)
    day_of = lambda s: int((pd.Timestamp(s, tz="UTC") - t0) / pd.Timedelta(days=1))
    dev = list(range(day_of("2018-01-08"), day_of("2018-01-08") + 14)) + list(range(day_of("2018-07-09"), day_of("2018-07-09") + 14))
    test = [d for s in ("2019-01-14", "2019-03-11", "2019-05-13", "2019-07-15", "2019-09-16", "2019-11-11")
            for d in range(day_of(s), day_of(s) + 7)]

    print(f"dev: {len(dev)} days in 2018; test: {len(test)} days in six weeks of 2019. Costs in million EUR.")
    best = None
    for rho in (0.0, 0.2, 0.4, 0.6, 0.8):
        c = run(dev, data, rho=rho).sum() / 1e3
        print(f"  dev  reserve rule rho={rho:.1f}: {c:8.1f}", flush=True)
        if best is None or c < best[0]:
            best = (c, rho)
    rho = best[1]
    print(f"chosen on dev: rho={rho}")

    pf = run(test, data, mode="foresight")
    res = {"reserve rule, rho=0 (no wind reserve)": run(test, data, rho=0.0),
           f"reserve rule, rho={rho} (tuned on dev)": run(test, data, rho=rho),
           "stochastic, 10 scenarios": run(test, data, mode="stochastic", n_scen=10),
           "stochastic, 10 scenarios (other draw)": run(test, data, mode="stochastic", n_scen=10, seed=1),
           "stochastic, 20 scenarios": run(test, data, mode="stochastic", n_scen=20),
           "stochastic, 10 spread scenarios": run(test, data, mode="stochastic-spread", n_scen=10)}
    ref = res[f"reserve rule, rho={rho} (tuned on dev)"]
    print(f"\n{'controller':<40} {'cost':>9} {'regret':>9} {'regret vs tuned rule':>21} {'weeks cheaper than tuned rule':>30}")
    print(f"{'perfect foresight (lower bound)':<40} {pf.sum() / 1e3:9.1f} {0:9.1f}")
    for k, c in res.items():
        reg, reg_ref = (c - pf).sum(), (ref - pf).sum()
        weeks = sum(c[i:i + 7].sum() < ref[i:i + 7].sum() - 1e-6 for i in range(0, len(test), 7))
        print(f"{k:<40} {c.sum() / 1e3:9.1f} {reg / 1e3:9.1f} {reg / reg_ref - 1:>+21.1%} {weeks:>26}/6")


if __name__ == "__main__":
    main()
