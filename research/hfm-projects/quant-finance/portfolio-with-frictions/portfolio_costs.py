"""Long-only portfolios under trading costs: how should a signal be traded?

Experiment for HFM's "Cost-aware long-only portfolio construction under trading
frictions" track. HFM fixes a signal model (a ridge alpha) and lets agents change
how it is turned into trades; its current idea is to rebalance only when the
portfolio has drifted far enough from the target.

Data: Kenneth French's 49 US industry portfolios, daily value-weighted returns
(public, https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
These are indices of industries, not single stocks, and have no price limits or
halts, so this tests the trading rule, not HFM's backtester. Numbers are NOT
comparable with HFM's, and nothing here is investment advice.

Two signals, both known at the previous close:
  slow   12-month momentum, skipping the last month
  fast   the same plus 5-day reversal (changes quickly, so costs bite)
Target portfolio: equal weight in the 10 best-ranked industries.

Ways to trade towards the target (cost charged per unit traded, one way):
  daily            trade fully to the target every day
  monthly          trade fully every 21 days
  when drifted     trade fully only when the trade needed exceeds a threshold (HFM's idea)
  part-way         trade a fixed fraction of the way every day (the optimal form in
                   Garleanu and Pedersen 2013 for quadratic costs)
  hold until weak  buy the top 10, sell only when a holding leaves the top N
Each rule's one setting is chosen on 1970-1999 and tested on 2000 onwards.

Usage: python portfolio_costs.py <49_Industry_Portfolios_Daily.csv> [cost in basis points]
"""
from __future__ import annotations

import sys

import numpy as np

TOP = 10


def load(path):
    dates, rows = [], []
    started = False
    for line in open(path, encoding="latin-1"):
        p = [x.strip() for x in line.split(",")]
        if len(p) > 40 and p[0].isdigit() and len(p[0]) == 8:
            started = True
            dates.append(int(p[0]))
            rows.append([float(x) for x in p[1:50]])
        elif started:
            break                      # end of the value-weighted block
    R = np.array(rows) / 100.0
    R[R < -0.99] = np.nan
    return np.array(dates), R


def signals(R):
    L = np.cumsum(np.log1p(np.nan_to_num(R)), axis=0)
    T = len(R)
    mom = np.full_like(R, np.nan)
    rev = np.full_like(R, np.nan)
    mom[254:] = L[232:T - 22] - L[1:T - 253]          # days t-253..t-22, known at close t-1
    rev[254:] = -(L[253:T - 1] - L[248:T - 6])        # minus the last 5 days' return
    ok = ~np.isnan(R)
    ok[1:] &= ok[:-1]

    def z(x):
        x = np.where(ok, x, np.nan)
        return (x - np.nanmean(x, axis=1, keepdims=True)) / (np.nanstd(x, axis=1, keepdims=True) + 1e-12)

    zm, zr = z(mom), z(rev)
    return {"slow": zm, "fast": zm + zr}


def run(R, S, rule, par, cost, a, b):
    """Daily net returns over days a..b-1. Returns (net returns, mean daily turnover)."""
    n = R.shape[1]
    w = np.zeros(n)
    held = np.zeros(n, bool)
    out, turn = np.zeros(b - a), 0.0
    for t in range(a, b):
        s = np.where(np.isnan(S[t]), -np.inf, S[t])
        rank = np.empty(n, int)
        rank[np.argsort(-s)] = np.arange(n)
        if rule == "hold":
            held &= rank < par
            need = TOP - held.sum()
            if need > 0:
                cand = np.argsort(-s)
                add = [i for i in cand if not held[i]][:need]
                held[add] = True
            target = held / held.sum()
        else:
            target = (rank < TOP) / TOP
        gap = np.abs(target - w).sum() / 2
        if w.sum() == 0:
            new = target
        elif rule == "daily":
            new = target
        elif rule == "monthly":
            new = target if (t - a) % 21 == 0 else w
        elif rule == "drift":
            new = target if gap > par else w
        elif rule == "part":
            new = w + par * (target - w)
        elif rule == "hold":
            new = target if gap > 1e-3 and (held != (w > 1e-9)).any() else w
        elif rule == "equal":
            new = np.full(n, 1.0 / n) if (t - a) % 21 == 0 else w
        traded = np.abs(new - w).sum()
        r = np.nan_to_num(R[t])
        gross = new @ r
        out[t - a] = gross - cost * traded
        turn += traded / 2
        w = new * (1 + r) / (1 + gross)
    return out, turn / (b - a)


def stats(x):
    ann = (1 + x).prod() ** (252 / len(x)) - 1
    return ann, x.mean() / x.std() * np.sqrt(252)


GRID = {"daily": [None], "monthly": [None], "drift": [0.05, 0.1, 0.2, 0.3, 0.4, 0.6], "part": [0.01, 0.02, 0.05, 0.1, 0.2, 0.4],
        "hold": [12, 15, 20, 25, 30], "equal": [None]}
NAMES = {"equal": "all industries, equal weight", "daily": "trade fully every day", "monthly": "trade fully every 21 days",
         "drift": "trade only when drifted (HFM's idea)", "part": "trade part-way every day", "hold": "hold until a holding weakens"}


def main():
    dates, R = load(sys.argv[1])
    cost = (float(sys.argv[2]) if len(sys.argv) > 2 else 10.0) / 1e4
    S = signals(R)
    a = int(np.searchsorted(dates, 19700101))
    m = int(np.searchsorted(dates, 20000101))
    b = len(dates)
    years = dates[m:b] // 10000
    print(f"{R.shape[1]} industries, {dates[0]}-{dates[-1]}; settings chosen on 1970-1999, tested {dates[m]}-{dates[b-1]}; cost {cost*1e4:.0f} basis points per unit traded")
    for sig in ("slow", "fast"):
        print(f"\nsignal: {sig}")
        print(f"  {'rule':<38} {'setting':>8} {'net return':>11} {'Sharpe':>7} {'before costs':>13} {'yearly turnover':>16} {'years better than monthly':>26}")
        res = {}
        for rule in ("equal", "daily", "monthly", "drift", "part", "hold"):
            best = max(GRID[rule], key=lambda p: stats(run(R, S[sig], rule, p, cost, a, m)[0])[1])
            net, turn = run(R, S[sig], rule, best, cost, m, b)
            gross, _ = run(R, S[sig], rule, best, 0.0, m, b)
            res[rule] = (best, net, turn, gross)
        ref = res["monthly"][1]
        yr = lambda x: np.array([(1 + x[years == y]).prod() - 1 for y in np.unique(years)])
        for rule, (best, net, turn, gross) in res.items():
            ann, sh = stats(net)
            wins = int((yr(net) > yr(ref) + 1e-12).sum())
            print(f"  {NAMES[rule]:<38} {str(best) if best is not None else '-':>8} {ann:>+11.2%} {sh:>7.2f} {stats(gross)[0]:>+13.2%} {turn*252:>15.1f}x {wins:>21}/{len(np.unique(years))}")


if __name__ == "__main__":
    main()
