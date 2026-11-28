"""Routing a small order between two exchanges, replayed on public trade prints.

Experiment for HFM's "Cross-Venue Allocation of Perpetual Orders" track (Binance,
Bybit, OKX). This uses the two venues with free public trade archives:
  Binance USD-M futures aggregate trades  https://data.binance.vision  (CC BY-NC-SA 4.0)
  Bybit derivatives trades                https://public.bybit.com/trading/
for BTCUSDT perpetuals on the same days. No order books are public for free, so the
price a small taker order would get is approximated by the next trade on that venue
in the same direction (a buyer-initiated trade for a buy), at least 200 ms after the
decision. Orders are assumed too small to move the price. Fees are taken as equal.

Every 30 seconds a parent order arrives, alternating buy and sell. The router picks
one venue using only trades before the decision time.

Policies
  always Binance / always Bybit
  last same-side price   pick the venue whose last same-direction trade was cheaper
  basis-adjusted         as above, after removing each venue's slow price offset
                         (an average of the price gap over the last few minutes),
                         so the choice reacts to short-lived gaps only
  hindsight              the venue that turned out cheaper (bound, not achievable)

Cost is in basis points against the average of the two venues' last prices at the
decision. The first day chooses the averaging window; other days are the test.
This is a study of routing mechanics, not investment advice, and it is NOT
comparable to HFM's reward.

Usage: python routing_replay.py <binance-day-1.csv> <bybit-day-1.csv> [more day pairs ...]
"""
import sys

import numpy as np
import pandas as pd

LATENCY, EVERY = 0.2, 30.0


def load_binance(path):
    d = pd.read_csv(path, header=None, low_memory=False)
    if not str(d.iloc[0, 0]).replace(".", "").isdigit():
        d = d.iloc[1:]
    t = d.iloc[:, 5].astype(float).values / 1000.0
    price = d.iloc[:, 1].astype(float).values
    maker = d.iloc[:, 6].astype(str).str.lower().isin(["true", "1"]).values      # buyer is maker -> seller-initiated
    return t, price, ~maker


def load_bybit(path):
    d = pd.read_csv(path)
    t = d["timestamp"].astype(float).values
    order = np.argsort(t, kind="stable")
    return t[order], d["price"].astype(float).values[order], (d["side"].astype(str).str.lower() == "buy").values[order]


class Venue:
    def __init__(self, t, price, buy):
        self.side = {True: (t[buy], price[buy]), False: (t[~buy], price[~buy])}
        self.t, self.price = t, price

    def last(self, when, buy=None):
        t, p = (self.t, self.price) if buy is None else self.side[buy]
        i = np.searchsorted(t, when, side="right") - 1
        return p[np.maximum(i, 0)]

    def next(self, when, buy):
        t, p = self.side[buy]
        i = np.minimum(np.searchsorted(t, when + LATENCY, side="left"), len(t) - 1)
        return p[i]


def day_costs(a, b, window):
    t0, t1 = max(a.t[0], b.t[0]) + 600, min(a.t[-1], b.t[-1]) - 60
    when = np.arange(t0, t1, EVERY)
    buy = (np.arange(len(when)) % 2 == 0)
    sign = np.where(buy, 1.0, -1.0)
    ref = 0.5 * (a.last(when) + b.last(when))
    ex_a = np.where(buy, a.next(when, True), a.next(when, False))
    ex_b = np.where(buy, b.next(when, True), b.next(when, False))
    cost_a, cost_b = sign * (ex_a - ref) / ref * 1e4, sign * (ex_b - ref) / ref * 1e4
    la = np.where(buy, a.last(when, True), a.last(when, False))
    lb = np.where(buy, b.last(when, True), b.last(when, False))
    # slow offset between venues: mean of (a - b) last prices over the preceding window, sampled every 5 s
    grid = np.arange(t0 - 600, t1, 5.0)
    gap = a.last(grid) - b.last(grid)
    k = max(int(window / 5), 1)
    csum = np.concatenate([[0.0], np.cumsum(gap)])
    idx = np.searchsorted(grid, when, side="right")
    basis = (csum[idx] - csum[np.maximum(idx - k, 0)]) / np.minimum(k, idx)
    pick_last = sign * (la - lb) <= 0                      # True -> venue a is cheaper for this side
    pick_adj = sign * (la - lb - basis) <= 0
    return dict(a=cost_a, b=cost_b, last=np.where(pick_last, cost_a, cost_b), adj=np.where(pick_adj, cost_a, cost_b),
                best=np.minimum(cost_a, cost_b), share_a_last=pick_last.mean(), share_a_adj=pick_adj.mean())


def main():
    pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))
    days = []
    for pa, pb in pairs:
        a, b = Venue(*load_binance(pa)), Venue(*load_bybit(pb))
        days.append((a, b))
        print(f"loaded {pa.split('/')[-1]}: {len(a.t)} Binance trades, {len(b.t)} Bybit trades", flush=True)
    best = min(((np.mean(day_costs(*days[0], w)["adj"]), w) for w in (30, 120, 600, 3600)))
    window = best[1]
    print(f"averaging window chosen on the first day: {window} s")
    rows = [day_costs(a, b, window) for a, b in days[1:]]
    n = sum(len(r["a"]) for r in rows)
    print(f"\ntest: {len(rows)} days, {n} orders; cost in basis points (lower is better)")
    print(f"  {'policy':<24} {'mean cost':>10} {'vs always Binance':>18} {'days better than Binance':>25}")
    base = np.concatenate([r["a"] for r in rows]).mean()
    for key, name in (("a", "always Binance"), ("b", "always Bybit"), ("last", "last same-side price"), ("adj", "basis-adjusted"), ("best", "hindsight (bound)")):
        c = np.concatenate([r[key] for r in rows]).mean()
        better = sum(r[key].mean() < r["a"].mean() - 1e-12 for r in rows)
        print(f"  {name:<24} {c:>10.3f} {c - base:>+18.3f} {better:>21}/{len(rows)}")
    print(f"  share routed to Binance: last same-side price {np.mean([r['share_a_last'] for r in rows]):.0%}, basis-adjusted {np.mean([r['share_a_adj'] for r in rows]):.0%}")


if __name__ == "__main__":
    main()
