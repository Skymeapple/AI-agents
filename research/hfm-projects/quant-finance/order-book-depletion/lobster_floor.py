"""Order-book depth forecasting and execution replay on LOBSTER's free sample day, on CPU.

Stand-in for two HFM tracks that need licensed Nasdaq TotalView data:
  - "Nasdaq Ten-Level Order Book Depth Exhaustion Ranking": rank ten-second buckets by
    how much of the book's depth will be used up next.
  - "Execution Policy Development from Ten-Level Order Book Data": lower the cost of
    working parent orders against the replayed book.
LOBSTER (lobsterdata.com) publishes one free sample day (2012-06-21) of ten-level
Nasdaq books with full messages for AMZN, AAPL, GOOG, INTC and MSFT. One day for five
stocks is a small sample; every result below is split by time within that day.

Part 1, depth. For each stock and side, at the end of every ten-second bucket: total
displayed size in the ten levels, and what happens to it over the next bucket. Target:
log ratio of the next bucket's end depth to this one's (negative = depletion). Methods,
fitted on the morning and scored on the afternoon, by Spearman rank correlation pooled
over stocks and sides:
  zero          predict no change (rank ties)
  reversion     predict minus the last bucket's change
  ridge         linear model on recent depth changes, executed volume, cancellations,
                imbalance, spread and time of day
  boosted       gradient-boosted trees on the same features

Part 2, execution. Parent orders to buy or sell a fixed number of shares within five
minutes, started every five minutes in the afternoon. Cost is implementation shortfall
against the mid-price at arrival, in basis points (positive = cost). Policies:
  sweep         everything at once, walking the book
  twap          ten equal market slices, one every thirty seconds
  passive       each slice rests at the near touch for its thirty seconds; what has not
                filled is then sent as a market order. A resting order fills only when
                later executions at its price exceed the size queued ahead of it.
  adaptive      as passive, but a slice crosses at once when the book imbalance
                predicts the price moving away (threshold fitted on the morning)

Usage: python lobster_floor.py <dir with the unzipped sample files> [parent size in shares]
"""
import glob
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

BUCKET = 10.0
OPEN, CLOSE = 34200 + 900, 57600 - 900            # skip the first and last 15 minutes
NOON = 43200


def load(folder, sym):
    msg = glob.glob(os.path.join(folder, f"{sym}_*_message_10.csv"))[0]
    book = glob.glob(os.path.join(folder, f"{sym}_*_orderbook_10.csv"))[0]
    m = np.loadtxt(msg, delimiter=",", usecols=range(6))
    b = np.loadtxt(book, delimiter=",")
    return m, b


def depth_rows(sym, m, b):
    t = m[:, 0]
    ask_sz, bid_sz = b[:, 1::4], b[:, 3::4]
    ask_p, bid_p = b[:, 0::4] / 1e4, b[:, 2::4] / 1e4
    edges = np.arange(OPEN, CLOSE + BUCKET, BUCKET)
    idx = np.searchsorted(t, edges, side="right") - 1          # last event at or before each edge
    rows = []
    for k in range(2, len(edges) - 1):
        i0, i1, i2 = idx[k - 1], idx[k], idx[k + 1]
        lo, hi = i0 + 1, i1 + 1
        typ, size, direction = m[lo:hi, 1], m[lo:hi, 3], m[lo:hi, 5]
        mid = (ask_p[i1, 0] + bid_p[i1, 0]) / 2
        spread = (ask_p[i1, 0] - bid_p[i1, 0]) / mid * 1e4
        for side, sz, sgn in (("ask", ask_sz, -1), ("bid", bid_sz, 1)):
            d0, d1, d2 = sz[i0].sum(), sz[i1].sum(), sz[i2].sum()
            if min(d0, d1, d2) <= 0:
                continue
            on_side = direction == sgn
            execd = size[on_side & np.isin(typ, (4, 5))].sum()
            canc = size[on_side & np.isin(typ, (2, 3))].sum()
            added = size[on_side & (typ == 1)].sum()
            other = (bid_sz if side == "ask" else ask_sz)[i1].sum()
            near = sz[i1, :3].sum() / d1
            rows.append(dict(sym=sym, side=side, t=edges[k], y=np.log(d2 / d1),
                             x=[np.log(d1 / d0), execd / d1, canc / d1, added / d1, np.log(d1 / other), near, spread,
                                (edges[k] - OPEN) / (CLOSE - OPEN), side == "ask"]))
    return rows


def part1(data):
    rows = [r for sym, (m, b) in data.items() for r in depth_rows(sym, m, b)]
    X = np.array([r["x"] for r in rows], float)
    y = np.array([r["y"] for r in rows])
    am = np.array([r["t"] < NOON for r in rows])
    print(f"\n===== Part 1: depth change over the next ten seconds; {am.sum()} morning and {(~am).sum()} afternoon bucket-sides =====")
    print(f"  share of afternoon bucket-sides where depth falls: {np.mean(y[~am] < 0):.3f}; mean log change {y[~am].mean():+.4f}")
    preds = {"reversion": -X[~am, 0]}
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(X[am], y[am])
    preds["ridge"] = ridge.predict(X[~am])
    gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(X[am], y[am])
    preds["boosted"] = gb.predict(X[~am])
    syms = np.array([r["sym"] for r in rows])[~am]
    for name, p in preds.items():
        pooled = spearmanr(p, y[~am]).correlation
        per = [spearmanr(p[syms == s], y[~am][syms == s]).correlation for s in data]
        big = np.quantile(-y[~am], 0.9)
        top = p <= np.quantile(p, 0.1)
        hit = np.mean(-y[~am][top] >= big)
        print(f"RESULT depth {name:10s} Spearman pooled {pooled:+.3f}; per stock " + " ".join(f"{s} {v:+.3f}" for s, v in zip(data, per))
              + f"; share of the 10% most-depleting buckets among its 10% picks {hit:.3f} (random 0.100)")
    names = ["last change", "executed/depth", "cancelled/depth", "added/depth", "log depth ratio to other side", "near-3 share", "spread bp", "time of day", "is ask"]
    print("  ridge coefficients (standardise by eye): " + ", ".join(f"{n} {c:+.3f}" for n, c in zip(names, ridge.coef_)))


class Replay:
    def __init__(self, m, b):
        self.t, self.m = m[:, 0], m
        self.ask_p, self.ask_s = b[:, 0::4] / 1e4, b[:, 1::4]
        self.bid_p, self.bid_s = b[:, 2::4] / 1e4, b[:, 3::4]

    def at(self, t):
        return max(0, np.searchsorted(self.t, t, side="right") - 1)

    def mid(self, i):
        return (self.ask_p[i, 0] + self.bid_p[i, 0]) / 2

    def sweep(self, i, qty, buy):
        """Average price of walking the book for qty; beyond ten levels, pay the tenth level plus one tick per level."""
        p, s = (self.ask_p[i], self.ask_s[i]) if buy else (self.bid_p[i], self.bid_s[i])
        left, cost = qty, 0.0
        for lv in range(10):
            take = min(left, s[lv])
            cost += take * p[lv]
            left -= take
            if left <= 0:
                return cost / qty
        cost += left * (p[9] + (0.01 if buy else -0.01) * 5)
        return cost / qty

    def rest(self, i, qty, buy, t_end):
        """Shares filled by resting qty at the near touch from event i until t_end (back of the queue)."""
        price = self.bid_p[i, 0] if buy else self.ask_p[i, 0]
        ahead = self.bid_s[i, 0] if buy else self.ask_s[i, 0]
        j = self.at(t_end)
        seg = self.m[i + 1:j + 1]
        side = 1 if buy else -1                          # LOBSTER direction of the resting order that was hit
        hits = seg[(np.isin(seg[:, 1], (4, 5))) & (seg[:, 5] == side)]
        through = hits[(hits[:, 4] / 1e4 <= price) if buy else (hits[:, 4] / 1e4 >= price)]
        traded = through[:, 3].sum()
        # executions strictly through our price fill us fully; at our price, only past the queue ahead
        strictly = through[(through[:, 4] / 1e4 < price) if buy else (through[:, 4] / 1e4 > price)][:, 3].sum()
        filled = qty if strictly > 0 else min(qty, max(0.0, traded - ahead))
        return filled, price, j

    def imbalance(self, i, levels=3):
        b, a = self.bid_s[i, :levels].sum(), self.ask_s[i, :levels].sum()
        return (b - a) / (b + a)


def run_policy(r, t0, qty, buy, policy, thr=None, slices=10, horizon=300.0):
    i0 = r.at(t0)
    arrival = r.mid(i0)
    sgn = 1 if buy else -1
    if policy == "sweep":
        px = r.sweep(i0, qty, buy)
        return sgn * (px - arrival) / arrival * 1e4
    child = qty / slices
    paid, done = 0.0, 0.0
    for k in range(slices):
        ts = t0 + k * horizon / slices
        i = r.at(ts)
        if policy == "twap":
            paid += child * r.sweep(i, child, buy)
            done += child
            continue
        imb = r.imbalance(i)
        if policy == "adaptive" and sgn * imb > thr:     # price likely to move away: cross now
            paid += child * r.sweep(i, child, buy)
            done += child
            continue
        filled, price, j = r.rest(i, child, buy, ts + horizon / slices)
        paid += filled * price
        if child - filled > 0:
            paid += (child - filled) * r.sweep(j, child - filled, buy)
        done += child
    return sgn * (paid / done - arrival) / arrival * 1e4


def part2(data, qty):
    print(f"\n===== Part 2: execution of {qty}-share parents over five minutes, buys and sells, started every five minutes =====")
    allres = {}
    for sym, (m, b) in data.items():
        r = Replay(m, b)
        # fit the adaptive threshold on morning parents
        starts_am = np.arange(OPEN, NOON - 300, 300)
        starts_pm = np.arange(NOON, CLOSE - 300, 300)
        best = None
        for thr in (0.0, 0.2, 0.4, 0.6, 0.8, 1.01):
            c = np.mean([run_policy(r, t, qty, buy, "adaptive", thr) for t in starts_am for buy in (True, False)])
            if best is None or c < best[0]:
                best = (c, thr)
        res = {}
        for pol in ("sweep", "twap", "passive", "adaptive"):
            res[pol] = [run_policy(r, t, qty, buy, pol, best[1]) for t in starts_pm for buy in (True, False)]
        half_spread = np.mean([(r.ask_p[r.at(t), 0] - r.bid_p[r.at(t), 0]) / r.mid(r.at(t)) * 1e4 / 2 for t in starts_pm])
        print(f"RESULT exec {sym}: half-spread {half_spread:.2f} bp; adaptive threshold {best[1]}; cost bp (mean ± se over {len(res['sweep'])}): "
              + "  ".join(f"{p} {np.mean(v):+.2f}±{np.std(v) / np.sqrt(len(v)):.2f}" for p, v in res.items()), flush=True)
        for p, v in res.items():
            allres.setdefault(p, []).extend(v)
    print("RESULT exec all stocks: " + "  ".join(f"{p} {np.mean(v):+.2f}±{np.std(v) / np.sqrt(len(v)):.2f}" for p, v in allres.items()))
    tw = np.array(allres["twap"])
    for p in ("passive", "adaptive"):
        d = tw - np.array(allres[p])
        print(f"RESULT exec saving of {p} over twap: {d.mean():+.2f} bp (se {d.std() / np.sqrt(len(d)):.2f}), better in {np.mean(d > 0):.2f} of parents")


def main():
    folder = sys.argv[1]
    qty = float(sys.argv[2]) if len(sys.argv) > 2 else 2000
    syms = sorted({os.path.basename(f).split("_")[0] for f in glob.glob(os.path.join(folder, "*_message_10.csv"))})
    data = {s: load(folder, s) for s in syms}
    for s, (m, b) in data.items():
        print(f"{s}: {len(m)} messages, {np.isin(m[:, 1], (4, 5)).sum()} executions, mean ten-level depth ask {b[:, 1::4].sum(1).mean():.0f} bid {b[:, 3::4].sum(1).mean():.0f}")
    part1(data)
    part2(data, qty)


if __name__ == "__main__":
    main()
