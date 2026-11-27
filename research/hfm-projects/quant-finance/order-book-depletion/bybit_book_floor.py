"""Order-book depth forecasting and execution replay on public crypto order books, on CPU.

Stand-in for two HFM tracks that need licensed Nasdaq TotalView data (Databento):
  - "Nasdaq Ten-Level Order Book Depth Exhaustion Ranking": rank ten-second buckets by
    how much of the ten-level depth will be used up next.
  - "Execution Policy Development from Ten-Level Order Book Data": lower the cost of
    working parent orders against the replayed book.
LOBSTER's free equity samples now require an application, so this uses Bybit's public
archives instead: full 500-level order-book snapshots and deltas for USDT perpetuals
(quote-saver.bycsi.com) and every public trade (public.bybit.com). Crypto perpetuals
trade around the clock with no auction, so this is an analogue, not a replica.

The book is rebuilt from the delta stream and sampled once a second (ten levels per side).
Days before the last are used for fitting; the last day is the test.

Part 1, depth. At the end of every ten-second bucket, for each side: total size in the ten
best levels and its log change over the next bucket (negative = depletion). Methods:
reversion (minus the last change), ridge and gradient-boosted trees on recent depth
change, traded volume against the side, imbalance, spread and hour. Score: Spearman rank
correlation on the test day, and the share of the 10% most-depleting buckets among each
method's 10% picks.

Part 2, execution. Parent orders of a fixed notional, buy and sell, over five minutes,
started every five minutes on the test day. Cost is implementation shortfall against the
arrival mid, in basis points. Policies: sweep (all at once), twap (ten market slices),
passive (each slice rests at the near touch for 30 seconds, then crosses for what is
left; a resting order fills only from later trades at or through its price beyond the
size queued ahead), adaptive (as passive, but crosses at once when the three-level
imbalance predicts the price moving away; threshold fitted on the earlier days).

Usage: python bybit_book_floor.py <data dir> <SYMBOL> <day> [<day> ...] -- <notional> [<notional> ...]
       files: <data dir>/<day>_<SYMBOL>_ob500.data.zip and <data dir>/<SYMBOL><day>.csv.gz
"""
import gzip
import io
import json
import os
import sys
import zipfile
from datetime import datetime, timezone

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

LEVELS = 10
BUCKET = 10


def day_start(day):
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def book_grid(path, t0):
    """Ten-level book once a second for one UTC day: arrays (86400, 10) of bid/ask price and size."""
    bp, bs, ap, as_ = (np.full((86400, LEVELS), np.nan) for _ in range(4))
    bids, asks = {}, {}
    nxt = 0
    with zipfile.ZipFile(path) as z, z.open(z.namelist()[0]) as f:
        for line in io.TextIOWrapper(f, encoding="utf-8"):
            msg = json.loads(line)
            sec = msg["ts"] // 1000 - t0
            while nxt < min(sec, 86400):
                if bids and asks:
                    b = sorted(bids.items(), reverse=True)[:LEVELS]
                    a = sorted(asks.items())[:LEVELS]
                    if len(b) == LEVELS and len(a) == LEVELS:
                        bp[nxt], bs[nxt] = zip(*b)
                        ap[nxt], as_[nxt] = zip(*a)
                nxt += 1
            d = msg["data"]
            if msg["type"] == "snapshot":
                bids = {float(p): float(s) for p, s in d["b"]}
                asks = {float(p): float(s) for p, s in d["a"]}
                continue
            for side, book in (("b", bids), ("a", asks)):
                for p, s in d[side]:
                    p, s = float(p), float(s)
                    if s == 0:
                        book.pop(p, None)
                    else:
                        book[p] = s
    return bp, bs, ap, as_


def trades(path, t0):
    """Per trade: second of day, +1 for taker buy (hits asks) / -1 for taker sell, size, price."""
    out = []
    with gzip.open(path, "rt") as f:
        head = f.readline().strip().split(",")
        it, iside, isz, ip = head.index("timestamp"), head.index("side"), head.index("size"), head.index("price")
        for line in f:
            r = line.split(",")
            out.append((float(r[it]) - t0, 1 if r[iside] == "Buy" else -1, float(r[isz]), float(r[ip])))
    a = np.array(out)
    return a[np.argsort(a[:, 0])]


class Day:
    def __init__(self, folder, sym, day):
        t0 = day_start(day)
        self.bp, self.bs, self.ap, self.as_ = book_grid(os.path.join(folder, f"{day}_{sym}_ob500.data.zip"), t0)
        self.tr = trades(os.path.join(folder, f"{sym}{day}.csv.gz"), t0)
        self.ok = ~np.isnan(self.bp[:, 0])
        self.ok[self.ok] &= self.bp[self.ok, 0] < self.ap[self.ok, 0]          # drop crossed books

    def mid(self, s):
        return (self.bp[s, 0] + self.ap[s, 0]) / 2

    def depth_rows(self):
        rows = []
        vol_buy = np.bincount(self.tr[:, 0].astype(int).clip(0, 86399), weights=self.tr[:, 2] * (self.tr[:, 1] > 0), minlength=86400)
        vol_sell = np.bincount(self.tr[:, 0].astype(int).clip(0, 86399), weights=self.tr[:, 2] * (self.tr[:, 1] < 0), minlength=86400)
        for s in range(2 * BUCKET, 86400 - BUCKET, BUCKET):
            s0, s1, s2 = s - BUCKET, s, s + BUCKET
            if not (self.ok[s0] and self.ok[s1] and self.ok[s2]):
                continue
            spread = (self.ap[s1, 0] - self.bp[s1, 0]) / self.mid(s1) * 1e4
            for side, sz, other, hit in (("ask", self.as_, self.bs, vol_buy), ("bid", self.bs, self.as_, vol_sell)):
                d0, d1, d2 = sz[s0].sum(), sz[s1].sum(), sz[s2].sum()
                if min(d0, d1, d2) <= 0:
                    continue
                traded = hit[s0:s1].sum()
                imb3 = (sz[s1, :3].sum() - other[s1, :3].sum()) / (sz[s1, :3].sum() + other[s1, :3].sum())
                rows.append((s, np.log(d2 / d1), [np.log(d1 / d0), traded / d1, np.log(d1 / other[s1].sum()), imb3,
                                                  sz[s1, :3].sum() / d1, spread, (s // 3600), side == "ask"]))
        return rows

    def sweep(self, s, qty, buy):
        p, sz = (self.ap[s], self.as_[s]) if buy else (self.bp[s], self.bs[s])
        left, cost = qty, 0.0
        for lv in range(LEVELS):
            take = min(left, sz[lv])
            cost += take * p[lv]
            left -= take
            if left <= 0:
                return cost / qty
        worst = p[-1] * (1.0005 if buy else 0.9995)          # beyond ten levels: 5 bp past the tenth
        return (cost + left * worst) / qty

    def rest(self, s, qty, buy, s_end):
        price = self.bp[s, 0] if buy else self.ap[s, 0]
        ahead = self.bs[s, 0] if buy else self.as_[s, 0]
        lo, hi = np.searchsorted(self.tr[:, 0], [s + 1, s_end])
        seg = self.tr[lo:hi]
        hit = seg[seg[:, 1] == (-1 if buy else 1)]            # taker sells hit resting bids
        at_or_through = hit[(hit[:, 3] <= price) if buy else (hit[:, 3] >= price)]
        through = at_or_through[(at_or_through[:, 3] < price) if buy else (at_or_through[:, 3] > price)]
        if len(through):
            return qty, price
        return min(qty, max(0.0, at_or_through[:, 2].sum() - ahead)), price

    def imbalance(self, s):
        b, a = self.bs[s, :3].sum(), self.as_[s, :3].sum()
        return (b - a) / (b + a)

    def run(self, s0, notional, buy, policy, thr=None, slices=10, horizon=300):
        arrival = self.mid(s0)
        qty = notional / arrival
        sgn = 1 if buy else -1
        if policy == "sweep":
            return sgn * (self.sweep(s0, qty, buy) - arrival) / arrival * 1e4
        child, paid = qty / slices, 0.0
        step = horizon // slices
        for k in range(slices):
            s = s0 + k * step
            if policy == "twap" or (policy == "adaptive" and sgn * self.imbalance(s) > thr):
                paid += child * self.sweep(s, child, buy)
                continue
            filled, price = self.rest(s, child, buy, s + step)
            paid += filled * price + (child - filled) * self.sweep(s + step, child - filled, buy) if child - filled > 0 else filled * price
        return sgn * (paid / qty - arrival) / arrival * 1e4


def main():
    args = sys.argv[1:]
    cut = args.index("--")
    folder, sym, days = args[0], args[1], args[2:cut]
    notionals = [float(x) for x in args[cut + 1:]]
    D = {}
    for day in days:
        D[day] = Day(folder, sym, day)
        print(f"{sym} {day}: seconds with a full book {D[day].ok.sum()}, trades {len(D[day].tr)}, "
              f"median spread {np.nanmedian((D[day].ap[:, 0] - D[day].bp[:, 0]) / D[day].mid(slice(None)) * 1e4):.2f} bp, "
              f"median ten-level depth ${np.nanmedian((D[day].bs.sum(1) + D[day].as_.sum(1)) / 2 * D[day].mid(slice(None))):,.0f} per side", flush=True)
    fit, test = days[:-1], days[-1]
    # part 1
    rows_fit = [r for d in fit for r in D[d].depth_rows()]
    rows_test = D[test].depth_rows()
    Xf, yf = np.array([r[2] for r in rows_fit], float), np.array([r[1] for r in rows_fit])
    Xt, yt = np.array([r[2] for r in rows_test], float), np.array([r[1] for r in rows_test])
    preds = {"reversion": -Xt[:, 0],
             "ridge": RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xf, yf).predict(Xt),
             "boosted": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(Xf, yf).predict(Xt)}
    big = -yt >= np.quantile(-yt, 0.9)
    print(f"\n===== {sym} part 1: {len(yf)} fitting and {len(yt)} test bucket-sides; depth falls in {np.mean(yt < 0):.3f} of test buckets =====")
    for name, p in preds.items():
        top = p <= np.quantile(p, 0.1)
        print(f"RESULT {sym} depth {name:10s} Spearman {spearmanr(p, yt).correlation:+.3f}; top-decile hit {np.mean(big[top]):.3f} (random 0.100)", flush=True)
    # part 2
    for notional in notionals:
        starts_fit = [(d, s) for d in fit for s in range(600, 86400 - 600, 300) if D[d].ok[s:s + 301].all()]
        starts_test = [s for s in range(600, 86400 - 600, 300) if D[test].ok[s:s + 301].all()]
        best = None
        for thr in (0.0, 0.2, 0.4, 0.6, 0.8, 1.01):
            c = np.mean([D[d].run(s, notional, b, "adaptive", thr) for d, s in starts_fit[::2] for b in (True, False)])
            if best is None or c < best[0]:
                best = (c, thr)
        res = {p: np.array([D[test].run(s, notional, b, p, best[1]) for s in starts_test for b in (True, False)])
               for p in ("sweep", "twap", "passive", "adaptive")}
        print(f"RESULT {sym} exec ${notional:,.0f}: {len(res['sweep'])} parents, adaptive threshold {best[1]}; cost bp: "
              + "  ".join(f"{p} {v.mean():+.2f}±{v.std() / np.sqrt(len(v)):.2f}" for p, v in res.items()), flush=True)
        for p in ("passive", "adaptive"):
            d = res["twap"] - res[p]
            print(f"RESULT {sym} exec ${notional:,.0f} saving of {p} over twap {d.mean():+.2f} bp (se {d.std() / np.sqrt(len(d)):.2f}), better in {np.mean(d > 0):.2f}", flush=True)


if __name__ == "__main__":
    main()
