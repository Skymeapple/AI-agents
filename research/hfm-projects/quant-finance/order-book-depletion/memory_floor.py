"""Mapping C test: does the order book's path predict depth depletion beyond its current state?

The framework compares order-book depletion with steel hysteresis and convective
instability: path-dependent systems near tipping points, where the history of loading
(not only the present state) sets the next response. If that holds, features that
summarise the path should add to the state features of bybit_book_floor.py.

Same data, buckets, target and fit/test split as bybit_book_floor.py part 1 (ten-second
buckets, log change of ten-level depth on each side over the next bucket; earlier days
fit, last day test). Feature sets:
  state     the eight features of bybit_book_floor.py (last change, traded volume against
            the side, depth relative to the other side, three-level imbalance, share of
            depth in the top three levels, spread, hour, side)
  + memory  also: depth changes over the previous 2-6 buckets (one minute of lags);
            drawdown of depth from its running peak over 5 and 30 minutes (the analogue
            of accumulated plastic strain); depth relative to its 30-minute minimum
            (distance from the recent "exhausted" state); traded volume against the side
            over 1 and 5 minutes relative to depth; mid-price volatility over 1 and 5
            minutes; the other side's last change
Models: ridge and gradient-boosted trees for each set. Score: Spearman on the test day and
top-decile hit rate, as in bybit_book_floor.py.

Usage: python memory_floor.py <data dir> <SYMBOL> <day> [<day> ...]
"""
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bybit_book_floor as bf

B = bf.BUCKET


def rows(D):
    n = 86400
    vol = {+1: np.bincount(D.tr[:, 0].astype(int).clip(0, n - 1), weights=D.tr[:, 2] * (D.tr[:, 1] > 0), minlength=n),
           -1: np.bincount(D.tr[:, 0].astype(int).clip(0, n - 1), weights=D.tr[:, 2] * (D.tr[:, 1] < 0), minlength=n)}
    mid = (D.bp[:, 0] + D.ap[:, 0]) / 2
    lmid = np.log(mid)
    out = []
    start = 1800 + B
    for s in range(start, n - B, B):
        idx = [s - k * B for k in range(0, 7)] + [s + B]
        if not all(D.ok[i] for i in idx):
            continue
        spread = (D.ap[s, 0] - D.bp[s, 0]) / mid[s] * 1e4
        for side, sz, other, hit in (("ask", D.as_, D.bs, vol[+1]), ("bid", D.bs, D.as_, vol[-1])):
            d = {k: sz[s - k * B].sum() for k in range(7)}
            d2 = sz[s + B].sum()
            if min(min(d.values()), d2) <= 0:
                continue
            o0, o1 = other[s].sum(), other[s - B].sum()
            imb3 = (sz[s, :3].sum() - other[s, :3].sum()) / (sz[s, :3].sum() + other[s, :3].sum())
            state = [np.log(d[0] / d[1]), hit[s - B:s].sum() / d[0], np.log(d[0] / o0), imb3, sz[s, :3].sum() / d[0], spread, s // 3600, side == "ask"]
            win5 = sz[s - 300:s + 1:B].sum(1)
            win30 = sz[s - 1800:s + 1:B].sum(1)
            win5, win30 = win5[win5 > 0], win30[win30 > 0]
            if len(win5) < 5 or len(win30) < 20:
                continue
            lm5 = lmid[s - 300:s + 1]
            lm1 = lmid[s - 60:s + 1]
            memory = [np.log(d[k - 1] / d[k]) for k in range(2, 7)] + [
                np.log(d[0] / win5.max()), np.log(d[0] / win30.max()), np.log(d[0] / win30.min()),
                hit[s - 60:s].sum() / d[0], hit[s - 300:s].sum() / d[0],
                np.nanstd(np.diff(lm1)) * 1e4, np.nanstd(np.diff(lm5)) * 1e4,
                np.log(o0 / o1) if o1 > 0 else 0.0]
            out.append((np.log(d2 / d[0]), state, state + memory))
    return out


def main():
    folder, sym, days = sys.argv[1], sys.argv[2], sys.argv[3:]
    data = {day: rows(bf.Day(folder, sym, day)) for day in days}
    fit = [r for day in days[:-1] for r in data[day]]
    test = data[days[-1]]
    yf, yt = np.array([r[0] for r in fit]), np.array([r[0] for r in test])
    big = -yt >= np.quantile(-yt, 0.9)
    print(f"{sym}: {len(fit)} fitting and {len(test)} test bucket-sides", flush=True)
    for name, col in (("state", 1), ("+ memory", 2)):
        Xf = np.nan_to_num(np.array([r[col] for r in fit], float))
        Xt = np.nan_to_num(np.array([r[col] for r in test], float))
        for model, m in (("ridge", RidgeCV(alphas=np.logspace(-3, 3, 13))),
                         ("boosted", HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0))):
            p = m.fit(Xf, yf).predict(Xt)
            top = p <= np.quantile(p, 0.1)
            print(f"RESULT {sym} {name:9s} {model:8s} Spearman {spearmanr(p, yt).correlation:+.3f}; top-decile hit {np.mean(big[top]):.3f}", flush=True)


if __name__ == "__main__":
    main()
