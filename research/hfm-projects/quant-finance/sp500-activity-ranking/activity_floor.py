"""Ranking instruments by forward trading-activity surprise, on public crypto trades, on CPU.

Stand-in for HFM's "S&P 500 Minute Trading Activity Ranking", which needs licensed
one-minute consolidated-tape bars. The task: every minute, rank members by how much more
they will trade over the next thirty minutes than over the last thirty. Here the members
are 24 USDT perpetuals on Bybit, built from every public trade (public.bybit.com), and
the score is the Spearman rank correlation across members at each minute, averaged over
test minutes (one "instance" per minute).

Target for member i at minute t: log(volume in (t, t+30] / volume in (t-30, t]), volume in
quote currency.

Predictors, all from data up to minute t:
  reversion      minus log(last 30 / previous 30): bursts fade
  seasonal       log(typical volume in the next 30 minutes / in the last 30) at the same
                 time of day, from the member's earlier days (median over days)
  ridge / boosted, on: the two above, log(last 5 / last 30), log(last 30 / trailing day
                 mean), absolute return over the last 30 and 120 minutes, trade-count
                 ratio, hour; with and without signed price momentum over the last ten
                 minutes (HFM's current best idea is to add this covariate)
Fitted on the first days, scored on the last days.

Usage: python activity_floor.py <minute-bars.npz> <test days>
       python activity_floor.py build <data dir> <out.npz> <day> [<day> ...] -- <SYMBOL> [...]
"""
import gzip
import os
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.stats import rankdata
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV

W = 30


def build(folder, out, days, syms):
    import pandas as pd
    n = len(days) * 1440
    vol = np.zeros((len(syms), n))
    cnt = np.zeros((len(syms), n))
    close = np.full((len(syms), n), np.nan)
    t0 = int(datetime.strptime(days[0], "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())
    for i, s in enumerate(syms):
        for d in days:
            p = os.path.join(folder, f"{s}{d}.csv.gz")
            if not os.path.exists(p):
                print(f"missing {p}", flush=True)
                continue
            df = pd.read_csv(p, usecols=["timestamp", "size", "price"])
            m = ((df["timestamp"].values - t0) // 60).astype(int)
            ok = (m >= 0) & (m < n)
            m, sz, px = m[ok], df["size"].values[ok], df["price"].values[ok]
            np.add.at(vol[i], m, sz * px)
            np.add.at(cnt[i], m, 1)
            last = pd.Series(px).groupby(m).last()
            close[i, last.index.values] = last.values
        print(f"  {s}: total volume ${vol[i].sum() / 1e9:.2f}bn, trades {cnt[i].sum():.0f}", flush=True)
    for i in range(len(syms)):                                      # carry prices forward over empty minutes
        c = close[i]
        idx = np.where(~np.isnan(c), np.arange(n), 0)
        np.maximum.accumulate(idx, out=idx)
        close[i] = c[idx]
    np.savez_compressed(out, vol=vol, cnt=cnt, close=close, syms=np.array(syms), days=np.array(days))


def csum(a):
    return np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(a, 1)], 1)


def window(cs, t, w):
    return cs[:, t + 1] - cs[:, t + 1 - w]


def main():
    if sys.argv[1] == "build":
        cut = sys.argv.index("--")
        return build(sys.argv[2], sys.argv[3], sys.argv[4:cut], sys.argv[cut + 1:])
    z = np.load(sys.argv[1])
    vol, cnt, close, syms = z["vol"], z["cnt"], z["close"], z["syms"]
    test_days = int(sys.argv[2])
    n = vol.shape[1]
    ndays = n // 1440
    eps = 1e3                                                        # $1,000 floor on volume windows
    cv, cc = csum(vol), csum(cnt)
    lr = np.log(close)
    rows = []                                                        # per minute: (t, X (members x features), y)
    for t in range(1440 * 3, n - W, 5):                              # every fifth minute, after three days of history
        day, mod = divmod(t, 1440)
        last = window(cv, t, W) + eps
        prev = window(cv, t - W, W) + eps
        nxt = cv[:, t + 1 + W] - cv[:, t + 1] + eps
        y = np.log(nxt / last)
        # seasonal: median over earlier days of log(next30/last30) at the same minute of day
        past = [t - 1440 * k for k in range(1, day + 1) if t - 1440 * k - W >= 0]
        seas = np.median([np.log((cv[:, p + 1 + W] - cv[:, p + 1] + eps) / (window(cv, p, W) + eps)) for p in past], 0)
        trail = (cv[:, t + 1] - cv[:, max(0, t + 1 - 1440)]) / min(1440, t + 1) * W + eps
        last5 = window(cv, t, 5) + eps
        X = np.column_stack([
            -np.log(last / prev), seas, np.log(last5 * W / 5 / last), np.log(last / trail),
            np.abs(lr[:, t] - lr[:, t - W]), np.abs(lr[:, t] - lr[:, t - 120]),
            np.log((window(cc, t, W) + 1) / (window(cc, t - W, W) + 1)), np.full(len(syms), mod // 60),
            lr[:, t] - lr[:, t - 10],                                # signed ten-minute momentum (last column)
        ])
        rows.append((t, day, X, y))
    fit = [r for r in rows if r[1] < ndays - test_days]
    test = [r for r in rows if r[1] >= ndays - test_days]
    print(f"members {len(syms)}; fitting instances {len(fit)}, test instances {len(test)} (every fifth minute)")

    def score(pred_fn):
        s = [np.corrcoef(rankdata(pred_fn(X)), rankdata(y))[0, 1] for _, _, X, y in test]
        s = np.array(s)
        days = np.array([d for _, d, _, _ in test])
        per_day = [s[days == d].mean() for d in np.unique(days)]
        return s.mean(), np.std(per_day) / np.sqrt(len(per_day)), per_day

    def demean(rs):
        """Cross-sectional ranking only needs relative values: centre features and target per instance."""
        X = np.concatenate([r[2] - r[2].mean(0) for r in rs])
        y = np.concatenate([r[3] - r[3].mean() for r in rs])
        return X, y

    res = {"reversion": score(lambda X: X[:, 0]), "seasonal": score(lambda X: X[:, 1]),
           "reversion + seasonal": score(lambda X: X[:, 0] + X[:, 1])}
    Xf, yf = demean(fit)
    for mom in (False, True):
        cols = slice(None) if mom else slice(0, -1)
        tag = " + momentum" if mom else ""
        ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xf[:, cols], yf)
        res["ridge" + tag] = score(lambda X: ridge.predict(X[:, cols]))
        gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, random_state=0).fit(Xf[:, cols], yf)
        res["boosted" + tag] = score(lambda X: gb.predict(X[:, cols]))
    for name, (m, se, per) in res.items():
        print(f"RESULT {name:24s} mean instance Spearman {m:+.4f} (se over days {se:.4f}); by test day " + " ".join(f"{v:+.3f}" for v in per), flush=True)
    for base in ("ridge", "boosted"):
        a = np.array(res[base + " + momentum"][2]) - np.array(res[base][2])
        print(f"RESULT momentum added to {base}: {a.mean():+.4f} per test day (range {a.min():+.4f} to {a.max():+.4f})")


if __name__ == "__main__":
    main()
