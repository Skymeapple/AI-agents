"""Intraday overlay on a fixed basket, net of costs, on public crypto minute data, on CPU.

Stand-in for HFM's "Intraday Trading Overlay for A-Share Holdings", which needs
five-minute A-share data that are not freely redistributable (and whose reward is pinned
at 1.0). The task shape is kept: a fixed base book, intraday tilts around it, costs on
every change, and the tilt closed by the end of each day. HFM's current best idea is to
shrink the forecast's payoff slopes ten times harder (stronger ridge penalty).

Base book: equal weights in 24 USDT perpetuals (Bybit public trades, aggregated to minute
closes). Every five minutes the overlay sets a tilt w_i = clip(k * forecast_i, +-cap)
around the base, demeaned so it is self-financing, with cap = the base weight (so the
book stays long-only). The forecast is a ridge regression of the next interval's
cross-sectionally demeaned return on: the last 5-, 30- and 120-minute returns and the
30-minute volume surprise, all cross-sectionally demeaned. Costs: c basis points per unit
of turnover. A no-trade band skips a member's change when it is under b x cap.

Ridge penalties: the one chosen by cross-validation on the fitting days, then 10x and
100x it (HFM's idea), and an equal-weight sign rule (pure short-term reversal) for
comparison. k and b are chosen on the fitting days for each penalty and cost.

Usage: python overlay_floor.py <bars.npz from activity_floor.py build> <test days>
"""
import sys

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV

STEP = 5


def features(close, vol, t):
    lr = np.log(close)
    def dm(x):
        return x - x.mean()
    v30 = vol[:, t - 29:t + 1].sum(1) + 1e3
    v60 = vol[:, t - 59:t - 29].sum(1) + 1e3
    return np.column_stack([dm(lr[:, t] - lr[:, t - 5]), dm(lr[:, t] - lr[:, t - 30]), dm(lr[:, t] - lr[:, t - 120]), dm(np.log(v30 / v60))])


def main():
    z = np.load(sys.argv[1])
    close, vol = z["close"], z["vol"]
    test_days = int(sys.argv[2])
    n_sym, n = close.shape
    ndays = n // 1440
    lr = np.log(close)
    times = [t for t in range(1440, n - STEP, STEP)]
    X = {t: features(close, vol, t) for t in times}
    R = {t: lr[:, t + STEP] - lr[:, t] for t in times}               # next-interval log return
    fit_t = [t for t in times if t // 1440 < ndays - test_days]
    test_t = [t for t in times if t // 1440 >= ndays - test_days]
    Xf = np.concatenate([X[t] for t in fit_t])
    yf = np.concatenate([R[t] - R[t].mean() for t in fit_t])
    sd = Xf.std(0)
    cv = RidgeCV(alphas=np.logspace(-2, 6, 33)).fit(Xf / sd, yf)
    a0 = cv.alpha_
    models = {}
    for mult in (1, 10, 100):
        m = Ridge(alpha=a0 * mult).fit(Xf / sd, yf)
        models[f"ridge x{mult}"] = (lambda m: lambda x: m.predict(x / sd))(m)
    models["reversal sign"] = lambda x: -np.sign(x[:, 0])
    print(f"members {n_sym}; fitting intervals {len(fit_t)}, test intervals {len(test_t)}; CV ridge penalty {a0:.3g}")
    print("ridge x1 coefficients (per sd): " + ", ".join(f"{nm} {c * 1e4:+.2f} bp" for nm, c in zip(("ret5", "ret30", "ret120", "volsurp"), cv.coef_)))
    cap = 1.0 / n_sym

    def simulate(pred, ts, k, band, cost):
        """Daily net overlay P&L in basis points of the book."""
        w = np.zeros(n_sym)
        pnl = {}
        for t in ts:
            day, mod = divmod(t, 1440)
            if mod >= 1440 - STEP:                                   # close the tilt at the end of the day
                target = np.zeros(n_sym)
            else:
                p = pred(X[t])
                s = p.std()
                target = np.clip(k * (p - p.mean()) / (s if s > 0 else 1) * cap, -cap, cap)
                target -= target.mean()
                keep = np.abs(target - w) < band * cap
                target[keep] = w[keep]
            turnover = np.abs(target - w).sum()
            w = target
            pnl[day] = pnl.get(day, 0.0) + (w * (np.exp(R[t]) - 1)).sum() * 1e4 - turnover * cost
        return np.array(list(pnl.values()))

    for cost in (0.0, 2.0, 5.0):
        print(f"\n===== cost {cost} bp per unit turnover =====")
        for name, pred in models.items():
            best = None
            for k in (0.25, 0.5, 1.0, 2.0):
                for band in (0.0, 0.25, 0.5, 1.0):
                    g = simulate(pred, fit_t, k, band, cost).mean()
                    if best is None or g > best[0]:
                        best = (g, k, band)
            d = simulate(pred, test_t, best[1], best[2], cost)
            gross = simulate(pred, test_t, best[1], best[2], 0.0)
            print(f"RESULT cost {cost} {name:14s} k {best[1]} band {best[2]}: test net {d.mean():+.2f} bp/day (se {d.std() / np.sqrt(len(d)):.2f}, "
                  f"positive days {np.mean(d > 0):.2f}); gross {gross.mean():+.2f}; fitting-days net {best[0]:+.2f}", flush=True)


if __name__ == "__main__":
    main()
