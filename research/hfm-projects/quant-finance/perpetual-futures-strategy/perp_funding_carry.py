"""Does a tuned perpetual-futures strategy survive on a later period?

Public proxy for HFM's "Minute-Level Perpetual Trading Strategy from Price, Flow,
and Funding Data" track, at hourly resolution, on Binance's public data archive
(https://data.binance.vision, USD-margined perpetuals, 24 large contracts,
2022-2025): hourly bars with taker-buy volume, and funding rates.

research/human-frontier-mission.md section 10 does not propose a trading idea for
this track. It proposes a rule: fix one hypothesis family, tune it on one period,
and judge it only on a later period. This script applies that rule to the two
signal families HFM's agents report using (funding carry and taker-flow reversal):

  * a grid of 36 strategy settings is evaluated on 2022-2023 (development);
  * the same settings are then scored on 2024-2025 (test), net of costs.

Every strategy is market-neutral: long the bottom and short the top of the
cross-section by signal, equal weights, rebalanced at each 8-hour funding time.
Profit = price return + funding received - 4 basis points per unit of turnover.
Sharpe ratios are annualised from 8-hour returns.

This is a study of evaluation, not investment advice, and it is NOT comparable to
HFM's reward.

Usage: python research/hfm-experiments/perp_funding_carry.py <dir-with-binance-zips>
"""
from __future__ import annotations

import glob
import itertools
import os
import sys
import zipfile

import numpy as np
import pandas as pd

COST = 4e-4
PER_YEAR = 3 * 365


def read_zips(pattern, cols):
    frames = []
    for z in sorted(glob.glob(pattern)):
        with zipfile.ZipFile(z) as zf, zf.open(zf.namelist()[0]) as f:
            df = pd.read_csv(f, header=None, low_memory=False)
        if not str(df.iloc[0, 0]).isdigit():
            df = df.iloc[1:]
        frames.append(df.iloc[:, cols].astype(float))
    return pd.concat(frames)


def load(d):
    symbols = sorted({os.path.basename(f).split("-")[0] for f in glob.glob(os.path.join(d, "*-fundingRate-*.zip"))})
    close, flow, fund = {}, {}, {}
    for s in symbols:
        k = read_zips(os.path.join(d, f"{s}-1h-*.zip"), [0, 4, 5, 9])
        k.columns = ["t", "close", "vol", "buy"]
        k = k.set_index(pd.to_datetime(k.t, unit="ms")).sort_index()
        k = k[~k.index.duplicated()]
        close[s] = k.close.resample("8h").last()
        flow[s] = (2 * k.buy - k.vol).resample("8h").sum() / k.vol.resample("8h").sum()   # signed taker flow share
        f = read_zips(os.path.join(d, f"{s}-fundingRate-*.zip"), [0, 2])
        f.columns = ["t", "rate"]
        f = f.set_index(pd.to_datetime(f.t, unit="ms").dt.round("1h")).sort_index()
        fund[s] = f.rate[~f.index.duplicated()].resample("8h").sum()
    return pd.DataFrame(close), pd.DataFrame(flow), pd.DataFrame(fund).reindex(pd.DataFrame(close).index).fillna(0.0)


def backtest(signal, close, fund, frac):
    """Positions set at the close of period t from signal[t], held over period t+1."""
    ret = close.pct_change().shift(-1)            # return over the next 8 hours
    fnext = fund.shift(-1)                        # funding paid at the end of that period (long pays if positive)
    rank = signal.rank(axis=1, pct=True)
    valid = signal.notna() & ret.notna()
    w = (rank >= 1 - frac).astype(float) * -1 + (rank <= frac).astype(float)   # short high signal, long low signal
    w = w.where(valid, 0.0)
    w = w.div(w.abs().sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    gross = (w * (ret - fnext)).sum(axis=1)
    turnover = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    return (gross - COST * turnover).dropna()


def sharpe(x):
    return float(x.mean() / x.std() * np.sqrt(PER_YEAR)) if x.std() > 0 else 0.0


def main():
    close, flow, fund = load(sys.argv[1])
    print(f"{close.shape[1]} contracts, {len(close)} eight-hour periods, {close.index[0].date()} to {close.index[-1].date()}")
    signals = {}
    for n in (1, 3, 9, 21):        # funding carry: average funding over the last n periods (high funding -> short)
        signals[f"funding carry, {n}-period average"] = fund.rolling(n).mean().where(close.notna())
    for n in (1, 3, 9):            # taker-flow reversal: heavy recent buying -> short
        signals[f"flow reversal, {n}-period"] = flow.rolling(n).mean()
    for n in (3, 9):               # price reversal as a control family
        signals[f"price reversal, {n}-period"] = close.pct_change(n)
    rows = []
    for (name, sig), frac in itertools.product(signals.items(), (0.125, 0.25, 0.375, 0.5)):
        pnl = backtest(sig, close, fund, frac)
        dev, test = pnl[:"2023-12-31"], pnl["2024-01-01":]
        rows.append(dict(setting=f"{name}, top/bottom {frac:.0%}", family=name.split(",")[0], dev=sharpe(dev), test=sharpe(test),
                         test_ret=float(test.mean() * PER_YEAR)))
    r = pd.DataFrame(rows).sort_values("dev", ascending=False).reset_index(drop=True)
    print(f"\n{len(r)} settings, ranked by development-period Sharpe (2022-2023); test is 2024-2025, all net of costs")
    print(r.head(8).round(2).to_string())
    top = r.iloc[0]
    pos = r[r.dev > 0]
    print(f"\nbest setting on development: Sharpe {top.dev:.2f} -> {top.test:.2f} on test")
    print(f"top five on development: mean Sharpe {r.dev.head(5).mean():.2f} -> {r.test.head(5).mean():.2f} on test")
    print(f"settings profitable on development: {len(pos)} of {len(r)}; of those, still profitable on test: {(pos.test > 0).sum()}")
    print(f"rank correlation between development and test Sharpe: {r.dev.corr(r.test, method='spearman'):+.2f}")
    fc = r[r.family == "funding carry"]
    print(f"within the funding-carry family ({len(fc)} settings): rank correlation {fc.dev.corr(fc.test, method='spearman'):+.2f}")
    print("\nby family (mean Sharpe across its settings):")
    print(r.groupby("family")[["dev", "test"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
