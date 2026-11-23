"""Monte Carlo + analytic model of AI-accelerator output under supply disruptions.

Accelerator output is Leontief in three inputs (advanced logic dies, CoWoS-class
packaging, HBM).  Each input is produced at sites in regions; hazards remove site
capacity for random durations.  AI's claim on each input depends on the allocation
rule (priority vs proportional rationing) and on AI's share of that input.
"""
import numpy as np

INPUTS = ("logic", "cowos", "hbm")


def _overlap(start, dur, W):
    """(E, W) matrix of the fraction of each week [w, w+1) covered by [start, start+dur)."""
    w = np.arange(W)[None, :]
    s = start[:, None]
    e = (start + dur)[:, None]
    return np.clip(np.minimum(w + 1, e) - np.maximum(w, s), 0.0, 1.0)


def _lognormal(rng, median, sigma, n):
    return median * np.exp(sigma * rng.standard_normal(n))


def simulate(cfg, n_years=20000, seed=0, hazards_on=None, immune_inputs=(), batch=10000,
             return_detail=False):
    """Simulate n_years independent years. Returns dict of per-year metrics.

    hazards_on: iterable of hazard names to include (default: all in cfg).
    immune_inputs: inputs whose sites never lose capacity (counterfactual 'perfectly hardened').
    """
    hz = cfg["hazards"]
    hazards_on = list(hz.keys()) if hazards_on is None else list(hazards_on)
    m = cfg["model"]
    W = m["weeks"]
    sites = cfg["sites"]
    s_ai = cfg["ai_share"]
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("loss", "loss_nobuf", "geo", "min_y")}
    if return_detail:
        out["y"] = []
    for b0 in range(0, n_years, batch):
        N = min(batch, n_years - b0)
        L = np.zeros((len(sites), N, W), dtype=np.float64)
        port_mask = np.zeros((N, W))  # 1 where non-TSMC logic is usable by AI
        geo_flag = np.zeros(N, dtype=bool)

        def apply(idx_sites, years, start, dur, loss):
            if len(years) == 0:
                return
            ov = _overlap(start, dur, W) * loss[:, None]
            for j in idx_sites:
                if sites[j]["inp"] in immune_inputs:
                    continue
                np.add.at(L[j], years, ov)

        def draw(n, h):
            years = rng.integers(0, N, n)
            start = rng.uniform(0, W, n)
            dur = _lognormal(rng, h["dur_median"], h["dur_sigma"], n)
            lo, hi = h["loss"]
            loss = rng.uniform(lo, hi, n) if hi > lo else np.full(n, lo)
            return years, start, dur, loss

        if "local" in hazards_on:
            h = hz["local"]
            for j in range(len(sites)):
                n = rng.poisson(h["rate_per_site"] * N)
                apply([j], *draw(n, h))
        for name, region in (("tw_quake", "TW"), ("kr_regional", "KR")):
            if name in hazards_on:
                h = hz[name]
                n = rng.poisson(h["rate"] * N)
                idx = [j for j, s in enumerate(sites) if s["region"] == region]
                apply(idx, *draw(n, h))
        if "hbm_yield" in hazards_on:
            h = hz["hbm_yield"]
            for sup in sorted({s["supplier"] for s in sites if s["inp"] == "hbm"}):
                n = rng.poisson(h["rate_per_supplier"] * N)
                idx = [j for j, s in enumerate(sites) if s["inp"] == "hbm" and s["supplier"] == sup]
                apply(idx, *draw(n, h))
        if "tw_geo" in hazards_on:
            h = hz["tw_geo"]
            yrs = np.nonzero(rng.random(N) < h["rate"])[0]
            n = len(yrs)
            start = rng.uniform(0, W, n)
            dur = _lognormal(rng, h["dur_median"], h["dur_sigma"], n)
            loss = np.ones(n)
            idx = [j for j, s in enumerate(sites) if s["region"] == "TW"]
            apply(idx, yrs, start, dur, loss)
            geo_flag[yrs] = True
            # non-TSMC logic becomes usable after the porting lag, while the cut-off lasts
            ps = start + m["port_lag_weeks"]
            pd = np.maximum(0.0, start + dur - ps)
            port_mask[yrs] = _overlap(ps, pd, W)

        A = np.clip(1.0 - L, 0.0, 1.0)
        r = []
        for inp in INPUTS:
            usable = np.zeros((N, W))
            base = 0.0
            for j, s in enumerate(sites):
                if s["inp"] != inp:
                    continue
                if s["portable"]:
                    usable += s["share"] * A[j]
                    base += s["share"]
                else:
                    usable += s["share"] * A[j] * port_mask
            if m["allocation"] == "priority":
                r.append(np.minimum(usable / s_ai[inp], 1.0 / s_ai[inp]))
            else:
                r.append(usable / base * (1.0 + m["headroom_prop"]))
        y = np.minimum.reduce(r)
        # inventory buffer
        B = m["buffer_weeks"]
        inv = np.full(N, B)
        d = np.empty_like(y)
        for t in range(W):
            d[:, t] = np.minimum(1.0, y[:, t] + inv)
            inv = np.minimum(B, inv + y[:, t] - d[:, t])
        out["loss"].append(1.0 - d.mean(1))
        out["loss_nobuf"].append(1.0 - np.minimum(1.0, y).mean(1))
        out["geo"].append(geo_flag)
        out["min_y"].append(np.minimum(1.0, y).min(1))
        if return_detail:
            out["y"].append(y)
    return {k: np.concatenate(v) for k, v in out.items()}


def risk_metrics(loss, alpha=(0.95, 0.99, 0.999)):
    loss = np.asarray(loss)
    res = {"mean": float(loss.mean()), "p_gt_1pct": float((loss > 0.01).mean()),
           "p_gt_10pct": float((loss > 0.10).mean()), "p_gt_25pct": float((loss > 0.25).mean())}
    srt = np.sort(loss)
    n = len(srt)
    for a in alpha:
        k = int(np.floor(a * n))
        res[f"VaR{a}"] = float(srt[min(k, n - 1)])
        res[f"CVaR{a}"] = float(srt[k:].mean()) if k < n else float(srt[-1])
    # bootstrap s.e. of the mean
    res["mean_se"] = float(loss.std(ddof=1) / np.sqrt(n))
    return res


# ---------------------------------------------------------------------------
# Analytic results (Propositions 1-3 in the paper)
# ---------------------------------------------------------------------------
def slack_tolerance(s):
    """Prop. 1: largest fraction of an input's capacity that can be lost with zero AI
    output loss under priority allocation."""
    return 1.0 - s


def output_under_capacity_loss(loss_frac, s, mode="priority"):
    """AI output (fraction of demand) when an input loses `loss_frac` of capacity."""
    avail = 1.0 - np.asarray(loss_frac)
    if mode == "priority":
        return np.minimum(1.0, avail / s)
    return avail


def regional_cutoff_output(out_share, s, mode="priority"):
    """Prop. 2: AI output during a full regional cut-off, given each input's capacity share
    outside the region (immediately usable) and AI shares s.  Leontief => min over inputs."""
    vals = {}
    for k in out_share:
        vals[k] = min(1.0, out_share[k] / s[k]) if mode == "priority" else out_share[k]
    return min(vals.values()), vals


def required_outside_capacity(y, s):
    """Prop. 2 corollary: minimum outside-region capacity share per input to keep fraction y of AI output."""
    return {k: y * v for k, v in s.items()}


def buffered_loss_lognormal(b, median, sigma, n=400000, seed=1):
    """Prop. 3 helper: E[(D-b)^+]/E[D] for lognormal outage duration D (full outage)."""
    rng = np.random.default_rng(seed)
    D = median * np.exp(sigma * rng.standard_normal(n))
    return float(np.maximum(D - b, 0).mean() / D.mean())
