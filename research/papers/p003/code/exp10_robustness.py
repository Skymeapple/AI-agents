"""E10 (revision): robustness checks requested in review round 1.

 (a) alternative exploit encodings (W7): near-miss, anti-correlated, heterogeneous eta
 (b) optimiser variants (P5): entropy bonus, diagonal natural gradient, Adam
 (c) learning speed under SGD vs Adam (t90 vs J; the tension with Lv et al. 2025)
 (d) learning-rate and group-size sensitivity of the E4 noisy-judge result
 (e) E5 replacement column J = 0.75 (alpha = 0.90 < eta), since J = 0.8 made alpha = eta
 (f) E6 metastability: exploit-mass trajectories under a KL penalty vs the tilt prediction
All runs: sequence task of E4 (binary strings, length 10), 3000 steps unless noted."""
import json, os, sys
from multiprocessing import Pool
import numpy as np
import exp4_sequence as E

R = os.path.join(os.path.dirname(__file__), "..", "results")
X, CORRECT = E.X, E.CORRECT
P0_DEFAULT = np.array([0.6] * 5 + [0.5] * 3 + [0.15] * 2)

ENC = {
    "base": (X[:, 0] == 0) & (X[:, 8] == 1) & (X[:, 9] == 1),
    "near-miss": X[:, :4].all(1) & (X[:, 4] == 0) & (X[:, 9] == 1),
    "anti-correlated": (X[:, 1] == 0) & (X[:, 2] == 0) & (X[:, 8] == 1) & (X[:, 9] == 1),
}
ENC["hetero-eta"] = ENC["base"]
ETA_HET = np.random.default_rng(0).uniform(0.6, 1.0, len(X))


def verifier(alpha, beta, eta, enc="base"):
    ex = ENC[enc]
    e = ETA_HET if enc == "hetero-eta" else np.full(len(X), eta)
    return np.where(CORRECT, alpha, np.where(ex, e, beta)), ex


def make(pname, ps=None):
    E.P0 = P0_DEFAULT.copy() if ps is None else np.array([0.6] * 5 + [0.5] * 3 + [ps] * 2)
    return {"factorised": E.Factorised, "AR1": E.AR1, "tabular": E.Tabular}[pname]()


def train(pol, f, ex, steps=3000, G=16, B=8, lr=0.5, seed=0, opt="sgd", c_ent=0.0, log_every=10):
    rng = np.random.default_rng(seed)
    m = v = None; hist = []
    for s in range(steps + 1):
        p = pol.probs()
        if s % log_every == 0:
            hist.append((s, float(p @ CORRECT), float(p @ ex)))
        idx = rng.choice(len(X), size=B * G, p=p)
        r = (rng.random(B * G) < f[idx]).astype(float).reshape(B, G)
        sd = r.std(1, keepdims=True)
        adv = np.where(sd > 1e-8, (r - r.mean(1, keepdims=True)) / np.maximum(sd, 1e-8), 0).ravel()
        g = (adv[:, None] * pol.grad_logp(idx)).mean(0)
        if c_ent > 0:  # factorised only: dH/dtheta_i = -q(1-q) theta_i
            q = E.sig(pol.th); g = g + c_ent * (-q * (1 - q) * pol.th)
        if opt == "sgd":
            d = lr * g
        elif opt == "ngd":  # diagonal Fisher of a Bernoulli logit is q(1-q)
            q = E.sig(pol.th); d = lr * g / np.maximum(q * (1 - q), 1e-3)
        elif opt == "adam":
            if m is None: m = np.zeros_like(g); v = np.zeros_like(g)
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
            mh = m / (1 - 0.9 ** (s + 1)); vh = v / (1 - 0.999 ** (s + 1))
            d = lr * mh / (np.sqrt(vh) + 1e-8)
        if hasattr(pol, "step"): pol.step(d)
        else: pol.th += d
    return np.array(hist)


def job(args):
    tag, pname, ps, (a, b, e, enc), kw, seed = args
    pol = make(pname, ps); f, ex = verifier(a, b, e, enc)
    h0 = float(pol.probs() @ ex)
    h = train(pol, f, ex, seed=seed, **kw)
    hit = np.where(h[:, 1] >= 0.9)[0]
    return {"tag": tag, "policy": pname, "ps": ps, "alpha": a, "beta": b, "eta": e, "enc": enc,
            "kw": {k: v for k, v in kw.items() if k != "log_every"}, "seed": seed, "h0": h0,
            "collapse": bool(h[-1, 2] > 0.5), "final_A": float(h[-1, 1]), "peak_A": float(h[:, 1].max()),
            "t90": float(h[hit[0], 0]) if len(hit) else None}


def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0, c - hw)), float(min(1, c + hw))]


def summarise(res, keys):
    groups = {}
    for r in res:
        k = tuple(json.dumps(r[x], sort_keys=True) if isinstance(r[x], dict) else r[x] for x in keys)
        groups.setdefault(k, []).append(r)
    out = []
    for k, rr in groups.items():
        n = len(rr); c = sum(r["collapse"] for r in rr)
        t = [r["t90"] for r in rr if r["t90"] is not None and not r["collapse"]]
        out.append({**{x: rr[0][x] for x in keys}, "h0": rr[0]["h0"], "n": n, "n_collapse": c,
                    "collapse_frac": c / n, "collapse_CI95": wilson(c, n),
                    "median_t90": float(np.median(t)) if t else None, "n_t90": len(t),
                    "mean_final_A": float(np.mean([r["final_A"] for r in rr]))})
    return out


if __name__ == "__main__":
    jobs = []
    NJ, ML = (0.60, 0.40, 0.95), (0.75, 0.15, 0.95)
    # (a) encodings
    for enc in ENC:
        for vname, v in [("noisy-judge", NJ), ("math-like", ML)]:
            for pname in ["factorised", "AR1", "tabular"]:
                for s in range(8):
                    jobs.append((f"a:{vname}", pname, None, (*v, enc), {}, s))
    # (b) optimisers (factorised)
    OPTS = {"sgd": {"opt": "sgd", "lr": 0.5}, "entropy0.01": {"opt": "sgd", "lr": 0.5, "c_ent": 0.01},
            "entropy0.05": {"opt": "sgd", "lr": 0.5, "c_ent": 0.05}, "ngd": {"opt": "ngd", "lr": 0.12},
            "adam": {"opt": "adam", "lr": 0.05}}
    for oname, kw in OPTS.items():
        for ps, J in [(0.08, 0.2), (0.08, 0.45), (0.15, 0.2), (0.15, 0.45), (0.15, 0.6)]:
            for s in range(16):
                jobs.append((f"b:{oname}", "factorised", ps, (0.15 + J, 0.15, 0.95, "base"), kw, s))
    # (c) t90 under SGD vs Adam, low exploit mass
    for oname in ["sgd", "adam"]:
        for J in [0.1, 0.2, 0.3, 0.45, 0.6, 0.75]:
            for s in range(16):
                jobs.append((f"c:{oname}", "factorised", 0.08, (0.15 + J, 0.15, 0.95, "base"),
                             {**OPTS[oname], "log_every": 5}, 100 + s))
    # (d) lr / group size
    for lr in [0.25, 0.5, 1.0]:
        for G in [8, 16, 32]:
            for pname in ["factorised", "AR1"]:
                for s in range(8):
                    jobs.append((f"d:lr{lr}_G{G}", pname, None, (*NJ, "base"), {"lr": lr, "G": G}, s))
    # (e) J = 0.75 column for E5
    for ps in [0.08, 0.15, 0.25]:
        for pname in ["factorised", "AR1"]:
            for s in range(8):
                jobs.append(("e:J0.75", pname, ps, (0.90, 0.15, 0.95, "base"), {}, s))
    print(len(jobs), "jobs", flush=True)
    with Pool(int(os.environ.get("NPROC", 3))) as pool:
        res = pool.map(job, jobs, chunksize=4)
    out = {"raw": res}
    out["a_encodings"] = summarise([r for r in res if r["tag"].startswith("a:")], ["tag", "enc", "policy"])
    out["b_optimisers"] = summarise([r for r in res if r["tag"].startswith("b:")], ["tag", "ps", "alpha"])
    out["d_lr_group"] = summarise([r for r in res if r["tag"].startswith("d:")], ["tag", "policy"])
    out["e_J075"] = summarise([r for r in res if r["tag"].startswith("e:")], ["tag", "ps", "policy"])
    # (c) learning-speed analysis with bootstrap CIs
    c = {}
    rng = np.random.default_rng(0)
    for oname in ["sgd", "adam"]:
        rows = []
        for J in [0.1, 0.2, 0.3, 0.45, 0.6, 0.75]:
            rr = [r for r in res if r["tag"] == f"c:{oname}" and abs(r["alpha"] - 0.15 - J) < 1e-9]
            t = np.array([r["t90"] for r in rr if r["t90"] is not None and not r["collapse"]])
            bs = [np.median(rng.choice(t, len(t))) for _ in range(2000)] if len(t) else [np.nan]
            rows.append({"J": J, "n_collapse": sum(r["collapse"] for r in rr), "n_t90": int(len(t)),
                         "median_t90": float(np.median(t)) if len(t) else None,
                         "t90_CI95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                         "t90_times_J": float(np.median(t) * J) if len(t) else None, "t90_all": t.tolist()})
        ok = [x for x in rows if x["n_t90"] >= 4]
        lj = np.log([x["J"] for x in ok]); lt = np.log([x["median_t90"] for x in ok])
        slopes = []
        for _ in range(2000):
            lt_b = [np.log(np.median(rng.choice(x["t90_all"], len(x["t90_all"])))) for x in ok]
            slopes.append(np.polyfit(lj, lt_b, 1)[0])
        c[oname] = {"rows": rows, "loglog_slope_t90_vs_J": float(np.polyfit(lj, lt, 1)[0]),
                    "slope_CI95": [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))]}
    # initial reward std (explains why J/sigma ~ J here)
    E.P0 = np.array([0.6] * 5 + [0.5] * 3 + [0.08] * 2); p0 = E.Factorised().probs()
    c["sigma0"] = {str(J): float(np.sqrt((lambda fb: fb * (1 - fb))(p0 @ verifier(0.15 + J, 0.15, 0.95)[0])))
                   for J in [0.1, 0.2, 0.3, 0.45, 0.6, 0.75]}
    out["c_learning_speed"] = c
    json.dump(out, open(os.path.join(R, "exp10.json"), "w"), indent=1)
    for k in ["a_encodings", "b_optimisers", "d_lr_group", "e_J075"]:
        for r in out[k]: print(k, r)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"} if k != "sigma0" else v
                      for k, v in c.items()}))
    for o in ["sgd", "adam"]:
        for r in c[o]["rows"]: print(o, {k: v for k, v in r.items() if k != "t90_all"})
