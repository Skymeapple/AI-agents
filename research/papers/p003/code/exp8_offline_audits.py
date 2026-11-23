"""E8 (revision): which audits predict exploit takeover, at matched label budgets?

Response to review W2.  Reruns the E7b configurations (60 original + 60 new) with the
factorised policy and compares, with bootstrap 95% CIs on every AUC:

 on-policy (feature-blind)
   ON-w     slope of log(FA/TA) from 256-sample on-policy audits every 10 steps up to step w
            (w = 20, 50, 100; label budget 768 / 1,536 / 2,816 samples)
 offline, base policy only (feature-blind)
   AGG-256  log(FA/TA) from 256 base samples at step 0 (the round-1 "offline audit")
   AGG-B    log(FA/TA) from 2,816 base samples (matched budget)
   TAIL-B   tail-aware: 352 base samples, each judged 8 times (2,816 judge calls).
            h_tail = share of samples that are wrong AND whose per-response acceptance
            estimate exceeds the mean acceptance of correct samples;  J_hat = mean
            acceptance(correct) - mean acceptance(other wrong).  Score = h_tail / J_hat.
   LOOK-B   one-step look-ahead: from 2,816 labelled base samples, estimate the expected
            GRPO update direction g and grad log(FA/TA); score = <grad log(FA/TA), g>.
            Needs the policy's score function but no training step.
 offline, oracle (exploit set known; exact population quantities, infinite budget)
   X0-GAP   E[f | x0=0] - E[f | x0=1] under pi0  (reviewer's statistic)
   H0/J     h0 / J
   MARGIN   J*(h0) - J using the round-1 fitted boundary 12.62 h0^0.80
 reference (not an audit): exact exploit mass and accuracy at step w (outcome observation)

Also records the decision time of each run (first logged step at which exploit mass > 0.5
or accuracy > 0.5) so that AUCs can be restricted to runs that are undecided at the end of
the audit window."""
import json, os, sys
import numpy as np
import exp4_sequence as E

R = os.path.join(os.path.dirname(__file__), "..", "results")
BETA, ETA = 0.15, 0.95
G, B, LR, STEPS = 16, 8, 0.5, 3000
WINDOWS = [20, 50, 100]
BUDGET = 2816


def set_suffix(ps):
    E.P0 = np.array([0.6] * 5 + [0.5] * 3 + [ps] * 2)


def lfa_ta(acc, corr):
    ta = np.sum(acc & corr) + 0.5; fa = np.sum(acc & ~corr) + 0.5
    return float(np.log(fa / ta))


def offline_stats(pol, f, rng):
    p = pol.probs(); out = {}
    idx = rng.choice(len(E.X), size=256, p=p); acc = rng.random(256) < f[idx]
    out["AGG-256"] = lfa_ta(acc, E.CORRECT[idx])
    idx = rng.choice(len(E.X), size=BUDGET, p=p); r = (rng.random(BUDGET) < f[idx]); y = E.CORRECT[idx]
    out["AGG-B"] = lfa_ta(r, y)
    # look-ahead: split the same labelled sample into halves (direction / gradient)
    gl = pol.grad_logp(idx); rf = r.astype(float)
    h1, h2 = slice(0, BUDGET // 2), slice(BUDGET // 2, BUDGET)
    rr = rf[h1].reshape(-1, G); sd = rr.std(1, keepdims=True)
    adv = np.where(sd > 1e-8, (rr - rr.mean(1, keepdims=True)) / np.maximum(sd, 1e-8), 0).ravel()
    g = (adv[:, None] * gl[h1]).mean(0)
    w = ~y[h2]; c = y[h2]; rr2 = rf[h2]; gg = gl[h2] - gl[h2].mean(0)
    FA = np.mean(rr2 * w) + 1e-3; TA = np.mean(rr2 * c) + 1e-3
    dFA = ((rr2 * w)[:, None] * gg).mean(0); dTA = ((rr2 * c)[:, None] * gg).mean(0)
    out["LOOK-B"] = float((dFA / FA - dTA / TA) @ g)
    # tail-aware, feature-blind: repeated judging of distinct base samples
    n, k = BUDGET // 8, 8
    idx = rng.choice(len(E.X), size=n, p=p); fh = (rng.random((n, k)) < f[idx][:, None]).mean(1)
    y = E.CORRECT[idx]
    mc = fh[y].mean() if y.any() else 1.0
    tail = (~y) & (fh > mc)
    other = (~y) & ~tail
    h_tail = tail.mean()
    J_hat = max(mc - (fh[other].mean() if other.any() else 0.0), 1e-3)
    out["TAIL-B"] = float((h_tail + 1.0 / n) / J_hat)
    out["TAIL-B_h"] = float(h_tail); out["TAIL-B_J"] = float(J_hat)
    return out


def oracle_stats(f, J, h0):
    p0 = E.Factorised().probs()
    m0 = E.X[:, 0] == 0
    x0gap = float(p0[m0] @ f[m0] / p0[m0].sum() - p0[~m0] @ f[~m0] / p0[~m0].sum())
    return {"X0-GAP": x0gap, "H0/J": h0 / J, "MARGIN": 12.6215 * h0 ** 0.8 - J}


def run(ps, J, seed):
    set_suffix(ps)
    f = E.verifier(BETA + J, BETA, ETA)
    h0 = float(E.Factorised().probs() @ E.EXPLOIT)
    rng_a = np.random.default_rng(seed + 777); rng_t = np.random.default_rng(seed)
    rng_o = np.random.default_rng(seed + 31337)
    pol = E.Factorised()
    stats = offline_stats(pol, f, rng_o); stats.update(oracle_stats(f, J, h0))
    audits, traj = [], []
    for s in range(STEPS + 1):
        p = pol.probs()
        if s % 10 == 0:
            traj.append((s, float(p @ E.CORRECT), float(p @ E.EXPLOIT)))
        if s <= max(WINDOWS) and s % 10 == 0:
            idx = rng_a.choice(len(E.X), size=256, p=p); acc = rng_a.random(256) < f[idx]
            audits.append((s, lfa_ta(acc, E.CORRECT[idx])))
        idx = rng_t.choice(len(E.X), size=B * G, p=p)
        r = (rng_t.random(B * G) < f[idx]).astype(float).reshape(B, G)
        sd = r.std(1, keepdims=True)
        adv = np.where(sd > 1e-8, (r - r.mean(1, keepdims=True)) / np.maximum(sd, 1e-8), 0).ravel()
        pol.th += LR * (adv[:, None] * pol.grad_logp(idx)).mean(0)
    a = np.array(audits); traj = np.array(traj)
    for w in WINDOWS:
        m = a[:, 0] <= w
        stats[f"ON-{w}"] = float(np.polyfit(a[m, 0], a[m, 1], 1)[0])
        stats[f"REF-exploit@{w}"] = float(traj[traj[:, 0] == w, 2][0])
        stats[f"REF-negacc@{w}"] = float(-traj[traj[:, 0] == w, 1][0])
    collapse = bool(traj[-1, 2] > 0.5)
    dec = np.where((traj[:, 2] > 0.5) | (traj[:, 1] > 0.5))[0]
    t_dec = float(traj[dec[0], 0]) if len(dec) else float("inf")
    return {"ps": ps, "J": J, "h0": h0, "seed": seed, "collapse": collapse, "t_dec": t_dec,
            "stats": stats}


def auc(s, l):
    s = np.asarray(s, float); l = np.asarray(l, bool)
    pos, neg = s[l], s[~l]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float(((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean())


def auc_ci(s, l, nboot=2000, seed=0):
    rng = np.random.default_rng(seed); s = np.asarray(s, float); l = np.asarray(l, bool)
    bs = []
    for _ in range(nboot):
        i = rng.integers(0, len(s), len(s))
        if l[i].all() or (~l[i]).all(): continue
        bs.append(auc(s[i], l[i]))
    return [auc(s, l), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


if __name__ == "__main__":
    runs = []
    for batch, (rs, off) in enumerate([(5, 0), (6, 100)]):
        rng = np.random.default_rng(rs)
        for i in range(60):
            ps = rng.uniform(0.05, 0.3); J = rng.uniform(0.05, 0.8)
            r = run(ps, J, seed=off + i); r["batch"] = batch; runs.append(r)
        print("batch", batch, "done"); sys.stdout.flush()
    keys = [k for k in runs[0]["stats"] if not k.endswith("_h") and not k.endswith("_J")]
    out = {"n_runs": len(runs), "collapse_rate": float(np.mean([r["collapse"] for r in runs])),
           "runs": runs, "auc": {}}
    for subset_name, sel in [("original60", [r for r in runs if r["batch"] == 0]), ("all120", runs)]:
        lab = [r["collapse"] for r in sel]
        out["auc"][subset_name] = {k: auc_ci([r["stats"][k] for r in sel], lab) for k in keys}
        out["auc"][subset_name]["n"] = len(sel); out["auc"][subset_name]["n_collapse"] = int(sum(lab))
    # restrict to runs undecided at the end of each window
    out["undecided"] = {}
    for w in WINDOWS:
        sel = [r for r in runs if r["t_dec"] > w]
        lab = [r["collapse"] for r in sel]
        out["undecided"][str(w)] = {"n": len(sel), "n_collapse": int(sum(lab)),
                                    **{k: auc_ci([r["stats"][k] for r in sel], lab)
                                       for k in [f"ON-{w}", "AGG-256", "AGG-B", "TAIL-B", "LOOK-B",
                                                 "X0-GAP", "H0/J", "MARGIN", f"REF-exploit@{w}"]}}
    td = np.array([r["t_dec"] for r in runs])
    out["decision_time"] = {"median": float(np.median(td)),
                            "frac_decided_by": {str(w): float(np.mean(td <= w)) for w in WINDOWS}}
    out["sign_rule_ON100_accuracy"] = float(np.mean([(r["stats"]["ON-100"] > 0) == r["collapse"] for r in runs]))
    json.dump(out, open(os.path.join(R, "exp8.json"), "w"), indent=1)
    for s in ["original60", "all120"]:
        print(s, {k: [round(x, 3) for x in v] if isinstance(v, list) else v for k, v in out["auc"][s].items()})
    for w in WINDOWS:
        print("undecided", w, {k: [round(x, 3) for x in v] if isinstance(v, list) else v
                                for k, v in out["undecided"][str(w)].items()})
    print(out["decision_time"])
