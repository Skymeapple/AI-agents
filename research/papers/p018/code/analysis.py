"""Survival analysis of finite-universe-verified graph conjectures.

For every candidate inequality template e(G) <= c over 14 exact invariants we compute the
per-order maximum M_e(j) over ALL connected graphs of order j (j = 2..10, exhaustive), over a
deterministic 4/256 slice of all connected 11-vertex graphs, and over a structured+random sample
of connected graphs with 11..16 vertices.  A conjecture "accepted at universe k" is the sharpest
member of the template that holds on every connected graph with 2..k vertices,
    C_k(e):  e(G) <= E_k(e) := max_{2<=j<=k} M_e(j),
which is exactly what a Graffiti/TxGraffiti-style generator with an exhaustive verifier returns.
We then measure how often C_k(e) is refuted at larger orders, and how well cheap statistics
computable from orders <= k predict refutation.

Run:  python3 analysis.py      (needs data/ produced by run_all.sh; ~2-4 minutes CPU)
"""
import os, json, itertools, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)

NAMES = ["n", "m", "delta", "Delta", "alpha", "omega", "chi", "gamma", "nu",
         "diam", "rad", "res", "ann", "tri"]
IDX = {s: i for i, s in enumerate(NAMES)}
ORDERS = list(range(2, 11))          # exhaustive orders
BIG = list(range(11, 17))            # sampled orders
KS = list(range(4, 11))              # universes at which conjectures are accepted


def load(fname):
    a = np.fromfile(os.path.join(DATA, fname), dtype=np.uint16).reshape(-1, len(NAMES))
    return a


def dedupe(a):
    """unique invariant vectors with multiplicities"""
    a8 = np.ascontiguousarray(a.astype(np.int32))
    u, inv, c = np.unique(a8, axis=0, return_inverse=True, return_counts=True)
    return u.astype(np.int64), c


def build_sets():
    sets = {}
    counts = {}
    for j in ORDERS:
        a = load(f"inv_n{j}.bin")
        counts[j] = int(len(a))
        sets[j] = dedupe(a)
    a = load("inv_n11_slice.bin"); counts["n11_slice"] = int(len(a))
    big = {11: [dedupe(a)[0]]}
    L = load("inv_large.bin")
    tags = open(os.path.join(DATA, "large_meta.txt")).read().split()
    counts["large_sample"] = int(len(L))
    for j in BIG:
        sel = L[:, 0] == j
        big.setdefault(j, []).append(L[sel].astype(np.int64))
    big = {j: np.vstack(v) for j, v in big.items()}
    return sets, big, counts, L.astype(np.int64), tags


# ---------------------------------------------------------------- templates
A_LIN = [(1, 3), (1, 2), (2, 3), (1, 1), (3, 2), (2, 1), (3, 1)]   # a = p/q, closed under inversion
A_SUM = [(1, 2), (1, 1), (2, 1)]
A_PROD = [(1, 2), (1, 1)]


def templates():
    T = []
    k = len(NAMES)
    for X, Y in itertools.permutations(range(k), 2):
        for p, q in A_LIN:
            T.append(("lin", X, Y, -1, p, q, +1))
    for X in range(k):
        for Y, Z in itertools.combinations(range(k), 2):
            if X in (Y, Z):
                continue
            for p, q in A_SUM:
                for s in (+1, -1):
                    T.append(("sum", X, Y, Z, p, q, s))
        for Y, Z in itertools.combinations_with_replacement(range(k), 2):
            if X in (Y, Z):
                continue
            for p, q in A_PROD:
                for s in (+1, -1):
                    T.append(("prod", X, Y, Z, p, q, s))
    return T


def evaluate(t, R):
    kind, X, Y, Z, p, q, s = t
    if kind == "lin":
        W = R[:, Y]
    elif kind == "sum":
        W = R[:, Y] + R[:, Z]
    else:
        W = R[:, Y] * R[:, Z]
    return s * (q * R[:, X] - p * W)


def describe(t, c):
    """human-readable inequality equivalent to s*(qX - pW) <= c"""
    kind, X, Y, Z, p, q, s = t
    if kind == "lin":
        W = NAMES[Y]
    elif kind == "sum":
        W = f"({NAMES[Y]}+{NAMES[Z]})"
    else:
        W = f"{NAMES[Y]}*{NAMES[Z]}" if Y != Z else f"{NAMES[Y]}^2"
    a = f"{p}/{q}" if q != 1 else (f"{p}" if p != 1 else "")
    coef = (a + "*" if a and a != "1" else "")
    if s > 0:   # qX - pW <= c  ->  X <= (p/q)W + c/q
        b = c / q
        return f"{NAMES[X]} <= {coef}{W} {'+' if b >= 0 else '-'} {abs(b):g}"
    else:       # pW - qX <= c  ->  X >= (p/q)W - c/q
        b = -c / q
        return f"{NAMES[X]} >= {coef}{W} {'+' if b >= 0 else '-'} {abs(b):g}"


def auc(score, label):
    """Mann-Whitney AUC; label True = refuted; higher score = predicted refuted"""
    score = np.asarray(score, float); label = np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float)
    # average ties
    order = np.argsort(allv); sv = allv[order]; r = np.empty_like(ranks)
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def main():
    sets, big, counts, L, tags = build_sets()
    T = templates()
    nT = len(T)
    print("templates:", nT, "graph counts:", counts)
    M = np.full((nT, 17), np.iinfo(np.int64).min, dtype=np.int64)   # per-order maxima, index by order
    TIGHTCNT = np.zeros((nT, 17), dtype=np.int64)                    # #graphs of order j attaining M(j)
    for ti, t in enumerate(T):
        for j in ORDERS:
            u, c = sets[j]
            v = evaluate(t, u)
            mx = v.max(); M[ti, j] = mx; TIGHTCNT[ti, j] = c[v == mx].sum()
        for j in big:
            M[ti, j] = evaluate(t, big[j]).max()
    np.save(os.path.join(RES, "per_order_maxima.npy"), M)

    kinds = np.array([t[0] for t in T])
    rows = []          # one row per (template, k)
    summary = {"graph_counts": counts, "n_templates": nT,
               "templates_by_kind": {k: int((kinds == k).sum()) for k in ("lin", "sum", "prod")}}

    # --------------------------------------------------------- per-k statistics
    surv = {}
    pred = {}
    for k in KS:
        E = M[:, 2:k + 1].max(axis=1)                       # accepted bound
        # first counterexample order (exhaustive up to 10, then sampled 11..16)
        fco = np.full(nT, 0)
        for j in list(range(k + 1, 11)) + BIG:
            hit = (M[:, j] > E) & (fco == 0)
            fco[hit] = j
        exh_ref = (fco > 0) & (fco <= 10)
        any_ref = fco > 0
        # features computable from orders <= k
        jfirst = np.array([next(j for j in range(2, k + 1) if M[i, j] == E[i]) for i in range(nT)])
        age = k - jfirst
        attained_top = M[:, k] == E
        d1 = M[:, k] - M[:, k - 1]
        gap = E - M[:, k]
        drift_score = d1 - gap                              # >0 : linear extrapolation breaks the bound
        # least-squares slope over last 3 orders
        jj = np.array([k - 2, k - 1, k], float)
        slope3 = ((M[:, k - 2:k + 1] - M[:, k - 2:k + 1].mean(1, keepdims=True)) * (jj - jj.mean())).sum(1) / ((jj - jj.mean()) ** 2).sum()
        drift3 = slope3 - gap
        tiny_only = jfirst <= 3 & (~attained_top)
        degenerate = np.array([all(M[i, j] < E[i] for j in range(4, k + 1)) for i in range(nT)])
        S = {}
        for name, mask in [("all", np.ones(nT, bool)), ("lin", kinds == "lin"), ("sum", kinds == "sum"),
                           ("prod", kinds == "prod"), ("attained_at_top_order", attained_top),
                           ("not_attained_at_top_order", ~attained_top),
                           ("degenerate_tiny_only", degenerate),
                           ("drift_safe(drift_score<=0)", drift_score <= 0),
                           ("drift_unsafe(drift_score>0)", drift_score > 0)]:
            nm = int(mask.sum())
            S[name] = {"n": nm,
                       "refuted_by_order10": int((exh_ref & mask).sum()),
                       "refuted_any_observed": int((any_ref & mask).sum()),
                       "frac_refuted_by_order10": float((exh_ref & mask).sum() / nm) if nm else None,
                       "frac_refuted_any": float((any_ref & mask).sum() / nm) if nm else None}
        # survival curve S(k -> K): fraction not refuted at any order <= K
        curve = {}
        for K in range(k, 17):
            curve[K] = float(1 - ((fco > 0) & (fco <= K)).mean())
        S["survival_curve_all"] = curve
        # relaxed bound: one unit of slack in X
        q = np.array([t[5] for t in T])
        E1 = E + q
        relaxed_ref = np.zeros(nT, bool)
        for j in list(range(k + 1, 11)) + BIG:
            relaxed_ref |= M[:, j] > E1
        S["relaxed_by_one_unit"] = {"frac_refuted_any": float(relaxed_ref.mean())}
        surv[k] = S
        # predictor quality, labels = refuted by order 10 (exhaustive) for k<=8
        if k <= 9:
            lab = exh_ref if k <= 8 else any_ref
            pred[k] = {"label": "refuted_by_order10_exhaustive" if k <= 8 else "refuted_any_observed",
                       "n_pos": int(lab.sum()), "n": nT,
                       "AUC_drift_score": auc(drift_score, lab),
                       "AUC_drift3": auc(drift3, lab),
                       "AUC_neg_age": auc(-age, lab),
                       "AUC_attained_top": auc(attained_top.astype(float), lab),
                       "AUC_neg_gap": auc(-gap, lab)}
            # confusion of the parameter-free rule drift_score>0
            r = drift_score > 0
            pred[k]["rule_drift>0"] = {"TP": int((r & lab).sum()), "FP": int((r & ~lab).sum()),
                                       "FN": int((~r & lab).sum()), "TN": int((~r & ~lab).sum())}
        for i in range(nT):
            rows.append((k, i, int(E[i]), int(fco[i]), int(age[i]), int(attained_top[i]), int(d1[i]), int(gap[i])))
    summary["survival"] = surv
    summary["prediction"] = pred

    # ---------------------------------------------------- Dalmatian-like significance at k<=9
    dal = {}
    for k in range(4, 10):
        E = M[:, 2:k + 1].max(axis=1)
        U = np.vstack([sets[j][0] for j in range(2, k + 1)])
        sig = np.zeros(nT, bool)
        groups = {}
        for i, t in enumerate(T):
            # target invariant and direction: s=+1 upper bound on X, s=-1 lower bound on X
            groups.setdefault((t[1], t[6]), []).append(i)
        for (X, s), idxs in groups.items():
            # bound value on X implied by each conjecture: X <= (pW + E)/q   or  X >= (pW - E)/q
            B = np.empty((len(idxs), len(U)))
            for r_, i in enumerate(idxs):
                kind, _, Y, Z, p, q, _s = T[i]
                W = U[:, Y] if kind == "lin" else (U[:, Y] + U[:, Z] if kind == "sum" else U[:, Y] * U[:, Z])
                B[r_] = (p * W + E[i]) / q if s > 0 else (p * W - E[i]) / q
            if s < 0:
                B = -B           # make "smaller is tighter" uniform
            best = B.min(0)
            nbest = (np.isclose(B, best[None, :])).sum(0)
            uniq = (np.isclose(B, best[None, :])) & (nbest[None, :] == 1)
            sig[np.array(idxs)[uniq.any(1)]] = True
        fco = np.zeros(nT, int)
        for j in list(range(k + 1, 11)) + BIG:
            hit = (M[:, j] > E) & (fco == 0); fco[hit] = j
        dal[k] = {"n_significant": int(sig.sum()),
                  "frac_refuted_any_significant": float((fco[sig] > 0).mean()),
                  "frac_refuted_any_nonsignificant": float((fco[~sig] > 0).mean()),
                  "frac_refuted_by10_significant": float(((fco[sig] > 0) & (fco[sig] <= 10)).mean()) if k < 10 else None,
                  "frac_refuted_by10_nonsignificant": float(((fco[~sig] > 0) & (fco[~sig] <= 10)).mean()) if k < 10 else None}
        np.save(os.path.join(RES, f"dalmatian_sig_k{k}.npy"), sig)
    summary["dalmatian_like"] = dal

    # ---------------------------------------------------- late counterexamples (survive <=10, refuted later)
    E10 = M[:, 2:11].max(axis=1)
    late = []
    for i in range(nT):
        for j in BIG:
            if M[i, j] > E10[i]:
                # find the witnessing graphs in the large sample (tags); n11 slice reported as such
                v = evaluate(T[i], L)
                w = np.where(v > E10[i])[0]
                fams = sorted(set(tags[x].split(":")[0] for x in w))
                in_slice = j == 11 and (evaluate(T[i], big[11][: len(big[11]) - int((L[:, 0] == 11).sum())]) > E10[i]).any()
                late.append({"conjecture_true_on_all_n<=10": describe(T[i], int(E10[i])),
                             "first_refuting_order": j, "families_in_large_sample": fams,
                             "refuted_in_n11_slice": bool(in_slice), "kind": T[i][0]})
                break
    summary["late_counterexamples_count"] = len(late)
    json.dump(late, open(os.path.join(RES, "late_counterexamples.json"), "w"), indent=1)

    # survivors at k=10 and sanity check on known theorems
    surv10 = [i for i in range(nT) if all(M[i, j] <= E10[i] for j in BIG)]
    summary["survivors_k10"] = len(surv10)
    known = {
        "alpha <= n - nu (König–Gallai-type, true for all graphs)": ("lin", IDX["alpha"], IDX["nu"], -1, 1, 1, +1, None),
        "gamma <= n/2 (Ore 1962)": ("lin", IDX["gamma"], IDX["n"], -1, 1, 2, +1, 0),
        "chi <= Delta + 1 (greedy)": ("lin", IDX["chi"], IDX["Delta"], -1, 1, 1, +1, 1),
        "omega <= chi": ("lin", IDX["omega"], IDX["chi"], -1, 1, 1, +1, 0),
        "diam <= 2 rad": ("lin", IDX["diam"], IDX["rad"], -1, 2, 1, +1, 0),
        "gamma <= alpha": ("lin", IDX["gamma"], IDX["alpha"], -1, 1, 1, +1, 0),
        "res <= alpha (Favaron-Mahéo-Saclé)": ("lin", IDX["res"], IDX["alpha"], -1, 1, 1, +1, 0),
        "alpha <= ann (Pepper)": ("lin", IDX["alpha"], IDX["ann"], -1, 1, 1, +1, 0),
    }
    checks = {}
    tindex = {t: i for i, t in enumerate(T)}
    for name, t in known.items():
        key = t[:7]
        if t[0] == "lin" and name.startswith("alpha <= n - nu"):
            # alpha + nu <= n : expressed as alpha <= n - nu is not a template; check directly
            ok = all((evaluate(("sum", IDX["n"], IDX["alpha"], IDX["nu"], 1, 1, -1), sets[j][0]) <= 0).all() for j in ORDERS)
            ok = ok and (big_ok := all((evaluate(("sum", IDX["n"], IDX["alpha"], IDX["nu"], 1, 1, -1), big[j]) <= 0).all() for j in BIG))
            checks[name] = {"holds_on_all_data": bool(ok)}
            continue
        i = tindex[key]
        checks[name] = {"accepted_bound_at_k10": describe(T[i], int(E10[i])),
                        "matches_theorem_constant": int(E10[i]) == t[7] * T[i][5] if t[7] is not None else None,
                        "survives_all_tests": i in set(surv10)}
    summary["known_theorem_checks"] = checks
    json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=1, default=str)
    with open(os.path.join(RES, "conjectures_by_k.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["k", "template_id", "bound_E_k", "first_counterexample_order(0=none)", "age", "attained_top", "d1", "gap"])
        w.writerows(rows)
    with open(os.path.join(RES, "templates.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["template_id", "kind", "X", "Y", "Z", "p", "q", "sign", "statement_at_k10"])
        for i, t in enumerate(T):
            w.writerow([i, t[0], NAMES[t[1]], NAMES[t[2]], NAMES[t[3]] if t[3] >= 0 else "", t[4], t[5], t[6], describe(t, int(E10[i]))])
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("all", "attained_at_top_order", "drift_safe(drift_score<=0)", "relaxed_by_one_unit")} for k, v in surv.items()}, indent=1))
    print(json.dumps(pred, indent=1)); print(json.dumps(dal, indent=1)); print("late:", len(late), "survivors k10:", len(surv10))
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
