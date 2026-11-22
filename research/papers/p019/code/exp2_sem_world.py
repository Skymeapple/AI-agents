"""E2/E3: synthetic 'scientific worlds' (linear SEMs with latent confounders).

An AI hypothesis generator mines a large observational dataset for ORIGINAL causal claims
"X_i -> X_j with sign s" (pairs not already in the simulated literature), and the claims are
verified by different strategies. Ground truth is the total causal effect T[i, j].

Verifiers:
  self      : significance on the generator's own data
  replicate : fresh observational sample, same environment (same confounding)
  envK      : K fresh observational samples from K environments with resampled confounding
              weights (independent bias for confounding; causal mechanism B invariant)
  intervene : randomized experiment on X_i (X_i assigned exogenously), n_int units

E3 (prediction test): per-modality pass rates estimated on calibration worlds are used to
predict FDR for K environments under (a) a naive independence model and (b) a bias-source
decomposition (reverse causation is shared across all observational modalities, confounding
is not). Predictions are compared with held-out worlds.
"""
import json, os, sys, time
import numpy as np
from scipy import stats

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
D, L, P_EDGE, GAMMA = 30, 6, 0.08, 0.8
N_OBS, N_INT, M_PROP, KMAX, ALPHA = 5000, 300, 40, 4, 0.05
LIT_FRAC = 0.4
MEAS_SD = 0.0  # measurement-noise sd added to every recorded variable (set >0 in the noisy sweep)
N_WORLDS = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def make_world(rng):
    order = rng.permutation(D)
    B = np.zeros((D, D))  # B[i, j] = direct effect of i on j
    for a in range(D):
        for b in range(a + 1, D):
            if rng.random() < P_EDGE:
                w = rng.uniform(0.3, 1.0) * rng.choice([-1, 1])
                B[order[a], order[b]] = w
    T = np.linalg.inv(np.eye(D) - B) - np.eye(D)  # total effects (i -> j), zero diag
    T[np.abs(T) < 1e-10] = 0.0
    masks = [rng.choice(D, size=6, replace=False) for _ in range(L)]
    return dict(B=B, T=T, masks=masks)


def confounding(world, rng):
    G = np.zeros((L, D))
    for l, m in enumerate(world["masks"]):
        G[l, m] = rng.normal(0, GAMMA, len(m))
    return G


def sample_obs(world, G, n, rng):
    U = rng.normal(size=(n, L))
    E = rng.normal(size=(n, D)) + U @ G
    X = E @ np.linalg.inv(np.eye(D) - world["B"])  # X = X B + E  -> X = E (I-B)^-1
    return X + MEAS_SD * rng.normal(size=X.shape) if MEAS_SD > 0 else X


def slope_t(x, y):
    x = x - x.mean(); y = y - y.mean()
    beta = (x @ y) / (x @ x)
    r = y - beta * x
    se = np.sqrt((r @ r) / (len(x) - 2) / (x @ x))
    return beta, beta / se


def one_sided_p(x, y, sign):
    beta, t = slope_t(x, y)
    return float(stats.t.sf(sign * t, len(x) - 2))


def one_sided_pass(x, y, sign):
    return one_sided_p(x, y, sign) < ALPHA


def intervene_pass(world, G, i, j, sign, rng):
    n = N_INT
    U = rng.normal(size=(n, L))
    E = rng.normal(size=(n, D)) + U @ G
    B = world["B"].copy(); B[:, i] = 0.0  # cut incoming edges to i
    E[:, i] = rng.normal(size=n)  # randomized assignment
    X = E @ np.linalg.inv(np.eye(D) - B)
    if MEAS_SD > 0:
        X = X + MEAS_SD * rng.normal(size=X.shape)
    return one_sided_p(X[:, i], X[:, j], sign)


def lit_graph_distance(lit_pairs):
    A = np.zeros((D, D), bool)
    for i, j in lit_pairs:
        A[i, j] = A[j, i] = True
    dist = np.full((D, D), 99)
    for s in range(D):
        dist[s, s] = 0; frontier = [s]; d = 0
        while frontier:
            d += 1; nxt = []
            for u in frontier:
                for v in np.nonzero(A[u])[0]:
                    if dist[s, v] == 99:
                        dist[s, v] = d; nxt.append(v)
            frontier = nxt
    return dist


def run_world(seed, n_obs=N_OBS, m_prop=M_PROP):
    rng = np.random.default_rng(seed)
    w = make_world(rng)
    T = w["T"]
    # literature: strongest LIT_FRAC of true causal pairs (by |total effect|) are already known
    true_pairs = np.argwhere(T != 0)
    if len(true_pairs) < 5:
        return []
    eff = np.abs(T[true_pairs[:, 0], true_pairs[:, 1]])
    k = int(np.ceil(LIT_FRAC * len(true_pairs)))
    lit = true_pairs[np.argsort(-eff)[:k]]
    lit_set = set(map(tuple, lit))
    dist = lit_graph_distance(lit)
    G0 = confounding(w, rng)
    X0 = sample_obs(w, G0, n_obs, rng)
    Xc = X0 - X0.mean(0)
    C = Xc.T @ Xc
    # generator: rank novel ordered pairs by |t| of simple regression of X_j on X_i
    cand = []
    for i in range(D):
        for j in range(D):
            if i == j or (i, j) in lit_set or (j, i) in lit_set:
                continue  # novelty: neither direction of the pair is in the literature
            beta = C[i, j] / C[i, i]
            r2 = C[i, j] ** 2 / (C[i, i] * C[j, j])
            t = np.sign(beta) * np.sqrt(r2 * (n_obs - 2) / max(1 - r2, 1e-12))
            cand.append((abs(t), i, j, np.sign(beta)))
    cand.sort(reverse=True)
    props = cand[:m_prop]
    # fresh data sets
    X_rep = sample_obs(w, G0, n_obs, rng)
    envs = []
    for _ in range(KMAX):
        Gk = confounding(w, rng)
        envs.append(sample_obs(w, Gk, n_obs, rng))
    rows = []
    for absT, i, j, sgn in props:
        true = bool(T[i, j] != 0 and np.sign(T[i, j]) == sgn)
        rev = bool(T[j, i] != 0)
        common = bool(np.any((T[:, i] != 0) & (T[:, j] != 0)))
        latent = bool(any((i in m) and (j in m) for m in w["masks"]))
        p_int = intervene_pass(w, G0, i, j, sgn, rng)
        p_int2 = intervene_pass(w, G0, i, j, sgn, rng)  # independent second experiment
        p_int3 = intervene_pass(w, G0, i, j, sgn, rng)  # independent third experiment
        row = dict(world=seed, i=int(i), j=int(j), absT=float(absT), true=true,
                   wrong_sign=bool(T[i, j] != 0 and np.sign(T[i, j]) != sgn),
                   reverse=rev and not true, dist=int(min(dist[i, j], 99)),
                   common_cause=common, latent_shared=latent, effect=float(T[i, j]),
                   self=bool(absT > stats.t.isf(ALPHA, n_obs - 2)),
                   replicate=bool(one_sided_pass(X_rep[:, i], X_rep[:, j], sgn)),
                   env=[bool(one_sided_pass(E[:, i], E[:, j], sgn)) for E in envs],
                   p_int=p_int, p_int2=p_int2, p_int3=p_int3, rank=len(rows), intervene=bool(p_int < ALPHA))
        rows.append(row)
    return rows


def main():
    t0 = time.time()
    rows = []
    for s in range(N_WORLDS):
        rows += run_world(1000 + s)
    json.dump(rows, open(os.path.join(OUT, "exp2_rows.json"), "w"))
    print("main: worlds", N_WORLDS, "candidates", len(rows), "time %.1fs" % (time.time() - t0))
    global MEAS_SD
    for tag, msd in [("noiseless", 0.0), ("noisy", 4.0)]:
        MEAS_SD = msd
        sweep = {}
        for n in [100, 300, 1000, 3000, 10000, 30000]:
            rs = []
            for s in range(N_WORLDS // 2):
                rs += run_world(5000 + s, n_obs=n)
            sweep[n] = [{k: r[k] for k in ("world", "true", "self", "replicate", "env", "intervene", "p_int")} for r in rs]
            print("sweep", tag, "n", n, len(rs), "time %.1fs" % (time.time() - t0))
        json.dump(sweep, open(os.path.join(OUT, f"exp2_sweep_{tag}.json"), "w"))
    MEAS_SD = 0.0
    # depth-of-search run: generator asked for 200 original claims per world
    rows = []
    for s in range(N_WORLDS // 2):
        rows += run_world(9000 + s, m_prop=200)
    json.dump(rows, open(os.path.join(OUT, "exp2_depth_rows.json"), "w"))
    print("depth run", len(rows), "time %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
