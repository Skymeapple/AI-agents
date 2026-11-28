"""E2: Smooth-latent regime with real trained networks.

A random teacher network defines K_MAX=16 categorical sub-answers ("steps") per input.
A student MLP with shared trunk and K_MAX softmax heads is trained online with Adam.
A k-step task = all of the first k sub-answers correct, sampled at temperature 1, so the
per-item success probability is exp(sum_j log p_ij) (pass@1 under sampling).
Scale variable x = log10(training step).  At every checkpoint we store, for N_EVAL items,
the per-item log-likelihood of the correct k-step answer. Forecasters from E1 are then applied
to subsampled benchmarks of size n with Bernoulli-sampled outcomes.
Usage: python3 e2_teacher_student.py [n_seeds]
"""
import json, os, sys, time
import numpy as np
from scipy.optimize import brentq
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
import e1_leadtime_theory as E1

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
D, TH, K_MAX, C = 8, 8, 16, 4
WIDTH, BATCH, STEPS, LR = 128, 64, 40000, 3e-3
N_EVAL = 10000
KS = [4, 8, 16]
NS = [100, 1000, 10000]


def train(seed):
    rng = np.random.default_rng(seed)
    Wt1 = rng.standard_normal((D, TH)) / np.sqrt(D)
    Wt2 = rng.standard_normal((TH, K_MAX, C)) * 2.0 / np.sqrt(TH)

    def teacher(X):
        return np.argmax(np.einsum("nh,hkc->nkc", np.tanh(X @ Wt1 * 2.0), Wt2), -1)

    params = dict(W1=rng.standard_normal((D, WIDTH)) / np.sqrt(D), b1=np.zeros(WIDTH),
                  W2=rng.standard_normal((WIDTH, K_MAX * C)) / np.sqrt(WIDTH) * 0.1, b2=np.zeros(K_MAX * C))
    m = {k: np.zeros_like(v) for k, v in params.items()}
    s = {k: np.zeros_like(v) for k, v in params.items()}
    Xe = rng.standard_normal((N_EVAL, D))
    Ye = teacher(Xe)
    ckpts = np.unique(np.round(np.logspace(1, np.log10(STEPS), 70)).astype(int))
    steps, LL = [], []      # LL: (n_ckpt, N_EVAL, K_MAX) log-prob of correct sub-answer
    ci = 0
    for t in range(1, STEPS + 1):
        X = rng.standard_normal((BATCH, D))
        Y = teacher(X)
        H = X @ params["W1"] + params["b1"]
        A = np.maximum(H, 0)
        Z = (A @ params["W2"] + params["b2"]).reshape(BATCH, K_MAX, C)
        Z -= Z.max(-1, keepdims=True)
        P = np.exp(Z); P /= P.sum(-1, keepdims=True)
        G = P.copy()
        G[np.arange(BATCH)[:, None], np.arange(K_MAX)[None, :], Y] -= 1
        G = G.reshape(BATCH, K_MAX * C) / BATCH
        grads = dict(W2=A.T @ G, b2=G.sum(0))
        GA = (G @ params["W2"].T) * (H > 0)
        grads["W1"] = X.T @ GA
        grads["b1"] = GA.sum(0)
        for k in params:
            m[k] = 0.9 * m[k] + 0.1 * grads[k]
            s[k] = 0.999 * s[k] + 0.001 * grads[k] ** 2
            params[k] -= LR * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(s[k] / (1 - 0.999 ** t)) + 1e-8)
        if t == ckpts[ci]:
            Ze = (np.maximum(Xe @ params["W1"] + params["b1"], 0) @ params["W2"] + params["b2"]).reshape(N_EVAL, K_MAX, C)
            Ze -= Ze.max(-1, keepdims=True)
            lse = np.log(np.exp(Ze).sum(-1))
            lp = np.take_along_axis(Ze, Ye[..., None], -1)[..., 0] - lse
            steps.append(t); LL.append(lp.astype(np.float32))
            ci += 1
            if ci == len(ckpts):
                break
    return np.array(steps), np.array(LL)


def lead_from_traj(ts, pr, xt, eps=E1.EPS):
    pr = np.asarray(pr, float)
    ok = np.abs(pr - xt) <= eps
    for i in range(len(ts)):
        if ok[i:].all():
            return float(xt - ts[i])
    return 0.0


def analyse(seed):
    steps, LL = train(seed)
    xs_all = np.log10(steps)
    rng = np.random.default_rng(seed + 12345)
    out = []
    for k in KS:
        Li = LL[:, :, :k].sum(-1).astype(float)          # (n_ckpt, N_EVAL)
        accm = np.exp(Li).mean(1)                       # expected pass@1 on full eval set
        if accm[-1] < E1.TAU:
            out.append(dict(seed=seed, k=k, reached=False, final_acc=float(accm[-1])))
            continue
        j = np.where(accm >= E1.TAU)[0][0]
        # log-linear interpolation of crossing in x
        xt = float(np.interp(E1.TAU, [accm[j - 1], accm[j]], [xs_all[j - 1], xs_all[j]]))
        for n in NS:
            idx = rng.choice(N_EVAL, n, replace=False)
            Lsub = Li[:, idx]
            cnt = (rng.random(Lsub.shape) < np.exp(Lsub)).sum(1).astype(float)
            Lbar = Lsub.mean(1)
            cthr = E1.detection_threshold(n)
            fire = np.where(cnt >= cthr)[0]
            lead_det = float(xt - xs_all[fire[0]]) if len(fire) else 0.0
            ts, P = [], {kk: [] for kk in ["acc", "proxy", "proxy_het", "proxy_lin", "proxy_item"]}
            for jj in range(6, len(xs_all), 2):
                if xs_all[jj] > xt:
                    break
                xs = xs_all[: jj + 1]
                ts.append(xs_all[jj])
                P["acc"].append(E1.fit_acc_logistic(xs, cnt[: jj + 1], n))
                P["proxy"].append(E1.fit_proxy(xs, Lbar[: jj + 1], k, False))
                P["proxy_het"].append(E1.fit_proxy(xs, Lbar[: jj + 1], k, True))
                P["proxy_lin"].append(E1.fit_proxy_linear(xs, Lbar[: jj + 1], W=0.75))
                P["proxy_item"].append(E1.fit_proxy_item(xs, Lbar[: jj + 1], Lsub[jj], k))
            ts = np.array(ts)
            row = dict(seed=seed, k=k, n=n, reached=True, x_tau=xt, cthr=cthr, lead_detect=lead_det,
                       max_possible_lead=float(xt - ts[0]) if len(ts) else 0.0,
                       traj=dict(t=ts.tolist(), **{kk: [float(v) for v in vv] for kk, vv in P.items()}))
            for kk in P:
                row["leadfc_" + kk] = lead_from_traj(ts, P[kk], xt) if len(ts) else 0.0
            out.append(row)
    # save a compact curve summary for plotting
    curves = dict(seed=seed, x=xs_all.tolist(),
                  acc={k: np.exp(LL[:, :, :k].sum(-1)).mean(1).tolist() for k in KS},
                  logp={k: LL[:, :, :k].sum(-1).mean(1).tolist() for k in KS},
                  item_sd={k: LL[:, :, :k].sum(-1).std(1).tolist() for k in KS})
    return out, curves


def main():
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    t0 = time.time()
    with Pool(4) as p:
        res = p.map(analyse, list(range(ns)), chunksize=1)
    rows = [r for o, _ in res for r in o]
    curves = [c for _, c in res]
    json.dump(dict(rows=rows, curves=curves), open(os.path.join(OUT, "e2_teacher_student.json"), "w"))
    summ = []
    for k in KS:
        for n in NS:
            rr = [r for r in rows if r["k"] == k and r.get("n") == n and r["reached"]]
            if not rr:
                continue
            row = dict(k=k, n=n, seeds=len(rr), x_tau=float(np.mean([r["x_tau"] for r in rr])),
                       max_possible_lead=float(np.mean([r["max_possible_lead"] for r in rr])),
                       lead_detect=float(np.mean([r["lead_detect"] for r in rr])),
                       lead_detect_sd=float(np.std([r["lead_detect"] for r in rr])))
            for kk in ["acc", "proxy", "proxy_het", "proxy_lin", "proxy_item"]:
                v = [r["leadfc_" + kk] for r in rr]
                row["leadfc_" + kk] = float(np.mean(v)); row["leadfc_" + kk + "_sd"] = float(np.std(v))
            summ.append(row)
    json.dump(summ, open(os.path.join(OUT, "e2_summary.json"), "w"), indent=1)
    for r in summ:
        print({a: (round(b, 3) if isinstance(b, float) else b) for a, b in r.items()})
    notreached = [r for r in rows if not r["reached"]]
    print("not reached:", [(r["seed"], r["k"], round(r["final_acc"], 3)) for r in notreached])
    print("elapsed", time.time() - t0)


if __name__ == "__main__":
    main()
