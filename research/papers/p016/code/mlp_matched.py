"""Revision (R1) MLP experiments, same network/data/phase-1 as mlp_experiments.py.
(1) Step-size / progress-matched optimiser comparison at rho in {0.1, 0.3}:
      GD   eta = c / lambda_max(G_rho(theta_0)),  c in {1, 0.3, 0.1}  (4000 / 8000 / 12000 steps)
      Adam lr in {1e-3, 3e-4}
      GGN-flow  delta=1e-3 (relative), lr 0.2, 150 steps
    For each run we store the full (L1, L2) trajectory metrics at matched new-task progress:
      G_q = max_{t <= tau_q} L1(t) - L1(0),  tau_q = first step with L2 <= q * L2(0),  q in QS.
(2) Multi-step short-time test of Thm 2: GD with a COMMON step eta_c = 1/max(lmax(G1), lmax(G2)) at theta_0
    for rho in {0, 0.01, 0.1, 0.3, 0.5}, 30 steps; record Delta L1(k).
(3) NTK-linearised prediction: the same GD dynamics run on the network linearised at theta_0
    (f = f0 + J (theta - theta0)), for the eta_c runs and for GD c=1 at rho in {0.01, 0.1, 0.3} (G, t_peak for rho in {0.1,0.3} from the stored log-spaced curve).
Usage: python mlp_matched.py <scenario> <seed>  ->  ../results/mlpm_<scen>_<seed>.json
"""
import torch, numpy as np, json, os, sys, time
from mlp_experiments import data, phase1, loss, grad, grad_and_G, jac, fwd, onehot
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
QS = [0.5, 0.25, 0.1, 0.05]


def matched(L1, L2):
    L1 = np.asarray(L1); L2 = np.asarray(L2); out = {}
    for q in QS:
        k = np.where(L2 <= q * L2[0])[0]
        out[str(q)] = None if len(k) == 0 else dict(step=int(k[0]), G=float(L1[:k[0] + 1].max() - L1[0]))
    return out


def run(th0, T, rho, method, steps, c=None, lr=None, delta=None):
    t1, t2 = T[0], T[1]; th = th0.clone()
    _, G1 = grad_and_G(th, t1, True); _, G2 = grad_and_G(th, t2, True)
    lmax0 = torch.linalg.eigvalsh((1 - rho) * G2 + rho * G1)[-1].item()
    if method == "gd": lr = c / lmax0
    m = torch.zeros_like(th); v = torch.zeros_like(th)
    L1 = [loss(th, t1)]; L2 = [loss(th, t2)]
    for s in range(1, steps + 1):
        if method == "ggn":
            g1, G1 = grad_and_G(th, t1, True); g2, G2 = grad_and_G(th, t2, True)
        else:
            g1 = grad(th, t1); g2 = grad(th, t2)
        g = (1 - rho) * g2 + rho * g1
        if method == "gd": th = th - lr * g
        elif method == "adam":
            m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g
            th = th - lr * (m / (1 - 0.9 ** s)) / ((v / (1 - 0.999 ** s)).sqrt() + 1e-8)
        else:
            Gr = (1 - rho) * G2 + rho * G1
            th = th - lr * torch.linalg.solve(Gr + delta * lmax0 * torch.eye(len(th)), g)
        L1.append(loss(th, t1)); L2.append(loss(th, t2))
    L1 = np.array(L1); L2 = np.array(L2)
    # store a log-spaced subsample of the curve for plotting
    idx = np.unique(np.concatenate([np.arange(min(60, len(L1))), np.geomspace(1, len(L1) - 1, 300).astype(int)]))
    return dict(rho=rho, method=method, c=c, lr=lr, delta=delta, steps=steps, G=float(L1.max() - L1[0]),
                t_peak=int(L1.argmax()), L1_0=float(L1[0]), L2_0=float(L2[0]), L2_final=float(L2[-1]),
                matched=matched(L1, L2), idx=idx.tolist(), L1c=L1[idx].tolist(), L2c=L2[idx].tolist())


def linearised(th0, T, rho, eta, steps):
    """GD on the network linearised at theta_0; returns L1, L2 trajectories."""
    out = []
    for t in T[:2]:
        f0, _ = fwd(th0, t[0]); N = t[0].shape[0]
        out.append(((f0 - onehot(t[1])).reshape(-1), jac(th0, t[0]).reshape(-1, len(th0)), N))
    (r1, J1, N1), (r2, J2, N2) = out
    D = torch.zeros_like(th0); L1 = []; L2 = []
    for s in range(steps + 1):
        a1 = r1 + J1 @ D; a2 = r2 + J2 @ D
        L1.append(0.5 * (a1 @ a1).item() / N1); L2.append(0.5 * (a2 @ a2).item() / N2)
        if s < steps:
            D = D - eta * ((1 - rho) * (J2.T @ a2) / N2 + rho * (J1.T @ a1) / N1)
    return np.array(L1), np.array(L2)


def gd_fixed(th0, T, rho, eta, steps):
    th = th0.clone(); L1 = [loss(th, T[0])]
    for _ in range(steps):
        th = th - eta * ((1 - rho) * grad(th, T[1]) + rho * grad(th, T[0])); L1.append(loss(th, T[0]))
    return np.array(L1)


if __name__ == "__main__":
    scen, seed = sys.argv[1], int(sys.argv[2]); t0 = time.time()
    T = data(scen, seed); th0 = phase1(seed, T[0])
    out = dict(scenario=scen, seed=seed, runs=[], short=[], lin=[])
    # (2)+(3a) short-time multi-step test with a common step
    _, G1 = grad_and_G(th0, T[0], True); _, G2 = grad_and_G(th0, T[1], True)
    l1max = torch.linalg.eigvalsh(G1)[-1].item(); l2max = torch.linalg.eigvalsh(G2)[-1].item()
    eta_c = 1.0 / max(l1max, l2max)
    for rho in [0.0, 0.01, 0.1, 0.3, 0.5]:
        L1 = gd_fixed(th0, T, rho, eta_c, 30); Ll, _ = linearised(th0, T, rho, eta_c, 30)
        out["short"].append(dict(rho=rho, eta=eta_c, dL1=(L1 - L1[0]).tolist(), dL1_lin=(Ll - Ll[0]).tolist()))
    print(scen, seed, "short done", f"{time.time()-t0:.0f}s", flush=True)
    # (1) matched comparison
    for rho in [0.1, 0.3]:
        for method, kw, steps in [("gd", dict(c=1.0), 4000), ("gd", dict(c=0.3), 8000), ("gd", dict(c=0.1), 12000),
                                  ("adam", dict(lr=1e-3), 4000), ("adam", dict(lr=3e-4), 8000),
                                  ("ggn", dict(lr=0.2, delta=1e-3), 150)]:
            out["runs"].append(run(th0, T, rho, method, steps, **kw))
        print(scen, seed, rho, f"{time.time()-t0:.0f}s", flush=True)
    # (3b) linearised vs real GD (c=1) for the full budget
    for rho in [0.01, 0.1, 0.3]:
        _, G1 = grad_and_G(th0, T[0], True); _, G2 = grad_and_G(th0, T[1], True)
        eta = 1.0 / torch.linalg.eigvalsh((1 - rho) * G2 + rho * G1)[-1].item()
        prev = [r for r in out["runs"] if r["rho"] == rho and r["method"] == "gd" and r["c"] == 1.0]
        if prev:   # reuse the identical GD (c=1) run from part (1)
            idx = np.array(prev[0]["idx"]); L1 = np.full(4001, np.nan); L1[idx] = prev[0]["L1c"]
        else:
            L1 = gd_fixed(th0, T, rho, eta, 4000)
            idx = np.unique(np.concatenate([np.arange(60), np.geomspace(1, 4000, 300).astype(int)]))
        Ll, L2l = linearised(th0, T, rho, eta, 4000)
        out["lin"].append(dict(rho=rho, eta=eta, G=float(np.nanmax(L1) - L1[0]), t_peak=int(np.nanargmax(L1)),
                               G_lin=float(Ll.max() - Ll[0]), t_peak_lin=int(Ll.argmax()),
                               idx=idx.tolist(), L1c=L1[idx].tolist(), L1c_lin=Ll[idx].tolist()))
    print(scen, seed, "lin done", f"{time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(os.path.join(R, f"mlpm_{scen}_{seed}.json"), "w"))
