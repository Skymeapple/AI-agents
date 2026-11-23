"""R2 nonlinear test of the first-order speed limit (pre-specified before running):
  prediction S1: recovery time of GD scales like rho^-1;
  prediction S2: heavy-ball momentum with FIXED beta=0.9 keeps the exponent -1 (prefactor ~(1-beta) smaller);
  prediction S3: heavy-ball with beta(rho) = 1 - sqrt(rho) (momentum tuned to the replay time scale, as the
                 Chebyshev/heavy-ball optimum on [a, L] with a ~ rho prescribes) changes the exponent to about -1/2;
  prediction S4: at matched new-task progress, momentum does not reduce the depth of the gap.
Network: 64-HID-10 tanh MLP, squared loss on one-hot targets, sklearn digits, N_PER samples per task so that
the joint problem is over-parameterised (joint training can drive both losses near zero).
Recovery time T = last step at which (L1 - L1(0) >= EPS * F_ref  or  L2 >= EPS * L2(0)), F_ref = peak excess old-task
loss of GD without replay (rho=0) in the same (scenario, seed). Censored at KMAX steps.
Usage: python mlp_speedlimit.py <scenario> <seed>  ->  ../results/mlps_<scen>_<seed>.json
"""
import torch, numpy as np, json, os, sys, time
from sklearn.datasets import load_digits
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
HID = 64; N_PER = 200; EPS = 0.1; KMAX = int(os.environ.get("KMAX", 300000))
RHOS = [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]


def data(scenario, seed):
    d = load_digits(); X = d.data / 16.0; y = d.target
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y)); X, y = X[idx], y[idx]
    mu = X.mean(0); sd = X.std(0) + 0.1; X = (X - mu) / sd; T = torch.tensor
    if scenario == "perm":
        perm = np.random.default_rng(1000 + seed).permutation(64)
        t1 = (T(X[:N_PER]), T(y[:N_PER])); t2 = (T(X[N_PER:2 * N_PER][:, perm]), T(y[N_PER:2 * N_PER]))
    else:
        m = y < 5; t1 = (T(X[m][:N_PER]), T(y[m][:N_PER])); t2 = (T(X[~m][:N_PER]), T(y[~m][:N_PER]))
    return t1, t2


def init(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.cat([(torch.randn(HID, 64, generator=g) / 8.0).reshape(-1), torch.zeros(HID),
                      (torch.randn(10, HID, generator=g) / np.sqrt(HID)).reshape(-1), torch.zeros(10)])


def unpack(th):
    i = 0; out = []
    for shp in [(HID, 64), (HID,), (10, HID), (10,)]:
        n = int(np.prod(shp)); out.append(th[i:i + n].view(shp)); i += n
    return out


def loss_grad(th, t, Y):
    W1, b1, W2, b2 = unpack(th); X = t[0]; N = X.shape[0]
    h = torch.tanh(X @ W1.T + b1); r = h @ W2.T + b2 - Y
    dpre = (r @ W2) * (1 - h ** 2)
    g = torch.cat([(dpre.T @ X).reshape(-1) / N, dpre.mean(0), (r.T @ h).reshape(-1) / N, r.mean(0)])
    return 0.5 * (r * r).sum().item() / N, g


def jac(th, X):
    W1, b1, W2, b2 = unpack(th); N = X.shape[0]
    h = torch.tanh(X @ W1.T + b1); D = 1 - h ** 2; A = W2.unsqueeze(0) * D.unsqueeze(1)
    JW1 = (A.unsqueeze(3) * X.unsqueeze(1).unsqueeze(2)).reshape(N, 10, -1)
    I = torch.eye(10).expand(N, 10, 10)
    JW2 = (I.unsqueeze(3) * h.unsqueeze(1).unsqueeze(2)).reshape(N, 10, -1)
    return torch.cat([JW1, A, JW2, I], 2).reshape(N * 10, -1)


def lmax_ggn(th, t1, t2, rho):
    J1 = jac(th, t1[0]); J2 = jac(th, t2[0])
    Jc = torch.cat([np.sqrt(rho / t1[0].shape[0]) * J1, np.sqrt((1 - rho) / t2[0].shape[0]) * J2])
    return torch.linalg.eigvalsh(Jc @ Jc.T)[-1].item()


def phase1(seed, t1, Y1, steps=30000):
    th = init(seed); lr = 1.0 / lmax_ggn(th, t1, t1, 1.0); v = torch.zeros_like(th)
    for _ in range(steps):                       # heavy ball 0.9 just to reach a deep task-1 minimum quickly
        _, g = loss_grad(th, t1, Y1); v = 0.9 * v - lr * g; th = th + v
    return th


def phase2(th0, T, Y, rho, beta, lr, Fref, kmax):
    t1, t2 = T; Y1, Y2 = Y; th = th0.clone(); v = torch.zeros_like(th)
    L10, _ = loss_grad(th, t1, Y1); L20, _ = loss_grad(th, t2, Y2)
    rec_k = []; rec1 = []; rec2 = []; last_bad = 0; peak = 0.0; kpeak = 0
    matched = {}
    for k in range(kmax + 1):
        L1, g1 = loss_grad(th, t1, Y1); L2, g2 = loss_grad(th, t2, Y2)
        ex = L1 - L10
        if ex > peak: peak, kpeak = ex, k
        for q in (0.5, 0.25, 0.1):
            if str(q) not in matched and L2 <= q * L20: matched[str(q)] = dict(k=k, G=peak)
        if Fref is not None and (ex >= EPS * Fref or L2 >= EPS * L20): last_bad = k
        if k < 200 or k % 50 == 0: rec_k.append(k); rec1.append(ex); rec2.append(L2)
        if Fref is not None and k - last_bad > 2000 + 0.5 * last_bad and k > 1000:
            break                               # recovered and stayed recovered for a long stretch
        g = (1 - rho) * g2 + rho * g1; v = beta * v - lr * g; th = th + v
    censored = Fref is not None and (k - last_bad) <= 2000 + 0.5 * last_bad
    return dict(rho=rho, beta=beta, lr=lr, L10=L10, L20=L20, G=peak, k_peak=kpeak, T=(None if censored else last_bad + 1),
                censored=bool(censored), steps_run=k, matched=matched, final_ex=ex, final_L2=L2,
                k=rec_k, ex=rec1, L2c=rec2)


if __name__ == "__main__":
    scen, seed = sys.argv[1], int(sys.argv[2]); t0 = time.time()
    t1, t2 = data(scen, seed); Y1 = torch.nn.functional.one_hot(t1[1], 10).double(); Y2 = torch.nn.functional.one_hot(t2[1], 10).double()
    th0 = phase1(seed, t1, Y1)
    out = dict(scenario=scen, seed=seed, HID=HID, N_PER=N_PER, P=len(th0), runs=[])
    print(scen, seed, "phase1 L1=%.2e" % loss_grad(th0, t1, Y1)[0], f"{time.time()-t0:.0f}s", flush=True)
    lr0 = 1.0 / lmax_ggn(th0, t1, t2, 0.0)
    r0 = phase2(th0, (t1, t2), (Y1, Y2), 0.0, 0.0, lr0, None, 20000); Fref = r0["G"]
    r0.update(method="GD"); out["runs"].append(r0); out["Fref"] = Fref
    for rho in RHOS:
        lr = 1.0 / lmax_ggn(th0, t1, t2, rho)
        for method, beta in [("GD", 0.0), ("HB_0.9", 0.9), ("HB_tuned", 1 - np.sqrt(rho))]:
            r = phase2(th0, (t1, t2), (Y1, Y2), rho, beta, lr, Fref, KMAX); r.update(method=method)
            out["runs"].append(r)
            print(scen, seed, rho, method, "G=%.3f T=%s" % (r["G"], r["T"]), f"{time.time()-t0:.0f}s", flush=True)
            json.dump(out, open(os.path.join(R, f"mlps_{scen}_{seed}.json"), "w"))
