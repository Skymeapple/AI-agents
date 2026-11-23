"""Nonlinear test of the linear-theory predictions: 2-layer tanh MLP (64-16-10) on
sklearn digits, squared loss on one-hot targets (so the Gauss-Newton matrix is J^T J,
the direct analogue of Sigma_k in the linear theory).
Scenarios: 'perm'  domain-incremental (task 2 = fixed pixel permutation, disjoint samples)
           'split' class-incremental (classes 0-4, then 5-9; shared 10-way head)
Phase 1: full-batch GD on task 1.  Phase 2: full-batch minimisation of
(1-rho) L2 + rho L1  (full task-1 training set replayed with weight rho), with
  gd   : gradient descent, step 1/lambda_max(G_rho(theta_0))
  adam : Adam lr 1e-3
  ggn  : damped Gauss-Newton  theta <- theta - (G_rho + delta*lambda_max(G_rho(theta_0)) I)^{-1} grad
Usage: python mlp_experiments.py <scenario> <seed>   (writes ../results/mlp_<scen>_<seed>.json)
"""
import torch, numpy as np, json, os, sys, time
from sklearn.datasets import load_digits
torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
HID = 16
RHOS = [0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5]

def data(scenario, seed):
    d = load_digits(); X = d.data / 16.0; y = d.target
    rng = np.random.default_rng(seed); idx = rng.permutation(len(y)); X, y = X[idx], y[idx]
    ntr = 1200; Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    mu = Xtr.mean(0); sd = Xtr.std(0) + 0.1; Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    T = torch.tensor
    if scenario == "perm":
        perm = np.random.default_rng(1000 + seed).permutation(64); h = ntr // 2
        t1 = (T(Xtr[:h]), T(ytr[:h])); t2 = (T(Xtr[h:][:, perm]), T(ytr[h:]))
        te1 = (T(Xte), T(yte)); te2 = (T(Xte[:, perm]), T(yte))
    else:
        m1 = ytr < 5; n1 = yte < 5
        t1 = (T(Xtr[m1]), T(ytr[m1])); t2 = (T(Xtr[~m1]), T(ytr[~m1]))
        te1 = (T(Xte[n1]), T(yte[n1])); te2 = (T(Xte[~n1]), T(yte[~n1]))
    return t1, t2, te1, te2

def init(seed):
    g = torch.Generator().manual_seed(seed)
    W1 = torch.randn(HID, 64, generator=g) / 8.0; b1 = torch.zeros(HID)
    W2 = torch.randn(10, HID, generator=g) / np.sqrt(HID); b2 = torch.zeros(10)
    return torch.cat([W1.reshape(-1), b1, W2.reshape(-1), b2])

def unpack(th):
    i = 0; out = []
    for shp in [(HID, 64), (HID,), (10, HID), (10,)]:
        n = int(np.prod(shp)); out.append(th[i:i+n].view(shp)); i += n
    return out

def fwd(th, X):
    W1, b1, W2, b2 = unpack(th); h = torch.tanh(X @ W1.T + b1); return h @ W2.T + b2, h

def onehot(y): return torch.nn.functional.one_hot(y, 10).double()

def loss(th, t):
    f, _ = fwd(th, t[0]); return 0.5 * ((f - onehot(t[1])) ** 2).sum(1).mean().item()

def jac(th, X):
    """analytic Jacobian of outputs wrt theta: N x 10 x P"""
    W1, b1, W2, b2 = unpack(th); N = X.shape[0]
    h = torch.tanh(X @ W1.T + b1); D = 1 - h ** 2                      # N x H
    A = W2.unsqueeze(0) * D.unsqueeze(1)                                 # N x 10 x H (d f / d pre)
    JW1 = (A.unsqueeze(3) * X.unsqueeze(1).unsqueeze(2)).reshape(N, 10, -1)
    Jb1 = A
    I = torch.eye(10).expand(N, 10, 10)
    JW2 = (I.unsqueeze(3) * h.unsqueeze(1).unsqueeze(2)).reshape(N, 10, -1)
    return torch.cat([JW1, Jb1, JW2, I], 2)

def grad(th, t):
    """manual backprop of 0.5*mean ||f-y||^2"""
    W1, b1, W2, b2 = unpack(th); X = t[0]; N = X.shape[0]
    h = torch.tanh(X @ W1.T + b1); f = h @ W2.T + b2; r = f - onehot(t[1])
    dpre = (r @ W2) * (1 - h ** 2)
    return torch.cat([(dpre.T @ X).reshape(-1) / N, dpre.mean(0), (r.T @ h).reshape(-1) / N, r.mean(0)])

def grad_and_G(th, t, need_G):
    g = grad(th, t)
    if not need_G: return g, None
    J = jac(th, t[0]).reshape(-1, len(th)); N = t[0].shape[0]
    return g, (J.T @ J) / N

def acc(th, t):
    f, _ = fwd(th, t[0]); return (f.argmax(1) == t[1]).double().mean().item()

def phase1(seed, t1, steps=4000):
    th = init(seed)
    _, G = grad_and_G(th, t1, True); lr = 1.0 / torch.linalg.eigvalsh(G)[-1].item()
    for _ in range(steps):
        g, _ = grad_and_G(th, t1, False); th = th - lr * g
    return th

def phase2(th0, T, rho, method, steps, delta=None, lr=None):
    t1, t2, te1, te2 = T; th = th0.clone()
    g1, G1 = grad_and_G(th, t1, True); g2, G2 = grad_and_G(th, t2, True)
    Gr = (1 - rho) * G2 + rho * G1; lmax0 = torch.linalg.eigvalsh(Gr)[-1].item()
    if method == "gd": lr = 1.0 / lmax0
    if method == "adam":
        m = torch.zeros_like(th); v = torch.zeros_like(th); b1_, b2_, eps = 0.9, 0.999, 1e-8
    rec = dict(L1=[loss(th, t1)], L2=[loss(th, t2)], A1=[acc(th, te1)], A2=[acc(th, te2)])
    # first-step diagnostics (linear theory: dL1 ~ 0.5 eta^2 (1-rho)^2 g2^T G1 g2 + first-order term)
    rec["g1norm0"] = g1.norm().item(); rec["quad_pred_coef"] = 0.5 * ((1-rho)**2) * (g2 @ G1 @ g2).item()
    rec["lr"] = lr if lr is not None else float('nan'); rec["lmax0"] = lmax0
    for s in range(1, steps + 1):
        need = method == "ggn"
        g1, G1 = grad_and_G(th, t1, need); g2, G2 = grad_and_G(th, t2, need)
        g = (1 - rho) * g2 + rho * g1
        if method == "gd": th = th - lr * g
        elif method == "adam":
            m = b1_ * m + (1 - b1_) * g; v = b2_ * v + (1 - b2_) * g * g
            mh = m / (1 - b1_ ** s); vh = v / (1 - b2_ ** s); th = th - lr * mh / (vh.sqrt() + eps)
        else:
            Gr = (1 - rho) * G2 + rho * G1  # damping relative to lambda_max at phase-2 start
            th = th - lr * torch.linalg.solve(Gr + delta * lmax0 * torch.eye(len(th)), g)
        rec["L1"].append(loss(th, t1)); rec["L2"].append(loss(th, t2))
        if s % 10 == 0 or s < 50:
            rec.setdefault("acc_steps", []).append(s); rec["A1"].append(acc(th, te1)); rec["A2"].append(acc(th, te2))
    return rec

if __name__ == "__main__":
    scen, seed = sys.argv[1], int(sys.argv[2]); t0 = time.time()
    T = data(scen, seed); th0 = phase1(seed, T[0])
    out = dict(scenario=scen, seed=seed, base=dict(L1=loss(th0, T[0]), L2=loss(th0, T[1]),
               A1=acc(th0, T[2]), A2=acc(th0, T[3])), runs=[])
    print(scen, seed, "phase1", out["base"], f"{time.time()-t0:.0f}s", flush=True)
    for rho in RHOS:
        for method, kw, steps in [("gd", {}, 4000), ("adam", dict(lr=1e-3), 4000),
                                  ("ggn", dict(delta=1e-2, lr=1.0), 60), ("ggn", dict(delta=1e-3, lr=1.0), 60),
                                  ("ggn", dict(delta=1e-3, lr=0.2), 150)]:
            if method == "ggn" and rho not in (0.01, 0.1, 0.3): continue
            r = phase2(th0, T, rho, method, steps, **kw); r.update(rho=rho, method=method, kw=kw)
            out["runs"].append(r)
        print(scen, seed, rho, f"{time.time()-t0:.0f}s", flush=True)
    json.dump(out, open(os.path.join(R, f"mlp_{scen}_{seed}.json"), "w"))
