"""Nonlinear check: one-hidden-layer ReLU networks, shared trunk vs specialists.
Teacher: y_m = A_m relu(S^T x) + B_m relu(P_m^T x), k_s shared + k_p private
features. Student capacity = number of hidden units h (masked from hmax).
All h values (and all specialists) are trained in parallel as a batch.
Early stopping on a validation set; test loss is noiseless excess risk.
Run: python3 exp_mlp.py   (about 10-15 min on 4 CPU cores)"""
import json, os, time, itertools
import numpy as np
import torch

torch.set_num_threads(2)
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
d, q, ks, kp, SIG = 32, 8, 6, 6, 0.5
HS = [2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
HMAX = max(HS)
STEPS, LR, EVAL, BATCH = 2000, 3e-3, 50, 256


def teacher(M, gen):
    S = torch.randn(d, ks, generator=gen); S /= S.norm(dim=0)
    P = [torch.randn(d, kp, generator=gen) for _ in range(M)]
    P = [p / p.norm(dim=0) for p in P]
    A = [torch.randn(ks, q, generator=gen) / np.sqrt(ks + kp) for _ in range(M)]
    B = [torch.randn(kp, q, generator=gen) / np.sqrt(ks + kp) for _ in range(M)]
    def f(m, x):
        return torch.relu(x @ S) @ A[m] + torch.relu(x @ P[m]) @ B[m]
    return f


def data(M, n, f, gen):
    out = []
    for m in range(M):
        X = torch.randn(n, d, generator=gen); Y = f(m, X) + SIG * torch.randn(n, q, generator=gen)
        Xv = torch.randn(min(n // 4 + 64, 1000), d, generator=gen); Yv = f(m, Xv) + SIG * torch.randn(len(Xv), q, generator=gen)
        Xt = torch.randn(1500, d, generator=gen); Yt = f(m, Xt)  # noiseless test
        out.append((X, Y, Xv, Yv, Xt, Yt))
    return torch.stack([o[0] for o in out]), torch.stack([o[1] for o in out]), \
        torch.stack([o[2] for o in out]), torch.stack([o[3] for o in out]), \
        torch.stack([o[4] for o in out]), torch.stack([o[5] for o in out])


def run(M, n, seed):
    gen = torch.Generator().manual_seed(seed)
    f = teacher(M, gen)
    X, Y, Xv, Yv, Xt, Yt = data(M, n, f, gen)  # [M, n, d] etc.
    H = len(HS)
    mask = torch.zeros(H, HMAX)
    for i, h in enumerate(HS):
        mask[i, :h] = 1
    torch.manual_seed(seed + 1)
    # unified: trunk [H, d, HMAX] shared over modalities, heads [H, M, HMAX, q]
    # specialists: trunk [H, M, d, HMAX], heads [H, M, HMAX, q]
    params = {
        "uW": torch.randn(H, 1, d, HMAX) / np.sqrt(d), "ub": torch.zeros(H, 1, 1, HMAX),
        "uV": torch.randn(H, M, HMAX, q) * 0.1, "uc": torch.zeros(H, M, 1, q),
        "sW": torch.randn(H, M, d, HMAX) / np.sqrt(d), "sb": torch.zeros(H, M, 1, HMAX),
        "sV": torch.randn(H, M, HMAX, q) * 0.1, "sc": torch.zeros(H, M, 1, q),
    }
    for p in params.values():
        p.requires_grad_(True)
    opt = torch.optim.Adam(params.values(), lr=LR)
    mk = mask[:, None, None, :]

    def fwd(Xb, pre):
        W, b, V, c = (params[pre + k] for k in "WbVc")
        if pre == "u":  # one trunk shared by all modalities
            Hn = torch.relu(torch.matmul(Xb.reshape(1, -1, d), W[:, 0]) + b[:, 0])
            Hn = Hn.reshape(W.shape[0], Xb.shape[0], Xb.shape[1], -1)
        else:
            Hn = torch.relu(torch.matmul(Xb[None], W) + b)
        Hn = Hn * mk
        return torch.matmul(Hn, V) + c
    best = {"u": [np.inf] * H, "s": [[np.inf] * M for _ in range(H)]}
    test_at_best = {"u": [None] * H, "s": [[None] * M for _ in range(H)]}
    for step in range(STEPS + 1):
        if step % EVAL == 0:
            with torch.no_grad():
                for pre in "us":
                    vl = ((fwd(Xv, pre) - Yv) ** 2).mean(dim=(2, 3))  # [H, M]
                    tl = ((fwd(Xt, pre) - Yt) ** 2).mean(dim=(2, 3))
                    for i in range(H):
                        if pre == "u":
                            v = vl[i].mean().item()
                            if v < best["u"][i]:
                                best["u"][i] = v; test_at_best["u"][i] = tl[i].mean().item()
                        else:
                            for m in range(M):
                                v = vl[i, m].item()
                                if v < best["s"][i][m]:
                                    best["s"][i][m] = v; test_at_best["s"][i][m] = tl[i, m].item()
        if step == STEPS:
            break
        opt.zero_grad()
        if n > BATCH:
            idx = torch.randint(0, n, (BATCH,))
            Xb, Yb = X[:, idx], Y[:, idx]
        else:
            Xb, Yb = X, Y
        loss = ((fwd(Xb, "u") - Yb) ** 2).mean(dim=(2, 3)).sum() + \
               ((fwd(Xb, "s") - Yb) ** 2).mean(dim=(2, 3)).sum()
        loss.backward(); opt.step()
    Lu = np.array(test_at_best["u"])
    Ls = np.array([np.mean(t) for t in test_at_best["s"]])
    return Lu, Ls


if __name__ == "__main__":
    t0 = time.time()
    res = dict(d=d, q=q, ks=ks, kp=kp, sigma=SIG, hs=HS, steps=STEPS, runs=[])
    for n, M, seed in itertools.product([256, 1024, 4096], [2, 4, 8], [0, 1, 2]):
        Lu, Ls = run(M, n, 1000 * seed + 7 * M + n)
        res["runs"].append(dict(n=n, M=M, seed=seed, Lu=Lu.tolist(), Ls=Ls.tolist()))
        print(f"n={n} M={M} seed={seed} t={time.time()-t0:.0f}s ratio=",
              np.round(Lu / Ls, 2).tolist(), flush=True)
        json.dump(res, open(os.path.join(OUT, "mlp_results.json"), "w"), indent=1)
    print("total", time.time() - t0)
