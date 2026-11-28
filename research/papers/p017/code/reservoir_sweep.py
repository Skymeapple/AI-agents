"""Review M1: stress-test the suffix-ceiling criterion with long-memory UNTRAINED reservoirs.
Sweep spectral norm lam in {0.9, 0.99, 1.0 (scaled orthogonal W)} x input gain in {0.03,0.1,0.3,1}
(bias scaled with the input gain), d = 512, plus a leaky ESN (leak 0.1, lam 0.99).
For each: linear probe AND a 1-hidden-layer MLP probe, k_eff against the exact ceiling,
truncation test (restart k steps back), restart distance delta(k), and CRA with controls.

Usage: python reservoir_sweep.py --mode balanced2 --seed 0
Output: ../results/sweep/res_<mode>_s<seed>.json
"""
import argparse, json, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import run_experiment as R
from world import TorusWorld, exact_curve
from models import fit_readout


class Reservoir(nn.Module):
    def __init__(self, vocab, H, lam, in_scale, seed, orth=False, leak=1.0, radius=False):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        if orth:
            Q, _ = torch.linalg.qr(torch.randn(H, H, generator=g)); W = Q * lam
        else:
            W = torch.randn(H, H, generator=g)
            W = W / (torch.linalg.eigvals(W).abs().max() if radius else torch.linalg.matrix_norm(W, ord=2)) * lam
        self.register_buffer("W", W)
        self.register_buffer("U", torch.randn(vocab, H, generator=g) * in_scale)
        self.register_buffer("b", torch.randn(H, generator=g) * 0.2 * in_scale)
        self.H, self.leak = H, leak

    def step(self, h, x):
        return (1 - self.leak) * h + self.leak * torch.tanh(h @ self.W.T + self.U[x] + self.b)

    def forward(self, toks, h0=None):
        N, L = toks.shape
        h = torch.zeros(N, self.H) if h0 is None else h0
        hs = []
        for t in range(L):
            h = self.step(h, toks[:, t]); hs.append(h)
        return None, torch.stack(hs, 1)


def mlp_probe(Xp, yp, Xt, yt, n_cls, seed, epochs=40):
    torch.manual_seed(seed)
    mu, sd = Xp.mean(0), Xp.std(0) + 1e-6
    Zp, Zt = (Xp - mu) / sd, (Xt - mu) / sd
    net = nn.Sequential(nn.Linear(Zp.shape[1], 512), nn.ReLU(), nn.Linear(512, n_cls))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
    y = torch.as_tensor(yp)
    for ep in range(epochs):
        perm = torch.randperm(len(y))
        for i in range(0, len(y), 512):
            idx = perm[i:i + 512]
            loss = F.cross_entropy(net(Zp[idx]), y[idx]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float((net(Zt).argmax(1).numpy() == yt).mean())


class Wrap:  # adapter so run_experiment.rollout_test can drive the reservoir
    def __init__(self, m): self.m = m
    def step(self, h, x): return self.m.step(h, x)


# (lam, input gain, W type): 'norm' = Gaussian scaled to spectral NORM lam (provable lam-contraction;
# its spectral radius is only ~lam/2), 'radius' = Gaussian scaled to spectral RADIUS lam (standard ESN
# practice; not a 2-norm contraction), 'orth' = lam * orthogonal (radius = norm = lam).
CONFIGS = [(0.99, 1.0, "norm"), (0.99, 0.1, "radius"), (0.99, 1.0, "radius"),
           (0.99, 0.1, "orth"), (0.99, 0.3, "orth"), (0.99, 1.0, "orth"),
           (1.0, 0.1, "orth"), (1.0, 0.3, "orth"), (1.0, 1.0, "orth")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--mode"); ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    torch.set_num_threads(1)
    out_dir = os.path.join(R.RES, "sweep"); os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    w = TorusWorld(5, a.mode, a.seed)
    w.sample(14000, R.T, rng)
    pr, pr_pos, _, pr_ob = w.sample(6000, R.T, rng)
    te, te_pos, te_mv, te_ob = w.sample(2000, R.T, rng)
    ce = exact_curve(w, te_mv, te_ob, R.EVAL_T)
    ct = np.mean([exact_curve(w, te_mv, te_ob, [t], kmax=16) for t in R.TRUNC_T], 0)
    res = {"mode": a.mode, "seed": a.seed, "exact_curve": ce, "exact_curve_trunc_t": ct.tolist(), "configs": []}
    for (lam, s, wt) in CONFIGS:
        orth, leak = wt == "orth", 1.0
        t0 = time.time()
        m = Reservoir(w.vocab, 512, lam, s, a.seed, orth, leak, radius=(wt == "radius"))
        with torch.no_grad():
            Hp = torch.cat([m(torch.as_tensor(pr[i:i + 1000]))[1] for i in range(0, len(pr), 1000)])
            Ht = torch.cat([m(torch.as_tensor(te[i:i + 1000]))[1] for i in range(0, len(te), 1000)])
        Xp = R.hid_at(Hp, R.EVAL_T).reshape(-1, 512); Xt = R.hid_at(Ht, R.EVAL_T).reshape(-1, 512)
        yp = pr_pos[:, R.EVAL_T].reshape(-1); yt = te_pos[:, R.EVAL_T].reshape(-1)
        op = pr_ob[:, R.EVAL_T].reshape(-1); ot = te_ob[:, R.EVAL_T].reshape(-1)
        W, b = fit_readout(Xp, yp, w.S)
        pa = float(((Xt @ W.T + b).argmax(1).numpy() == yt).mean())
        pm = mlp_probe(Xp, yp, Xt, yt, w.S, a.seed)
        Wo, bo = fit_readout(Xp, op, w.C)
        oa = float(((Xt @ Wo.T + bo).argmax(1).numpy() == ot).mean())
        keff = lambda acc: next((k for k, x in enumerate(ce) if x >= acc - 1e-12), None)
        trunc = {}
        for k in R.KS:
            accs, ds = [], []
            for t in R.TRUNC_T:
                with torch.no_grad():
                    hk = m(torch.as_tensor(te[:, 2 * (t - k):2 * t]))[1][:, -1]
                hf = Ht[:, 2 * t - 1]
                accs.append(((hk @ W.T + b).argmax(1).numpy() == te_pos[:, t]).mean())
                ds.append((hf - hk).norm(dim=1).mean().item())
            trunc[k] = {"trunc_acc": float(np.mean(accs)), "acc_star": float(ct[k]), "delta": float(np.mean(ds))}
        ro = R.rollout_test(Wrap(m), "esn", w, te, te_pos, te_mv, Ht, W, Wo, bo, a.seed)
        rec = {"lam": lam, "in_scale": s, "wtype": wt, "orth": orth, "leak": leak, "obs_acc": oa,
               "probe_lin": pa, "probe_mlp": pm, "k_eff_lin": keff(pa), "k_eff_mlp": keff(pm),
               "trunc": trunc, "cra": {k: v["cf_disc"] for k, v in ro.items()}}
        res["configs"].append(rec)
        print(a.mode, a.seed, lam, s, wt, f"obs={oa:.3f} lin={pa:.3f}(k{keff(pa)}) mlp={pm:.3f}(k{keff(pm)})",
              "delta8=%.3g" % trunc[8]["delta"], {k: round(v, 2) for k, v in rec["cra"].items()}, f"{time.time()-t0:.0f}s", flush=True)
        json.dump(res, open(os.path.join(out_dir, f"res_{a.mode}_s{a.seed}.json"), "w"), indent=1)
