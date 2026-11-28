"""E2: partial mimicry (Theorem 5) with learned policies and exact worst-case evaluation.

Content c = 3 tokens from a 6-token vocabulary (216 possible contents).  Legitimate in-context instructions
draw tokens i.i.d. from p_L, the attacker's native injections from p_A (so content is informative), and an
out-of-band provenance flag is correct with probability q (q=0.5: no unforgeable signal).  The attacker is
eps-contaminated: with prob. 1-eps its content comes from A0, with prob. eps it writes any content it likes.

Policies are small MLPs over (token embeddings, flag) with deterministic decisions phi = 1[logit > 0].
Two training regimes:
  STATIC : attacks drawn from A0 only (the usual static benchmark training).
  ROBUST : adversarial training against the eps-contaminated attacker; the training-time adversary is a
           greedy coordinate search (4 random restarts x 2 sweeps), i.e. weaker than the evaluator.
Evaluation is exact (all 216 x 2 views enumerated with their probabilities), so U and the worst-case
ASR_eps = (1-eps) E_{A0 x K_A} phi + eps max_c E_{K_A} phi(c, .) carry no sampling error.
Predictions tested (not guaranteed by construction):
  (a) ROBUST models approach the Theorem-5 frontier ASR*_eps(U) and beat the s-only frontier f_K(U);
  (b) STATIC models do not: at eps > 0 their worst-case ASR exceeds f_K(U) (content reliance backfires).
"""
import json, time, itertools
import numpy as np, torch, torch.nn as nn
from multiprocessing import Pool
from verify_partial import f_K, formula_T5, lp_T5

M, K = 6, 3
pL = np.array([.30, .25, .20, .12, .08, .05]); pA = pL[::-1].copy()
CONTENTS = np.array(list(itertools.product(range(M), repeat=K)))          # (216, 3)
Lc = np.prod(pL[CONTENTS], 1); A0c = np.prod(pA[CONTENTS], 1)
QS = [0.5, 0.8]; EPS = [0.1, 0.25, 0.5]; WS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]; SEEDS = [0, 1, 2]
STEPS, B = 1500, 512


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(M, 8); s.pos = nn.Parameter(torch.randn(K, 8) * 0.1); s.fl = nn.Embedding(2, 8)
        s.mlp = nn.Sequential(nn.Linear(8 * K + 8, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(s, c, f):
        h = (s.emb(c) + s.pos).flatten(1)
        return s.mlp(torch.cat([h, s.fl(f)], 1)).squeeze(1)


def kernels(q):
    # flag=1 means "pipeline says trusted"; K_L(1)=q, K_A(1)=1-q
    return np.array([1 - q, q]), np.array([q, 1 - q])


def g_soft(net, cs, KA):
    """E_{K_A} sigma(logit(c, s)) for a batch of contents."""
    n = len(cs); c = torch.as_tensor(cs).repeat(2, 1); f = torch.arange(2).repeat_interleave(n)
    p = torch.sigmoid(net(c, f)).view(2, n)
    return (torch.as_tensor(KA, dtype=torch.float32)[:, None] * p).sum(0)


def greedy_adv(net, KA, rng, restarts=4, sweeps=2):
    with torch.no_grad():
        best, bestv = None, -1
        for _ in range(restarts):
            c = rng.integers(0, M, K)
            for _ in range(sweeps):
                for pos in range(K):
                    cand = np.repeat(c[None], M, 0); cand[:, pos] = np.arange(M)
                    v = g_soft(net, cand, KA).numpy(); c = cand[int(v.argmax())]
            v = float(g_soft(net, c[None], KA)[0])
            if v > bestv: best, bestv = c.copy(), v
    return best


def evaluate(net, q):
    KL, KA = kernels(q)
    with torch.no_grad():
        n = len(CONTENTS); c = torch.as_tensor(CONTENTS).repeat(2, 1); f = torch.arange(2).repeat_interleave(n)
        phi = (net(c, f) > 0).float().view(2, n).numpy()           # phi[s, c]
    U = float((Lc[None] * KL[:, None] * phi).sum())
    ASR0 = float((A0c[None] * KA[:, None] * phi).sum())
    gmax = float((KA[:, None] * phi).sum(0).max())
    return U, ASR0, gmax


def train(q, eps_train, w, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    KL, KA = kernels(q)
    net = Net(); opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    adv = rng.integers(0, M, K)
    for step in range(STEPS):
        if eps_train > 0 and step % 25 == 0:
            adv = greedy_adv(net, KA, rng)
        cl = torch.as_tensor(rng.choice(M, (B, K), p=pL)); fl = torch.as_tensor((rng.random(B) < KL[1]).astype(int))
        ca = torch.as_tensor(rng.choice(M, (B, K), p=pA)); fa = torch.as_tensor((rng.random(B) < KA[1]).astype(int))
        if eps_train > 0:
            m = torch.as_tensor(rng.random(B) < eps_train)
            ca[m] = torch.as_tensor(adv)
        loss = bce(net(cl, fl), torch.ones(B)).mean() + w * bce(net(ca, fa), torch.zeros(B)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def run(job):
    q, regime, eps_train, w, seed = job
    torch.set_num_threads(1)
    net = train(q, eps_train, w, seed)
    U, ASR0, gmax = evaluate(net, q)
    return dict(q=q, regime=regime, eps_train=eps_train, w=w, seed=seed, U=U, ASR_static=ASR0, gmax=gmax)


if __name__ == "__main__":
    t0 = time.time()
    jobs = [(q, "STATIC", 0.0, w, s) for q in QS for w in WS for s in SEEDS]
    jobs += [(q, "ROBUST", e, w, s) for q in QS for e in EPS for w in WS for s in SEEDS]
    with Pool(4) as pool:
        rows = pool.map(run, jobs)
    # theory curves and per-model gaps
    out = dict(config=dict(M=M, K=K, pL=pL.tolist(), pA=pA.tolist(), QS=QS, EPS=EPS, WS=WS, SEEDS=SEEDS, STEPS=STEPS,
                           TV_content=float(0.5 * np.abs(Lc - A0c).sum())), rows=[], curves={})
    for r in rows:
        KL, KA = kernels(r["q"])
        for e in EPS:
            if r["regime"] == "ROBUST" and r["eps_train"] != e: continue
            asr = (1 - e) * r["ASR_static"] + e * r["gmax"]
            fr = formula_T5(Lc, A0c, KL, KA, r["U"], e) if r["U"] > 0 else 0.0
            fk = f_K(KL, KA, r["U"])
            out["rows"].append(dict(r, eps_eval=e, ASR_eps=asr, frontier_T5=fr, frontier_sonly=fk,
                                    gap_T5=asr - fr, content_gain=fk - asr))
    for q in QS:
        KL, KA = kernels(q)
        us = np.linspace(0.02, 0.99, 50)
        out["curves"][str(q)] = dict(U=us.tolist(), sonly=[f_K(KL, KA, u) for u in us],
                                     **{f"eps{e}": [formula_T5(Lc, A0c, KL, KA, u, e) for u in us] for e in [0.0] + EPS})
    # spot-check formula against LP on the real instance
    out["formula_vs_lp_maxerr"] = max(abs(formula_T5(Lc, A0c, *kernels(q), u, e) - lp_T5(Lc, A0c, *kernels(q), u, e))
                                      for q in QS for e in EPS for u in [0.3, 0.6, 0.9])
    out["runtime_s"] = time.time() - t0
    json.dump(out, open("../results/e2_partial.json", "w"), indent=1)
    print("done", out["runtime_s"], "formula_vs_lp", out["formula_vs_lp_maxerr"])
