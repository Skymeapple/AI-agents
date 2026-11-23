"""E3 (revision r1): a direct quantitative test of Theorem 1 in trained single-head models.

Model (exactly the theorem's setting):
  * one softmax head; query q = W_q e(query token) (history-independent, assumption A1);
  * key side: content logit q.k_t/sqrt(d) with k_t = W_k e(x_t) plus a learned bias b[a_t]
    for every age a_t (an arbitrary learned positional score);
  * logits clamped to [-B, B] by s = B tanh(s_raw / B) (assumption A2 with a KNOWN B;
    B = inf means no clamp, and we then use the measured spread);
  * factorized token embedding e(k,v) = E_key[k] + E_val[v] (no mixing across positions);
  * token-local value path: output = sum_t alpha_t W_v e(x_t), then a linear readout.
Task: K = 32 keys, V = 32 values, composite write tokens (k, v), then query(k).
Training distribution COVERS the test structure: the queried key has R versions with
R - 1 ~ U{0..7} (or a different mixture, --p_single), latest age ~ U[0, N-1], gaps between
versions ~ U{8..40}; N ~ U{48..256}. All test lengths (64, 128, 256) are inside training.

Evaluation = the proof's chain. For a given N, Delta = 16 and latest-age A, let R_A = floor(A/Delta)+1
and p_j be the positions with ages A - (j-1) Delta.  H_j = filler + writes of the queried key
at p_1..p_j with DISTINCT values (filler values differ from them too).  For each instance we
record alpha_{p_j}(H_j) and correctness on every H_j; m = min_j alpha_{p_j}(H_j).
Theorem 1 (rewritten for the measured worst-case weight m > 1/2):
      R_A * log(m/(1-m)) <= S - log(N-1),
where S is the logit spread (max - min) over all positions of all H_j of that instance.
Hence the ceiling  m <= 1 / (1 + exp(-(S - log(N-1)) / R_A)).  With the clamp, S <= 2B.
Usage: python3 e3_single_head.py --B 4 --seed 0 [--p_single 0.125] [--steps 3000]
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, json, os, argparse, time, math

p = argparse.ArgumentParser()
p.add_argument("--B", type=float, default=float("inf")); p.add_argument("--seed", type=int, default=0)
p.add_argument("--steps", type=int, default=3000); p.add_argument("--p_single", type=float, default=0.125)
p.add_argument("--threads", type=int, default=1); p.add_argument("--n_eval", type=int, default=500)
p.add_argument("--out", default="e3_single_head.json")
# --pointer adds a selection loss -log alpha_{t*} (attention CE on the latest version) to the value CE,
# i.e. the head is trained to SELECT the current version with margin, the quantity Theorem 1 bounds.
p.add_argument("--pointer", action="store_true")
args = p.parse_args()
torch.set_num_threads(args.threads)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
K, V, D, NMAX = 32, 32, 64, 256
VOC = K * V + K


class SingleHead(nn.Module):
    def __init__(s, B):
        super().__init__(); s.B = B
        s.ek = nn.Embedding(K + 1, D); s.ev = nn.Embedding(V + 1, D); s.Wq = nn.Linear(D, D, bias=False); s.Wk = nn.Linear(D, D, bias=False)
        s.Wv = nn.Linear(D, D, bias=False); s.out = nn.Linear(D, V); s.bias = nn.Parameter(torch.zeros(NMAX))

    def emb(s, x):
        # factorized, token-local embedding: e(k, v) = E_key[k] + E_val[v]; query(k) = E_key[k] + E_val[V]
        isq = x >= K * V
        kk = torch.where(isq, x - K * V, x // V); vv = torch.where(isq, torch.full_like(x, V), x % V)
        return s.ek(kk) + s.ev(vv)

    def logits(s, x):
        e = s.emb(x); q = s.Wq(e[:, -1]); k = s.Wk(e[:, :-1]); N = k.shape[1]
        raw = torch.einsum("bd,btd->bt", q, k) / math.sqrt(D) + s.bias[:N].flip(0)[None]  # bias by age
        return raw if math.isinf(s.B) else s.B * torch.tanh(raw / s.B), e

    def forward(s, x, return_attn=False):
        sc, e = s.logits(x); a = sc.softmax(-1)
        o = s.out(torch.einsum("bt,btd->bd", a, s.Wv(e[:, :-1])))
        return (o, a, sc) if return_attn else o


def batch_train(bs, rng, p_single):
    N = int(rng.integers(48, NMAX + 1))
    q = rng.integers(0, K, size=bs)
    other = rng.integers(0, K - 1, size=(bs, N)); keys = other + (other >= q[:, None])
    vals = rng.integers(0, V, size=(bs, N)); y = np.empty(bs, int); tpos = np.empty(bs, int)
    for b in range(bs):
        a = int(rng.integers(0, N)); last = N - 1 - a; tpos[b] = last
        nprev = 0 if rng.random() < p_single else int(rng.integers(1, 8))
        pos = [last]
        for _ in range(nprev):
            nxt = pos[-1] - int(rng.integers(8, 41))
            if nxt < 0: break
            pos.append(nxt)
        keys[b, pos] = q[b]; y[b] = vals[b, last]
    x = np.concatenate([keys * V + vals, (K * V + q)[:, None]], 1)
    return torch.tensor(x), torch.tensor(y), torch.tensor(tpos)


def chain_instances(n, N, A, Delta, rng):
    """Returns tensors for H_1..H_R (R = A//Delta + 1) for n random instances."""
    R = A // Delta + 1
    q = rng.integers(0, K, size=n)
    other = rng.integers(0, K - 1, size=(n, N)); keys = other + (other >= q[:, None])
    u = np.stack([rng.choice(V, size=R + 1, replace=False) for _ in range(n)])  # R version values + 1 filler value pool
    vals = np.stack([rng.choice(np.setdiff1d(np.arange(V), u[i, :R]), size=N) for i in range(n)])
    pos = np.array([N - 1 - (A - j * Delta) for j in range(R)])               # p_1..p_R (positions)
    xs, ys = [], []
    for j in range(R):
        kk = keys.copy(); vv = vals.copy()
        kk[:, pos[:j + 1]] = q[:, None]; vv[:, pos[:j + 1]] = u[:, :j + 1]
        xs.append(np.concatenate([kk * V + vv, (K * V + q)[:, None]], 1)); ys.append(u[:, j])
    return [torch.tensor(x) for x in xs], [torch.tensor(y) for y in ys], pos


def eval_chain(m, N, A, Delta, rng, n):
    xs, ys, pos = chain_instances(n, N, A, Delta, rng); R = len(xs)
    alph, corr, smax, smin = [], [], [], []
    with torch.no_grad():
        for j in range(R):
            o, a, sc = m(xs[j], return_attn=True)
            alph.append(a[:, pos[j]].numpy()); corr.append((o.argmax(-1) == ys[j]).numpy())
            smax.append(sc.max(-1).values.numpy()); smin.append(sc.min(-1).values.numpy())
    alph = np.stack(alph, 1); corr = np.stack(corr, 1)
    S = np.max(np.stack(smax, 1), 1) - np.min(np.stack(smin, 1), 1)          # per-instance spread
    mmin = alph.min(1)
    ceil = 1.0 / (1.0 + np.exp(-(S - np.log(N - 1)) / R))
    ceil = np.where(S - np.log(N - 1) > 0, ceil, 0.5)                          # if S <= log(N-1): m <= 1/2
    viol = np.mean((mmin > 0.5) & (R * np.log(np.clip(mmin, 1e-12, 1 - 1e-12) / np.clip(1 - mmin, 1e-12, 1))
                                   > S - np.log(N - 1) + 1e-6))
    return dict(N=N, A=A, Delta=Delta, R=R, n=n,
                m_mean=float(mmin.mean()), ceil_mean=float(ceil.mean()), S_mean=float(S.mean()),
                frac_m_ge_095=float(np.mean(mmin >= 0.95)), bound_violations=float(viol),
                alpha_hot=float(alph[:, -1].mean()), alpha_cold=float(alph[:, 0].mean()),
                acc_hot=float(corr[:, -1].mean()), acc_cold=float(corr[:, 0].mean()),
                acc_all=float(corr.all(1).mean()), acc_each=corr.mean(0).tolist(),
                alpha_each=alph.mean(0).tolist())


def eval_age0(m, N, R, rng, n, spaced=False):
    """Diagnostic for reviewer F2: R versions, latest at age 0; earlier versions uniformly placed
    (spaced=False, as in E2) or evenly spaced at ages g, 2g, ..., (R-1)g with g = (N-1)//(R-1)."""
    q = rng.integers(0, K, size=n)
    other = rng.integers(0, K - 1, size=(n, N)); keys = other + (other >= q[:, None])
    vals = rng.integers(0, V, size=(n, N)); y = np.empty(n, int)
    for b in range(n):
        if R == 1: pos = np.array([N - 1])
        elif spaced:
            g = (N - 1) // (R - 1); pos = N - 1 - g * np.arange(R)[::-1]
        else: pos = np.concatenate([np.sort(rng.choice(N - 1, size=R - 1, replace=False)), [N - 1]])
        keys[b, pos] = q[b]; y[b] = vals[b, N - 1]
    x = torch.tensor(np.concatenate([keys * V + vals, (K * V + q)[:, None]], 1))
    with torch.no_grad():
        o, a, _ = m(x, return_attn=True)
        return dict(acc=float((o.argmax(-1) == torch.tensor(y)).float().mean()), alpha=float(a[:, -1].mean()))


if __name__ == "__main__":
    path = os.path.join(OUT, args.out)
    tag = f"{'ptr' if args.pointer else 'val'}_B{args.B}_ps{args.p_single}_s{args.seed}"
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    m = SingleHead(args.B); opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=args.steps, pct_start=0.1)
    t0 = time.time(); losses = []
    for step in range(args.steps):
        x, y, tp = batch_train(128, rng, args.p_single)
        o, a, sc = m(x, return_attn=True); loss = F.cross_entropy(o, y)
        if args.pointer: loss = loss + F.cross_entropy(sc, tp)
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
        losses.append(loss.item())
        if step % 1000 == 0: print(tag, step, round(float(np.mean(losses[-100:])), 3), f"{time.time()-t0:.0f}s", flush=True)
    m.eval(); erng = np.random.default_rng(10_000 + args.seed)
    with torch.no_grad():
        xi, yi, _ = batch_train(2000, erng, args.p_single); indist = float((m(xi).argmax(-1) == yi).float().mean())
    chain = [eval_chain(m, N, A, 16, erng, args.n_eval) for N in [64, 128, 256] for A in [0, 16, 32, 48, 64, 96, 128, 192] if A < N]
    age0 = {f"N{N}_R{R}": eval_age0(m, N, R, erng, args.n_eval) for N in [64, 128, 256] for R in [1, 2, 4, 8, 16, 32] if R < N}
    age0_spaced = {f"N{N}_R{R}": eval_age0(m, N, R, erng, args.n_eval, spaced=True) for N in [64, 128, 256] for R in [1, 2, 4, 8, 16, 32] if R < N}
    rec = dict(objective="pointer" if args.pointer else "value", B=args.B, p_single=args.p_single, seed=args.seed, steps=args.steps, indist_acc=indist,
               final_loss=float(np.mean(losses[-200:])), train_time_s=time.time() - t0, chain=chain, age0=age0, age0_spaced=age0_spaced,
               age_bias=m.bias.detach().numpy().tolist())
    allres = json.load(open(path)) if os.path.exists(path) else {}
    allres[tag] = rec; json.dump(allres, open(path, "w"), indent=1)
    torch.save(m.state_dict(), os.path.join(OUT, "models", f"e3_{tag}.pt"))
    print(tag, "indist", round(indist, 3), "time", round(time.time() - t0), flush=True)
    for c in chain: print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items() if k not in ("acc_each", "alpha_each")})
