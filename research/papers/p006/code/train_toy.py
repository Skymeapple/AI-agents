"""E2: Trained small models on a key-value *revision* stream task (CPU, PyTorch).
Stream of N write tokens (key k, value v) -> single composite token; then a query token
for key q. Target: the value of the MOST RECENT write of q.
Models: causal Transformers with RoPE (1,2,4 layers) and a single-layer delta-rule
fast-weight memory (DeltaNet-style). Evaluation controls the number of versions R of the
queried key and the stream length N (incl. lengths beyond training).
Usage: python3 train_toy.py [--steps 3000] [--seeds 0 1 2]
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, json, os, argparse, time, math
p = argparse.ArgumentParser(); p.add_argument("--steps", type=int, default=3000)
p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); p.add_argument("--threads", type=int, default=4); p.add_argument("--out", default="train_toy.json"); p.add_argument("--models", nargs="+", default=["tf1", "tf2", "tf4", "delta"])
args = p.parse_args()
torch.set_num_threads(args.threads)
OUT = os.path.join(os.path.dirname(__file__), "..", "results"); os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
K, V, D = 16, 16, 64
VOC = K * V + K  # write tokens then query tokens

def batch_train(bs, rng):
    N = int(rng.integers(16, 129))
    keys = rng.integers(0, K, size=(bs, N)); vals = rng.integers(0, V, size=(bs, N))
    toks = keys * V + vals
    qpos = rng.integers(0, N, size=bs)  # pick a written key to query
    q = keys[np.arange(bs), qpos]
    y = np.empty(bs, int)
    for b in range(bs):
        idx = np.nonzero(keys[b] == q[b])[0]; y[b] = vals[b, idx[-1]]
    x = np.concatenate([toks, (K * V + q)[:, None]], 1)
    return torch.tensor(x), torch.tensor(y)

def batch_eval(bs, N, R, rng, latest_age=None):
    """Queried key q written exactly R times at random positions (latest at chosen age if given);
    remaining positions are writes of other keys (uniform over the other K-1 keys)."""
    q = rng.integers(0, K, size=bs)
    other = rng.integers(0, K - 1, size=(bs, N)); other = other + (other >= q[:, None])
    keys = other.copy(); vals = rng.integers(0, V, size=(bs, N)); y = np.empty(bs, int)
    for b in range(bs):
        if latest_age is None: pos = np.sort(rng.choice(N, size=R, replace=False))
        else:
            last = N - 1 - latest_age
            pos = np.sort(np.concatenate([rng.choice(last, size=R - 1, replace=False), [last]])) if R > 1 else np.array([last])
        keys[b, pos] = q[b]; y[b] = vals[b, pos[-1]]
    x = np.concatenate([keys * V + vals, (K * V + q)[:, None]], 1)
    return torch.tensor(x), torch.tensor(y)

def rope(x):  # x: B,H,T,dh
    B, H, T, dh = x.shape; half = dh // 2
    freqs = 1.0 / (10000 ** (torch.arange(half) / half)); ang = torch.arange(T)[:, None] * freqs[None]
    c, s = ang.cos(), ang.sin(); x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1)

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h; s.qkv = nn.Linear(d, 3 * d); s.o = nn.Linear(d, d)
        s.n1 = nn.LayerNorm(d); s.n2 = nn.LayerNorm(d); s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(s, x, record=None):
        B, T, d = x.shape; q, k, v = s.qkv(s.n1(x)).view(B, T, 3, s.h, d // s.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q), rope(k)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(d // s.h)
        att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
        if record is not None: record.append(att[:, :, -1, :].detach())  # logits at query position
        y = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, T, d)
        x = x + s.o(y); return x + s.mlp(s.n2(x))

class TF(nn.Module):
    def __init__(s, L, d=D, h=4):
        super().__init__(); s.emb = nn.Embedding(VOC, d); s.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        s.nf = nn.LayerNorm(d); s.head = nn.Linear(d, V)
    def forward(s, x, record=None):
        z = s.emb(x)
        for b in s.blocks: z = b(z, record)
        return s.head(s.nf(z[:, -1]))

class Delta(nn.Module):
    """Single-layer delta-rule fast weights: S_t = S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T,
    read r = S_T q at the query token; multi-head (4 heads x 16 dims)."""
    def __init__(s, d=D, h=4):
        super().__init__(); s.h = h; s.dh = d // h; s.emb = nn.Embedding(VOC, d)
        s.qkv = nn.Linear(d, 3 * d); s.beta = nn.Linear(d, h); s.out = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, 4 * d), nn.GELU(), nn.Linear(4 * d, V))
    def forward(s, x, record=None):
        B, T = x.shape; e = s.emb(x); q, k, v = s.qkv(e).view(B, T, 3, s.h, s.dh).unbind(2)
        q = F.normalize(q, dim=-1); k = F.normalize(k, dim=-1); beta = torch.sigmoid(s.beta(e))
        S = torch.zeros(B, s.h, s.dh, s.dh)
        for t in range(T - 1):
            kt, vt, bt = k[:, t], v[:, t], beta[:, t, :, None]
            pred = torch.einsum("bhij,bhj->bhi", S, kt)
            S = S + torch.einsum("bhi,bhj->bhij", bt * (vt - pred), kt)
        r = torch.einsum("bhij,bhj->bhi", S, q[:, -1]).reshape(B, -1)
        return s.out(torch.cat([r, e[:, -1]], -1))

def make(name): return {"tf1": lambda: TF(1), "tf2": lambda: TF(2), "tf4": lambda: TF(4), "delta": lambda: Delta()}[name]()

EB = 300  # eval streams per cell
EVAL_N = [32, 128, 512]; EVAL_R = [1, 2, 4, 8, 16, 32, 64]
results = json.load(open(os.path.join(OUT, args.out))) if os.path.exists(os.path.join(OUT, args.out)) else {}
for name in args.models:
    for seed in args.seeds:
        tag = f"{name}_s{seed}"
        if tag in results: print("skip", tag); continue
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        m = make(name); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=args.steps, pct_start=0.1)
        t0 = time.time(); losses = []
        for step in range(args.steps):
            x, y = batch_train(128, rng); loss = F.cross_entropy(m(x), y)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
            losses.append(loss.item())
            if step % 500 == 0: print(tag, step, np.mean(losses[-100:]), f"{time.time()-t0:.0f}s", flush=True)
        m.eval(); erng = np.random.default_rng(1000 + seed); grid = {}
        with torch.no_grad():
            xi, yi = batch_train(1000, erng); indist = (m(xi).argmax(-1) == yi).float().mean().item()
            for N in EVAL_N:
                # R sweep with latest version at a FIXED age (decouples R from recency)
                for af in [0.0, 0.25]:
                    a = int(af * N)
                    for R in EVAL_R:
                        if R - 1 > N - 1 - a: continue
                        x, y = batch_eval(EB, N, R, erng, latest_age=a)
                        grid[f"N{N}_R{R}_fixage{af}"] = (m(x).argmax(-1) == y).float().mean().item()
                # cold vs hot: R=1 with old latest; R=8 with recent latest
                for age_frac in [0.5, 0.95]:
                    a = int(age_frac * (N - 1))
                    for R in [1]:
                        if R - 1 > N - 1 - a: continue
                        x, y = batch_eval(EB, N, R, erng, latest_age=a)
                        grid[f"N{N}_R{R}_age{age_frac}"] = (m(x).argmax(-1) == y).float().mean().item()
            # logit range at query position (transformer final layer), in-distribution N=128
            logit_range = None
            if name.startswith("tf"):
                rec = []; x, _ = batch_train(256, np.random.default_rng(7)); m(x, rec)
                last = rec[-1][..., :-1]  # B,H,N (exclude the query token itself)
                logit_range = (last.max(-1).values - last.min(-1).values).mean(0).tolist()  # per head
        torch.save(m.state_dict(), os.path.join(OUT, "models", tag + ".pt"))
        results[tag] = dict(model=name, seed=seed, steps=args.steps, final_train_loss=float(np.mean(losses[-200:])),
                            indist_acc=indist, grid=grid, logit_range_per_head=logit_range, train_time_s=time.time() - t0)
        print(tag, "indist", indist, {k: round(v, 3) for k, v in grid.items() if "fixage" in k}, flush=True)
        json.dump(results, open(os.path.join(OUT, args.out), "w"), indent=1)
