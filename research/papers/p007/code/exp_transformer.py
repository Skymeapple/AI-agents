"""Small decoder-only transformers on real token streams: unified vs specialists.

Each modality's stream (prep_tokens.py) is cut to its first D training tokens.
A training example is a window of L=64 tokens preceded by that modality's tag
token.  Every model sees, per optimisation step, B_M=16 windows of EACH of its
modalities, so unified and specialist models see the same per-modality tokens
for the same number of steps (the same-size, same-data convention of
Aghajanyan et al. 2023).  All models share one vocabulary (144 tokens) and the
same architecture; only the data they are trained on differ.  Early stopping on
a validation half of the held-out stream; we report cross-entropy (nats/token)
on the other (test) half, both at the checkpoint with the best MEAN validation
loss ('test') and at each modality's own best-validation checkpoint
('test_per_mod_stop'; removes the confound that modalities overfit at
different rates, Wang et al. 2020).  Training stops when no modality has
improved for PATIENCE evaluations.

Usage:
  python3 exp_transformer.py all [part nparts]
     grid  : P1, 4 sizes x 2 budgets, M=2 (image+audio) + both specialists
     budget: P4, add text and fashion at d_model=32, D=25k
     (we ran 3 parallel workers: parts 0,1,2 of 3)
Results are appended (one JSON line per trained model) to
../results/transformer_runs.jsonl ; analyse with analyze_transformer.py
"""
import json, os, sys, time, math, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(os.environ.get("TT_THREADS", "1")))
HERE = os.path.dirname(os.path.abspath(__file__))
DC = os.path.join(HERE, "..", "data_cache")
OUT = os.path.join(HERE, "..", "results", "transformer_runs.jsonl")
T = np.load(os.path.join(DC, "tokens.npz"))
V = int(T["vocab_size"])
L, BM, MAXSTEPS, EVAL_EVERY, PATIENCE, LR = 64, 16, 3000, 100, 6, 2e-3
NEVAL = 160  # held-out windows per modality per split


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.register_buffer("mask", torch.triu(torch.ones(L + 1, L + 1, dtype=torch.bool), 1))

    def forward(self, x):
        t = x.shape[1]
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=self.mask[:t, :t], need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, d, nl, nh):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, d), nn.Parameter(torch.zeros(1, L + 1, d))
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(nl)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, V)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, idx):
        x = self.emb(idx) + self.pos[:, : idx.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))


SIZES = {16: (2, 2), 32: (2, 2), 64: (2, 4), 128: (2, 4)}  # d_model: (layers, heads)


def windows(stream, tag, starts):
    x = np.stack([stream[s: s + L + 1] for s in starts])
    x = np.concatenate([np.full((len(starts), 1), tag), x], 1)
    return torch.from_numpy(x)


def heldout(mod):
    v = T[mod + "_val"]
    half = len(v) // 2
    out = {}
    for split, seg in [("val", v[:half]), ("test", v[half:])]:
        n = min(NEVAL, (len(seg) - 1) // (L + 1))
        starts = np.arange(n) * (L + 1)
        out[split] = windows(seg, int(T[mod + "_tag"]), starts)
    return out


@torch.no_grad()
def evaluate(model, data):
    model.eval()
    xb = data
    logits = model(xb[:, :-1])
    # predict tokens 1..L+1 of the window from tag + prefix; skip predicting the tag
    loss = F.cross_entropy(logits.reshape(-1, V), xb[:, 1:].reshape(-1))
    model.train()
    return loss.item()


def train(mods, D, dm, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    nl, nh = SIZES[dm]
    model = GPT(dm, nl, nh)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0, betas=(0.9, 0.98))
    streams = {m: T[m + "_train"][:D] for m in mods}
    assert all(len(s) >= D for s in streams.values()), "budget exceeds stream"
    ho = {m: heldout(m) for m in mods}
    best, best_step, best_test, best_val = math.inf, 0, None, None
    pm = {m: dict(val=math.inf, test=None, step=0) for m in mods}  # per-modality best
    last_improve = 0
    t0 = time.time()
    for step in range(1, MAXSTEPS + 1):
        lr = LR * min(1.0, step / 200)
        for g in opt.param_groups:
            g["lr"] = lr
        xb = torch.cat([windows(streams[m], int(T[m + "_tag"]),
                                rng.integers(0, D - L - 1, BM)) for m in mods])
        logits = model(xb[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, V), xb[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % EVAL_EVERY == 0:
            vl = {m: evaluate(model, ho[m]["val"]) for m in mods}
            tl = {m: evaluate(model, ho[m]["test"]) for m in mods}
            v = float(np.mean(list(vl.values())))
            if v < best - 1e-4:
                best, best_step, best_test, best_val = v, step, tl, vl
                last_improve = step
            for m in mods:
                if vl[m] < pm[m]["val"] - 1e-4:
                    pm[m] = dict(val=vl[m], test=tl[m], step=step); last_improve = step
            if step - last_improve >= PATIENCE * EVAL_EVERY:
                break
    return dict(mods=list(mods), D=D, d_model=dm, seed=seed, n_params=nparam,
                best_step=best_step, steps_run=step, val=best_val, test=best_test,
                test_per_mod_stop={m: pm[m]["test"] for m in mods},
                step_per_mod_stop={m: pm[m]["step"] for m in mods},
                secs=round(time.time() - t0, 1))


def done_keys():
    keys = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            r = json.loads(line)
            keys.add((tuple(r["mods"]), r["D"], r["d_model"], r["seed"]))
    return keys


def jobs(which):
    """'all': seed-major order so that partial completion still yields whole seeds."""
    J = []
    order = ["image", "audio", "text", "fashion"]
    for seed in [0, 1]:
        if which in ("grid", "all"):
            for i, (D, dm) in enumerate(itertools.product([25000, 100000], [16, 32, 64, 128])):
                trip = [(("image", "audio"), D, dm, seed), (("image",), D, dm, seed), (("audio",), D, dm, seed)]
                J += trip[i % 3:] + trip[:i % 3]  # rotate so parallel workers share the slower unified runs
        if which in ("budget", "all"):
            # P4: add modalities at fixed size (d_model=32, D=25k); M=1,2 runs are shared with the grid
            J += [(tuple(order[:3]), 25000, 32, seed), (tuple(order[:4]), 25000, 32, seed),
                  (("text",), 25000, 32, seed), (("fashion",), 25000, 32, seed)]
    return J


if __name__ == "__main__":
    which = sys.argv[1]
    part, nparts = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    J = jobs(which)[part::nparts]
    if len(sys.argv) > 4 and sys.argv[4] == "rev":  # extra helper worker: same jobs, reverse order
        J = jobs(which)[::-1]
    for mods, D, dm, seed in J:
        if (tuple(mods), D, dm, seed) in done_keys():
            continue
        r = train(mods, D, dm, seed)
        with open(OUT, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps(r), flush=True)
