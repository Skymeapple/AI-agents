"""E1 -- Prediction, compression and emergent world models.

A transformer only ever sees a stream of tokens: egocentric actions
(Forward / Left / Right) and what the agent sees directly ahead (a cell colour,
or a wall). It is never told the agent's position or heading, nor that a grid
exists. Because colours repeat, single observations are ambiguous: predicting
the next observation well requires *inferring and tracking* a hidden state.

Questions:
  Q1 Compression: does next-token training approach the Bayes-optimal
     (exact HMM filter) code length, and beat surface n-gram statistics?
  Q2 Representation: is the hidden world state linearly decodable from the
     residual stream, beyond what an untrained network gives?
  Q3 Causality: if we edit the representation of the state, do predictions
     change as the world model implies?
  Q4 Robustness: does the model hold up when the action policy shifts?
  Q5 Dynamics: does the state representation emerge as excess loss falls?
"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

from .common import (RESULTS, Timer, init_transformer, probe_classify, probe_regress, save_json,
                     train, transformer_apply, transformer_embed, transformer_from, twonn_id)

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # N E S W
BOS, F, L, R = 0, 1, 2, 3
OBS0 = 4


class EgoGrid:
    def __init__(self, G=6, n_colors=3, n_obstacles=6, seed=0):
        rng = np.random.default_rng(seed)
        while True:
            free = np.ones((G, G), bool)
            cells = rng.choice(G * G, n_obstacles, replace=False)
            free.flat[cells] = False
            if self._connected(free):
                break
        self.G, self.free = G, free
        self.color = rng.integers(0, n_colors, (G, G))
        self.n_colors = n_colors
        self.n_obs = n_colors + 1  # + wall
        self.WALL = n_colors
        self.states = [(r, c, h) for r in range(G) for c in range(G) if free[r, c] for h in range(4)]
        self.index = {s: i for i, s in enumerate(self.states)}
        S = self.S = len(self.states)
        self.cells = sorted({(r, c) for r, c, _ in self.states})
        self.cell_of = np.array([self.cells.index((r, c)) for r, c, _ in self.states])
        self.heading_of = np.array([h for _, _, h in self.states])
        self.xy_of = np.array([(r, c) for r, c, _ in self.states], float)
        self.obs_of = np.array([self._obs(s) for s in self.states])
        self.trans = np.zeros((3, S), int)
        for i, (r, c, h) in enumerate(self.states):
            dr, dc = DIRS[h]
            nr, nc = r + dr, c + dc
            self.trans[0, i] = self.index[(nr, nc, h)] if self._free(nr, nc) else i
            self.trans[1, i] = self.index[(r, c, (h - 1) % 4)]
            self.trans[2, i] = self.index[(r, c, (h + 1) % 4)]
        self.vocab = OBS0 + self.n_obs
        self.O = np.eye(self.n_obs)[self.obs_of]  # [S, n_obs]
        self.M = np.zeros((3, S, S))
        for a in range(3):
            self.M[a, np.arange(S), self.trans[a]] = 1.0

    def _free(self, r, c):
        return 0 <= r < self.G and 0 <= c < self.G and self.free[r, c]

    def _obs(self, s):
        r, c, h = s
        dr, dc = DIRS[h]
        return self.color[r + dr, c + dc] if self._free(r + dr, c + dc) else self.WALL

    @staticmethod
    def _connected(free):
        G = free.shape[0]
        start = tuple(np.argwhere(free)[0])
        seen, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                n = (r + dr, c + dc)
                if 0 <= n[0] < G and 0 <= n[1] < G and free[n] and n not in seen:
                    seen.add(n)
                    stack.append(n)
        return len(seen) == free.sum()

    def sample(self, B, T, rng, p_forward=1 / 3):
        """Returns tokens [B, 2+2T], states after each action [B, T+1] (index 0 = start)."""
        s = rng.integers(0, self.S, B)
        toks = np.zeros((B, 2 + 2 * T), np.int32)
        states = np.zeros((B, T + 1), np.int32)
        toks[:, 0] = BOS
        toks[:, 1] = OBS0 + self.obs_of[s]
        states[:, 0] = s
        pa = np.array([p_forward, (1 - p_forward) / 2, (1 - p_forward) / 2])
        for t in range(1, T + 1):
            a = rng.choice(3, B, p=pa)
            s = self.trans[a, s]
            toks[:, 2 * t] = 1 + a
            toks[:, 2 * t + 1] = OBS0 + self.obs_of[s]
            states[:, t] = s
        return toks, states

    def bayes(self, toks):
        """Exact filtering. Returns predictive probability of each observed obs
        token [B, T+1] and the belief over states after each action/obs [B, T+1, S]
        (belief *before* seeing o_t, i.e. what is available when predicting o_t)."""
        B, Lseq = toks.shape
        T = (Lseq - 2) // 2
        b = np.full((B, self.S), 1.0 / self.S)
        p_obs = np.zeros((B, T + 1))
        prior = np.zeros((B, T + 1, self.S))
        for t in range(T + 1):
            if t > 0:
                a = toks[:, 2 * t] - 1
                b = np.einsum("bs,bst->bt", b, self.M[a])
            prior[:, t] = b
            o = toks[:, 2 * t + 1] - OBS0
            lik = self.O[:, o].T  # [B, S]
            p_obs[:, t] = (b * lik).sum(1)
            b = b * lik
            b /= b.sum(1, keepdims=True)
        return p_obs, prior


# ----------------------------------------------------------------------------- n-gram baseline
def ngram_nll(train_toks, test_toks, n, vocab, n_obs, k=0.05, min_count=3):
    """Backoff n-gram predicting observation tokens from the previous n-1 tokens.
    Returns mean NLL (nats) on obs tokens o_1..o_T of the test set."""
    def ctx_codes(toks, length):
        # code for the `length` tokens preceding each obs position (obs at odd indices >= 3)
        idx = np.arange(3, toks.shape[1], 2)
        codes = np.zeros((toks.shape[0], len(idx)), np.int64)
        for j in range(1, length + 1):
            pos = idx - j
            tok = np.where(pos >= 0, toks[:, np.clip(pos, 0, None)], vocab)  # pad symbol
            codes = codes * (vocab + 1) + tok
        return codes, toks[:, idx] - OBS0

    tables = []
    for length in range(n - 1, -1, -1):
        c, y = ctx_codes(train_toks, length)
        keys = c.ravel() * n_obs + y.ravel()
        uk, cnt = np.unique(keys, return_counts=True)
        tables.append((length, dict(zip(uk.tolist(), cnt.tolist()))))
    yt = None
    probs = None
    for length, table in tables:
        c, y = ctx_codes(test_toks, length)
        yt = y
        cflat = c.ravel()
        counts = np.array([[table.get(int(cc) * n_obs + o, 0) for o in range(n_obs)] for cc in cflat])
        tot = counts.sum(1)
        p = (counts + k) / (tot[:, None] + k * n_obs)
        p = p[np.arange(len(p)), y.ravel()]
        if probs is None:
            probs, have = p, tot >= min_count
        else:
            probs = np.where(have, probs, p)
            have = have | (tot >= min_count)
    return float(-np.log(probs).mean())


# ----------------------------------------------------------------------------- experiment
CFG = dict(G=6, n_colors=3, n_obstacles=6, world_seed=3, T=20, d_model=64, n_layers=4,
           n_heads=4, batch=64, steps=5000, lr=1e-3, wd=0.01)


def obs_nll(logits, toks, world):
    """Model NLL on obs tokens o_1..o_T (predicted from the action positions)."""
    logp = jax.nn.log_softmax(logits[:, :, OBS0:OBS0 + world.n_obs], -1)
    pos = np.arange(2, toks.shape[1], 2)  # action positions predict next obs
    tgt = toks[:, pos + 1] - OBS0
    lp = jnp.take_along_axis(logp[:, pos], tgt[..., None], -1)[..., 0]
    return -lp


def run(cfg=CFG, seed=0, quick=False):
    world = EgoGrid(cfg["G"], cfg["n_colors"], cfg["n_obstacles"], cfg["world_seed"])
    T = cfg["T"]
    rng = np.random.default_rng(1000 + seed)
    steps = 400 if quick else cfg["steps"]
    n_eval = 400 if quick else 3000
    ev_toks, ev_states = world.sample(n_eval, T, rng)
    ood_toks, ood_states = world.sample(n_eval, T, rng, p_forward=0.7)
    ev_p, ev_prior = world.bayes(ev_toks)
    ood_p, _ = world.bayes(ood_toks)
    bayes_nll = float(-np.log(ev_p[:, 1:]).mean())
    bayes_ood = float(-np.log(ood_p[:, 1:]).mean())
    uniform_nll = float(np.log(world.n_obs))

    # surface-statistics baselines
    tr_toks, _ = world.sample(20000 if quick else 200000, T, rng)
    ngram = {}
    for n in ([1, 3, 5] if quick else [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]):
        ngram[n] = ngram_nll(tr_toks, ev_toks, n, world.vocab, world.n_obs)
        print(f"  {n}-gram nll {ngram[n]:.4f}", flush=True)

    max_len = 2 + 2 * T
    params0 = init_transformer(jax.random.PRNGKey(seed), cfg["d_model"], cfg["n_layers"],
                               cfg["n_heads"], max_len, vocab=world.vocab)
    H = cfg["n_heads"]

    def loss_fn(p, toks):
        logits = transformer_apply(p, toks[:, :-1], H)
        logp = jax.nn.log_softmax(logits, -1)
        return -jnp.take_along_axis(logp, toks[:, 1:, None], -1).mean()

    def batch_fn(step, r):
        return world.sample(cfg["batch"], T, r)[0]

    fwd = jax.jit(lambda p, x: transformer_apply(p, x, H, return_resid=True))

    def evaluate(p, toks):
        out_l, out_r = [], []
        for i in range(0, len(toks), 500):
            lg, rs = fwd(p, toks[i:i + 500, :-1])
            out_l.append(np.asarray(obs_nll(lg, toks[i:i + 500], world)))
            out_r.append(np.stack([np.asarray(r) for r in rs]))
        return np.concatenate(out_l), np.concatenate(out_r, axis=1)  # [N,T], [L+1,N,len,d]

    pos = np.arange(2, max_len, 2)  # action positions; state after action t at states[:, t]
    tgt_state = ev_states[:, 1:]
    belief = ev_prior[:, 1:]  # [N, T, S]
    identified = belief.max(-1) > 0.99
    n_tr = int(0.7 * n_eval)

    def probe_all(resid, full=True, max_train=20000):
        """Probe each layer's residual at action positions for state, cell, heading, belief.
        full=False probes only the joint state (used for the training-dynamics curve)."""
        out = []
        sub = np.random.default_rng(0).permutation(n_tr * T)[:max_train]
        for li in range(resid.shape[0]):
            X = resid[li][:, pos]  # [N, T, d]
            Xtr = X[:n_tr].reshape(-1, X.shape[-1])[sub]
            Xte = X[n_tr:].reshape(-1, X.shape[-1])
            ytr, yte = tgt_state[:n_tr].ravel()[sub], tgt_state[n_tr:].ravel()
            idte = identified[n_tr:].ravel()
            acc_state, (W, b) = probe_classify(Xtr, ytr, Xte, yte, world.S)
            pred = (Xte @ W + b).argmax(1)
            acc_id = float((pred[idte] == yte[idte]).mean())
            d = dict(layer=li, acc_state=acc_state, acc_state_identified=acc_id)
            if full:
                d["acc_cell"], _ = probe_classify(Xtr, world.cell_of[ytr], Xte, world.cell_of[yte], len(world.cells))
                d["acc_heading"], _ = probe_classify(Xtr, world.heading_of[ytr], Xte, world.heading_of[yte], 4)
                d["r2_belief"] = probe_regress(Xtr, belief[:n_tr].reshape(-1, world.S)[sub],
                                               Xte, belief[n_tr:].reshape(-1, world.S))
            out.append(d)
        return out

    # Bayes-optimal "probe ceiling": accuracy of the MAP state under the exact belief
    bayes_map_acc = float((belief[n_tr:].argmax(-1) == tgt_state[n_tr:]).mean())

    # untrained control
    nll0, resid0 = evaluate(params0, ev_toks)
    probes_init = probe_all(resid0)

    dynamics = []
    probe_steps = set([250, 500, 1000, 2000, 3000, steps]) if not quick else {200, steps}

    def callback(step, p):
        if step in probe_steps:
            nll, rs = evaluate(p, ev_toks)
            pr = probe_all(rs, full=False, max_train=10000)
            best = max(pr, key=lambda d: d["acc_state"])
            dynamics.append(dict(step=step, excess_nll=float(nll.mean() - bayes_nll),
                                 best_layer=best["layer"], acc_state=best["acc_state"],
                                 acc_state_identified=best["acc_state_identified"]))
            print(f"  step {step}: excess {nll.mean() - bayes_nll:.4f}  probe {best['acc_state']:.3f}", flush=True)

    with Timer() as tm:
        params, hist = train(params0, loss_fn, batch_fn, steps, cfg["lr"], cfg["wd"],
                             log_every=250 if not quick else 100, callback=callback, seed=seed)
    nll, resid = evaluate(params, ev_toks)
    nll_ood, _ = evaluate(params, ood_toks)
    probes = probe_all(resid)
    model_nll = float(nll.mean())

    # per-timestep curves (information accumulates as the agent moves)
    per_t = dict(model=nll.mean(0), bayes=-np.log(ev_p[:, 1:]).mean(0),
                 belief_entropy=(-(belief * np.log(belief + 1e-12)).sum(-1)).mean(0))

    # ----- intrinsic dimension of residual stream per layer (identified positions)
    ids = []
    for li in range(resid.shape[0]):
        X = resid[li][:, pos][identified]
        ids.append(twonn_id(X, max_points=3000, seed=seed))

    # ----- geometry: do cell-mean representations reproduce the grid's layout?
    best_layer = max(probes, key=lambda d: d["acc_state"])["layer"]
    X = resid[best_layer][:, pos]
    cells = world.cell_of[tgt_state]
    cell_means = np.stack([X[(cells == c) & identified].mean(0) for c in range(len(world.cells))])
    grid_xy = np.array(world.cells, float)
    from scipy.stats import spearmanr
    iu = np.triu_indices(len(world.cells), 1)
    d_rep = np.linalg.norm(cell_means[:, None] - cell_means[None], axis=-1)[iu]
    d_grid = np.abs(grid_xy[:, None] - grid_xy[None]).sum(-1)[iu]
    rsa = float(spearmanr(d_rep, d_grid).correlation)
    Xc = cell_means - cell_means.mean(0)
    U, Sv, Vt = np.linalg.svd(Xc, full_matrices=False)
    pca2 = Xc @ Vt[:2].T

    # ----- causal intervention: swap class-mean of state A for state B
    interventions = []
    rng_i = np.random.default_rng(7 + seed)
    for li in range(1, cfg["n_layers"]):
        Xl = resid[li][:, pos]
        mu = np.zeros((world.S, Xl.shape[-1]))
        cnt = np.zeros(world.S)
        m = identified[:n_tr]
        for s in range(world.S):
            sel = (tgt_state[:n_tr] == s) & m
            cnt[s] = sel.sum()
            if cnt[s] > 0:
                mu[s] = Xl[:n_tr][sel].mean(0)
        trials = []
        test_idx = np.argwhere(identified[n_tr:])
        rng_i.shuffle(test_idx)
        for (i, t) in test_idx[:300 if not quick else 40]:
            n = n_tr + i
            A = tgt_state[n, t]
            cands = [s for s in range(world.S) if cnt[s] >= 5 and world.obs_of[s] != world.obs_of[A]]
            if cnt[A] < 5 or not cands:
                continue
            B = rng_i.choice(cands)
            delta = mu[B] - mu[A]
            rand = rng_i.normal(size=delta.shape)
            rand *= np.linalg.norm(delta) / np.linalg.norm(rand)
            trials.append((n, t, A, B, delta, rand))
        if not trials:
            continue
        seqs = ev_toks[[tr[0] for tr in trials], :-1]
        h = np.asarray(fwd_resid_at(params, seqs, H, li))
        res = {"patch": [], "random": [], "none": []}
        for kind in res:
            hp = h.copy()
            for j, (n, t, A, B, delta, rand) in enumerate(trials):
                if kind == "patch":
                    hp[j, pos[t]] += delta
                elif kind == "random":
                    hp[j, pos[t]] += rand
            logits = np.asarray(run_from(params, jnp.asarray(hp), H, li))
            for j, (n, t, A, B, delta, rand) in enumerate(trials):
                pr = jax.nn.softmax(logits[j, pos[t], OBS0:OBS0 + world.n_obs])
                res[kind].append((float(pr[world.obs_of[B]]), int(np.argmax(pr) == world.obs_of[B]),
                                  int(np.argmax(pr) == world.obs_of[A])))
        interventions.append({
            "layer": li, "n": len(trials),
            **{f"{k}_p_target": float(np.mean([r[0] for r in v])) for k, v in res.items()},
            **{f"{k}_acc_target": float(np.mean([r[1] for r in v])) for k, v in res.items()},
            **{f"{k}_acc_original": float(np.mean([r[2] for r in v])) for k, v in res.items()},
        })

    import pickle
    with open(RESULTS / f"e1_params_seed{seed}{'_quick' if quick else ''}.pkl", "wb") as fh:
        pickle.dump(jax.tree_util.tree_map(np.asarray, params), fh)
    out = dict(
        config=cfg, seed=seed, train_seconds=tm.elapsed, steps=steps, S=world.S,
        n_cells=len(world.cells), n_obs=world.n_obs,
        world=dict(free=world.free.astype(int), color=world.color),
        nll=dict(uniform=uniform_nll, bayes=bayes_nll, model=model_nll, ngram=ngram,
                 best_ngram=min(ngram.values()), model_untrained=float(nll0.mean()),
                 bayes_ood=bayes_ood, model_ood=float(nll_ood.mean())),
        frac_gap_closed=float((min(ngram.values()) - model_nll) / (min(ngram.values()) - bayes_nll)),
        probes=probes, probes_untrained=probes_init, bayes_map_acc=bayes_map_acc,
        frac_identified=float(identified.mean()), per_t={k: np.asarray(v) for k, v in per_t.items()},
        intrinsic_dim=ids, rsa_spearman=rsa, best_layer=best_layer,
        cell_pca=pca2, cell_xy=grid_xy, interventions=interventions, dynamics=dynamics,
        history=hist,
    )
    return out


def fwd_resid_at(params, toks, H, layer):
    _, rs = transformer_apply(params, toks, H, return_resid=True)
    return rs[layer]


def run_from(params, h, H, layer):
    return transformer_from(params, h, H, start_layer=layer)


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    seeds = [int(a) for a in sys.argv[1:] if a.isdigit()] or ([0] if quick else [0, 1, 2])
    for sd in seeds:
        print(f"E1 seed {sd}", flush=True)
        r = run(seed=sd, quick=quick)
        save_json(f"e1_seed{sd}" + ("_quick" if quick else ""), r)
        print({k: r["nll"][k] for k in ["uniform", "best_ngram", "model", "bayes", "model_ood", "bayes_ood"]}, flush=True)
