"""Shared utilities: a minimal pre-LN transformer in pure JAX, training loop,
linear probes, intrinsic-dimension estimation, power-law fitting, and I/O.

Kept dependency-light on purpose (jax, optax, numpy) so every experiment can be
read end to end without a framework in the way.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- I/O
def save_json(name: str, obj) -> Path:
    path = RESULTS / f"{name}.json"

    def default(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    path.write_text(json.dumps(obj, indent=1, default=default))
    return path


def load_json(name: str):
    return json.loads((RESULTS / f"{name}.json").read_text())


class Timer:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        self.elapsed = time.time() - self.t


# ----------------------------------------------------------------------------- transformer
def _dense(key, n_in, n_out, scale=None):
    scale = (1.0 / np.sqrt(n_in)) if scale is None else scale
    return {"w": jax.random.normal(key, (n_in, n_out)) * scale, "b": jnp.zeros((n_out,))}


def init_transformer(key, d_model, n_layers, n_heads, max_len, d_in=None, vocab=None,
                     d_out=None, d_mlp=None):
    """Either `vocab` (token inputs, learned embedding) or `d_in` (vector inputs,
    linear read-in). Output head size `d_out` defaults to `vocab`."""
    d_mlp = d_mlp or 4 * d_model
    d_out = d_out or vocab
    keys = jax.random.split(key, 4 + 4 * n_layers)
    p = {"pos": jax.random.normal(keys[0], (max_len, d_model)) * 0.02}
    if vocab is not None:
        p["embed"] = jax.random.normal(keys[1], (vocab, d_model)) * 0.02
    else:
        p["read_in"] = _dense(keys[1], d_in, d_model)
    p["ln_f"] = {"g": jnp.ones(d_model), "b": jnp.zeros(d_model)}
    p["head"] = _dense(keys[2], d_model, d_out, scale=0.02)
    layers = []
    for i in range(n_layers):
        k = keys[4 + 4 * i: 8 + 4 * i]
        layers.append({
            "ln1": {"g": jnp.ones(d_model), "b": jnp.zeros(d_model)},
            "qkv": _dense(k[0], d_model, 3 * d_model),
            "o": _dense(k[1], d_model, d_model, scale=0.02 / np.sqrt(2 * n_layers)),
            "ln2": {"g": jnp.ones(d_model), "b": jnp.zeros(d_model)},
            "fc1": _dense(k[2], d_model, d_mlp),
            "fc2": _dense(k[3], d_mlp, d_model, scale=0.02 / np.sqrt(2 * n_layers)),
        })
    p["layers"] = layers
    return p


def _ln(p, x):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + 1e-5) * p["g"] + p["b"]


def _attn(p, x, n_heads):
    B, T, D = x.shape
    qkv = x @ p["qkv"]["w"] + p["qkv"]["b"]
    q, k, v = jnp.split(qkv, 3, axis=-1)
    hd = D // n_heads
    q = q.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
    att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(hd)
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    att = jnp.where(mask, att, -1e9)
    att = jax.nn.softmax(att, axis=-1)
    out = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    return out @ p["o"]["w"] + p["o"]["b"]


def transformer_embed(params, inputs):
    if "embed" in params:
        h = params["embed"][inputs]
    else:
        h = inputs @ params["read_in"]["w"] + params["read_in"]["b"]
    return h + params["pos"][: h.shape[1]]


def transformer_from(params, h, n_heads, start_layer=0, return_resid=False):
    """Run layers start_layer.. on residual stream h. Returns logits (and the
    residual stream after each layer, if requested)."""
    resid = [h]
    for lp in params["layers"][start_layer:]:
        h = h + _attn(lp, _ln(lp["ln1"], h), n_heads)
        z = _ln(lp["ln2"], h)
        h = h + jax.nn.gelu(z @ lp["fc1"]["w"] + lp["fc1"]["b"]) @ lp["fc2"]["w"] + lp["fc2"]["b"]
        resid.append(h)
    logits = _ln(params["ln_f"], h) @ params["head"]["w"] + params["head"]["b"]
    return (logits, resid) if return_resid else logits


def transformer_apply(params, inputs, n_heads, return_resid=False):
    return transformer_from(params, transformer_embed(params, inputs), n_heads,
                            return_resid=return_resid)


def make_optimizer(lr, steps, wd=0.0, warmup=200, clip=1.0):
    sched = optax.warmup_cosine_decay_schedule(0.0, lr, min(warmup, steps // 10 + 1), steps, lr * 0.05)
    return optax.chain(optax.clip_by_global_norm(clip), optax.adamw(sched, weight_decay=wd))


def train(params, loss_fn, batch_fn, steps, lr=1e-3, wd=0.0, log_every=500, callback=None,
          warmup=200, seed=0):
    """Generic training loop. loss_fn(params, batch) -> scalar; batch_fn(step, rng) -> batch.
    callback(step, params) is invoked every log_every steps (and at the end)."""
    opt = make_optimizer(lr, steps, wd, warmup=warmup)
    state = opt.init(params)

    @jax.jit
    def step_fn(params, state, batch):
        loss, g = jax.value_and_grad(loss_fn)(params, batch)
        upd, state = opt.update(g, state, params)
        return optax.apply_updates(params, upd), state, loss

    rng = np.random.default_rng(seed)
    history = []
    for s in range(1, steps + 1):
        params, state, loss = step_fn(params, state, batch_fn(s, rng))
        if s % log_every == 0 or s == steps:
            history.append((s, float(loss)))
            if callback is not None:
                callback(s, params)
    return params, history


# ----------------------------------------------------------------------------- probes
def ridge_fit(X, Y, lam=1e-2):
    """Closed-form ridge with intercept. Returns (W, b)."""
    mu = X.mean(0)
    Xc = X - mu
    d = X.shape[1]
    W = np.linalg.solve(Xc.T @ Xc + lam * len(X) * np.eye(d), Xc.T @ (Y - Y.mean(0)))
    b = Y.mean(0) - mu @ W
    return W, b


def probe_classify(Xtr, ytr, Xte, yte, n_classes, l2=1e-4, steps=400, lr=0.05):
    """Linear probe: multinomial logistic regression (full-batch Adam on standardised
    inputs, small L2). Returns test accuracy and (W, b) in the original input space."""
    import optax as _optax
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xs = jnp.asarray((Xtr - mu) / sd, dtype=jnp.float32)
    y = jnp.asarray(ytr)
    params = {"W": jnp.zeros((Xtr.shape[1], n_classes)), "b": jnp.zeros(n_classes)}
    opt = _optax.adam(lr)
    st = opt.init(params)

    def loss(p):
        lp = jax.nn.log_softmax(Xs @ p["W"] + p["b"], -1)
        return -jnp.take_along_axis(lp, y[:, None], -1).mean() + l2 * (p["W"] ** 2).sum()

    @jax.jit
    def run(p, st):
        def body(c, _):
            p, st = c
            g = jax.grad(loss)(p)
            u, st = opt.update(g, st, p)
            return (_optax.apply_updates(p, u), st), None
        (p, st), _ = jax.lax.scan(body, (p, st), None, length=steps)
        return p

    params = run(params, st)
    W = np.asarray(params["W"]) / sd[:, None]
    b = np.asarray(params["b"]) - mu @ W
    return float(((Xte @ W + b).argmax(1) == yte).mean()), (W, b)


def probe_regress(Xtr, Ytr, Xte, Yte, lam=1e-3):
    """Linear probe for continuous targets. Returns mean R^2 over target columns."""
    W, b = ridge_fit(Xtr, Ytr, lam)
    P = Xte @ W + b
    ss_res = ((Yte - P) ** 2).sum(0)
    ss_tot = ((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-12
    return float(np.mean(1 - ss_res / ss_tot))


# ----------------------------------------------------------------------------- geometry
def twonn_id(X, discard=0.1, max_points=4000, seed=0):
    """TwoNN intrinsic-dimension estimator (Facco et al., 2017).
    Uses the ratio of second- to first-nearest-neighbour distances."""
    rng = np.random.default_rng(seed)
    if len(X) > max_points:
        X = X[rng.choice(len(X), max_points, replace=False)]
    X = np.unique(np.round(np.asarray(X, dtype=np.float64), 6), axis=0)
    sq = (X ** 2).sum(1)
    D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * X @ X.T, 0.0)
    np.fill_diagonal(D2, np.inf)
    part = np.partition(D2, 2, axis=1)[:, :2]
    r1, r2 = np.sqrt(part[:, 0]), np.sqrt(part[:, 1])
    ok = r1 > 1e-12
    mu = np.sort(r2[ok] / r1[ok])
    n = len(mu)
    keep = int(n * (1 - discard))
    F = np.arange(1, n + 1) / n
    x = np.log(mu[:keep])
    y = -np.log(1 - F[:keep])
    return float((x @ y) / (x @ x))


def powerlaw_fit(x, y):
    """Least-squares fit of log y = c + slope * log x. Returns slope, intercept, R^2."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    (slope, c), *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ np.array([slope, c])
    r2 = 1 - ((ly - pred) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    return float(slope), float(c), float(r2)


def bootstrap_slope(x, y_runs, n_boot=2000, seed=0):
    """y_runs: array [n_points, n_seeds]. Bootstraps seeds to get a CI on the slope."""
    rng = np.random.default_rng(seed)
    y_runs = np.asarray(y_runs)
    slopes = []
    for _ in range(n_boot):
        idx = rng.integers(0, y_runs.shape[1], y_runs.shape[1])
        slopes.append(powerlaw_fit(x, y_runs[:, idx].mean(1))[0])
    return float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))
