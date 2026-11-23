"""Minimal pre-LN transformer in JAX; returns logits at last position and the
residual stream at the last position after the embedding and after every block."""
import jax, jax.numpy as jnp
import numpy as np
import optax
from world import VOCAB, SEQ, VAL0, V


def init_params(key, d=128, n_layers=4, n_heads=4, d_mlp=512):
    ks = jax.random.split(key, 4 + 6 * n_layers)
    s = 0.02
    p = {"emb": jax.random.normal(ks[0], (VOCAB, d)) * s,
         "pos": jax.random.normal(ks[1], (SEQ, d)) * s,
         "unemb": jax.random.normal(ks[2], (d, V)) * s,
         "lnf": jnp.ones(d), "blocks": []}
    for l in range(n_layers):
        k = ks[4 + 6 * l: 4 + 6 * (l + 1)]
        p["blocks"].append({
            "ln1": jnp.ones(d), "ln2": jnp.ones(d),
            "wqkv": jax.random.normal(k[0], (d, 3 * d)) * s,
            "wo": jax.random.normal(k[1], (d, d)) * s / np.sqrt(2 * n_layers),
            "w1": jax.random.normal(k[2], (d, d_mlp)) * s,
            "b1": jnp.zeros(d_mlp),
            "w2": jax.random.normal(k[3], (d_mlp, d)) * s / np.sqrt(2 * n_layers)})
    return p


def ln(x, g):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + 1e-5) * g


def forward(p, X, n_heads=4):
    """X: (B, SEQ) int. Returns logits (B, V) and resid (B, L+1, d) at last pos."""
    h = p["emb"][X] + p["pos"][None]
    B, T, d = h.shape
    hd = d // n_heads
    mask = jnp.tril(jnp.ones((T, T), bool))
    resid = [h[:, -1]]
    for blk in p["blocks"]:
        x = ln(h, blk["ln1"])
        qkv = x @ blk["wqkv"]
        q, k, v = jnp.split(qkv, 3, -1)
        q = q.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
        att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(hd)
        att = jnp.where(mask, att, -1e9)
        att = jax.nn.softmax(att, -1)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
        h = h + o @ blk["wo"]
        x = ln(h, blk["ln2"])
        h = h + jax.nn.gelu(x @ blk["w1"] + blk["b1"]) @ blk["w2"]
        resid.append(h[:, -1])
    logits = ln(h[:, -1], p["lnf"]) @ p["unemb"]
    return logits, jnp.stack(resid, 1)


def ce_loss(p, X, Y):
    logits, _ = forward(p, X)
    return optax.softmax_cross_entropy_with_integer_labels(logits, Y).mean()


def train(world, seed, steps=1500, bs=256, lr=1e-3, wd=0.01, log=None, d=64, d_mlp=256):
    key = jax.random.PRNGKey(seed)
    p = init_params(key, d=d, d_mlp=d_mlp)
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0, lr, 200, steps, lr * 0.05),
                      weight_decay=wd)
    st = opt.init(p)

    @jax.jit
    def step(p, st, X, Y):
        l, g = jax.value_and_grad(ce_loss)(p, X, Y)
        up, st = opt.update(g, st, p)
        return optax.apply_updates(p, up), st, l

    rng = np.random.default_rng(seed + 1000)
    if getattr(world, "pool", None) is None:
        world.build_pool(60000, rng)
    for t in range(steps):
        X, Y = world.batch(bs, rng)
        p, st, l = step(p, st, X, Y)
        if log is not None and t % 500 == 0:
            log(f"  step {t} loss {float(l):.4f}")
    return p


def get_acts(p, X, chunk=4096):
    outs, res = [], []
    f = jax.jit(forward)
    for i in range(0, len(X), chunk):
        lo, r = f(p, X[i:i + chunk])
        outs.append(np.asarray(lo)); res.append(np.asarray(r))
    return np.concatenate(outs), np.concatenate(res)
