"""Small MLP helpers shared by the non-sequence experiments."""
import jax
import jax.numpy as jnp
import numpy as np


def init_mlp(key, sizes, scale=1.0):
    keys = jax.random.split(key, len(sizes) - 1)
    return [{"w": jax.random.normal(k, (a, b)) * scale / np.sqrt(a), "b": jnp.zeros(b)}
            for k, a, b in zip(keys, sizes[:-1], sizes[1:])]


def mlp_apply(params, x, act=jax.nn.relu, return_hidden=False):
    hs = []
    for i, l in enumerate(params):
        x = x @ l["w"] + l["b"]
        if i < len(params) - 1:
            x = act(x)
            hs.append(x)
    return (x, hs) if return_hidden else x
