import os, numpy as np
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "othello")
AGE_BINS = [("empty", None), ("age 0", (0, 0)), ("age 1-3", (1, 3)), ("age 4-7", (4, 7)),
            ("age 8-15", (8, 15)), ("age 16+", (16, 127))]


def load_lab(names):
    d = [np.load(os.path.join(OUT, f"data_{n}.npz")) for n in names]
    toks = np.concatenate([x["toks"] for x in d]).astype(np.int64)
    labs = np.concatenate([x["labs"] for x in d])
    ages = np.concatenate([x["ages"] for x in d])
    lens = np.concatenate([x["lens"] for x in d]).astype(np.int64)
    return toks, labs, ages, lens


def valid_mask(lens, L=60):
    return np.arange(L)[None, :] < lens[:, None]


def stratified(correct, ages):
    """correct, ages: (..., 64) arrays over valid positions. Returns dict bin -> (acc, count)."""
    out = {"all": (float(correct.mean()), int(correct.size))}
    for name, rng in AGE_BINS:
        sel = ages < 0 if rng is None else (ages >= rng[0]) & (ages <= rng[1])
        if sel.any():
            out[name] = (float(correct[sel].mean()), int(sel.sum()))
    occ = ages >= 0
    out["occupied"] = (float(correct[occ].mean()), int(occ.sum()))
    return out
