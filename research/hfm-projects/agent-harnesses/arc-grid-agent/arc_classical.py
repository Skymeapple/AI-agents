"""ARC without a language model: how many tasks does a small hand-written program search solve?

HFM's track builds an agent harness around a frozen language model and reports a solved
rate of 0.12 at baseline and 0.79 after harness work. That needs paid model access.
This script measures the floor a CPU reaches with no model at all, on the public
ARC-AGI-1 task files (github.com/fchollet/ARC-AGI, Apache 2.0; 400 training and 400
evaluation tasks):

  1. search over compositions of up to two grid operations (rotations, flips, crops,
     scaling, tiling, mirroring, gravity, hole filling, object selection), followed
     by an optional colour mapping learned from the examples;
  2. cell-wise rules: when output and input have the same shape, learn a lookup from
     each cell's local context to its output colour.

A program is accepted when it reproduces every example pair of the task. The primitives
were written while looking at training tasks only; the evaluation set is the held-out
measure. Up to two distinct predictions per test input are allowed, as in ARC's rules.

Usage: python arc_classical.py <ARC data dir with training/ and evaluation/>
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
from collections import Counter

import numpy as np


def bg(g):
    return Counter(g.ravel().tolist()).most_common(1)[0][0]


def components(g, diag=True):
    """Connected groups of non-background cells: list of (cells, bbox)."""
    b = bg(g)
    seen = np.zeros(g.shape, bool)
    out = []
    H, W = g.shape
    for r in range(H):
        for c in range(W):
            if g[r, c] != b and not seen[r, c]:
                stack, cells = [(r, c)], []
                seen[r, c] = True
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            if (dy or dx) and (diag or not (dy and dx)):
                                v, u = y + dy, x + dx
                                if 0 <= v < H and 0 <= u < W and not seen[v, u] and g[v, u] != b:
                                    seen[v, u] = True
                                    stack.append((v, u))
                ys, xs = [p[0] for p in cells], [p[1] for p in cells]
                out.append((cells, (min(ys), max(ys) + 1, min(xs), max(xs) + 1)))
    return out


def crop_bbox(g):
    b = bg(g)
    ys, xs = np.where(g != b)
    return g if len(ys) == 0 else g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def pick_object(g, key):
    cs = components(g)
    if not cs:
        return g
    cells, (a, b_, c, d) = key(cs)
    return g[a:b_, c:d]


def gravity(g, k):
    """Let non-background cells fall towards one side; k = number of quarter turns before falling down."""
    r = np.rot90(g, k).copy()
    b = bg(g)
    for c in range(r.shape[1]):
        col = r[:, c]
        keep = col[col != b]
        r[:, c] = b
        if len(keep):
            r[-len(keep):, c] = keep
    return np.rot90(r, -k)


def fill_holes(g):
    """Background cells not connected to the border take the colour of the cells around them (most common non-background neighbour)."""
    b = bg(g)
    H, W = g.shape
    out = g.copy()
    reach = np.zeros(g.shape, bool)
    stack = [(r, c) for r in range(H) for c in range(W) if (r in (0, H - 1) or c in (0, W - 1)) and g[r, c] == b]
    for p in stack:
        reach[p] = True
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            v, u = y + dy, x + dx
            if 0 <= v < H and 0 <= u < W and not reach[v, u] and g[v, u] == b:
                reach[v, u] = True
                stack.append((v, u))
    out[(g == b) & ~reach] = 10   # placeholder colour, resolved by the colour mapping
    return out


def scale(g, k):
    return np.kron(g, np.ones((k, k), dtype=g.dtype))


def down(g, k):
    return g[::k, ::k] if g.shape[0] % k == 0 and g.shape[1] % k == 0 else g


def symmetrise(g):
    """Overwrite each cell with the non-background value found among its mirror images (left-right and up-down)."""
    b = bg(g)
    out = g.copy()
    for m in (g[:, ::-1], g[::-1, :], g[::-1, ::-1]):
        out = np.where(out == b, m, out)
    return out


size = lambda c: len(c[0])
PRIMS = {
    "identity": lambda g: g,
    "rot90": lambda g: np.rot90(g, 1), "rot180": lambda g: np.rot90(g, 2), "rot270": lambda g: np.rot90(g, 3),
    "flip_lr": lambda g: g[:, ::-1], "flip_ud": lambda g: g[::-1, :], "transpose": lambda g: g.T, "anti_transpose": lambda g: np.rot90(g, 2).T,
    "crop_content": crop_bbox,
    "largest_object": lambda g: pick_object(g, lambda cs: max(cs, key=size)),
    "smallest_object": lambda g: pick_object(g, lambda cs: min(cs, key=size)),
    "top_half": lambda g: g[:g.shape[0] // 2], "bottom_half": lambda g: g[(g.shape[0] + 1) // 2:],
    "left_half": lambda g: g[:, :g.shape[1] // 2], "right_half": lambda g: g[:, (g.shape[1] + 1) // 2:],
    "scale2": lambda g: scale(g, 2), "scale3": lambda g: scale(g, 3), "down2": lambda g: down(g, 2), "down3": lambda g: down(g, 3),
    "tile_1x2": lambda g: np.tile(g, (1, 2)), "tile_2x1": lambda g: np.tile(g, (2, 1)), "tile_2x2": lambda g: np.tile(g, (2, 2)), "tile_3x3": lambda g: np.tile(g, (3, 3)),
    "mirror_right": lambda g: np.hstack([g, g[:, ::-1]]), "mirror_down": lambda g: np.vstack([g, g[::-1]]),
    "mirror_left": lambda g: np.hstack([g[:, ::-1], g]), "mirror_up": lambda g: np.vstack([g[::-1], g]),
    "mirror_4": lambda g: np.vstack([np.hstack([g, g[:, ::-1]]), np.hstack([g, g[:, ::-1]])[::-1]]),
    "fall_down": lambda g: gravity(g, 0), "fall_left": lambda g: gravity(g, 1), "fall_up": lambda g: gravity(g, 2), "fall_right": lambda g: gravity(g, 3),
    "fill_holes": fill_holes, "symmetrise": symmetrise,
}


def colour_map(pairs):
    """A single colour-to-colour mapping that turns every produced grid into its target, or None."""
    m = {}
    for p, t in pairs:
        if p.shape != t.shape:
            return None
        for a, b in zip(p.ravel().tolist(), t.ravel().tolist()):
            if m.setdefault(a, b) != b:
                return None
    return m


def apply_map(g, m):
    out = g.copy()
    for a, b in m.items():
        out[g == a] = b
    out[out == 10] = m.get(10, 0)
    return out


def search(train, tests, max_candidates=2):
    """Programs of one or two primitives plus a colour mapping that fit all examples; returns up to two predictions per test input."""
    names = list(PRIMS)
    progs = [(n,) for n in names] + [(a, b) for a in names for b in names if a != "identity" and b != "identity"]
    found, cache = [], {}

    def run(prog, g):
        for n in prog:
            key = (n, g.shape, g.tobytes())
            if key not in cache:
                try:
                    r = PRIMS[n](g)
                    cache[key] = r if r.size and max(r.shape) <= 30 else None
                except Exception:
                    cache[key] = None
            g = cache[key]
            if g is None:
                return None
        return g

    for prog in progs:
        outs = [run(prog, i) for i, _ in train]
        if any(o is None for o in outs):
            continue
        m = colour_map(list(zip(outs, [o for _, o in train])))
        if m is None:
            continue
        preds = []
        for t in tests:
            r = run(prog, t)
            if r is None or any(c not in m for c in set(r.ravel().tolist())):
                preds = None
                break
            preds.append(apply_map(r, m))
        if preds is not None and not any(all(np.array_equal(a, b) for a, b in zip(preds, f)) for _, f in found):
            found.append((prog, preds))
            if len(found) >= max_candidates:
                break
    return found


def context_features(g, r, c, kind):
    H, W = g.shape
    at = lambda y, x: int(g[y, x]) if 0 <= y < H and 0 <= x < W else -1
    if kind == "cell":
        return (at(r, c),)
    if kind == "cross":
        return (at(r, c), at(r - 1, c), at(r + 1, c), at(r, c - 1), at(r, c + 1))
    if kind == "square":
        return tuple(at(r + dy, c + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    if kind == "parity":
        return (at(r, c), r % 2, c % 2)
    if kind == "lines":   # the cell, and whether its row or column holds any non-background cell
        b = bg(g)
        return (at(r, c), bool((g[r] != b).any()), bool((g[:, c] != b).any()))
    if kind == "line_colours":
        b = bg(g)
        rr = [v for v in g[r].tolist() if v != b]
        cc = [v for v in g[:, c].tolist() if v != b]
        return (at(r, c), rr[0] if rr else -1, cc[0] if cc else -1)


def cellwise(train, tests):
    """Same-shape tasks: a lookup from local context to output colour, for several context definitions."""
    if any(i.shape != o.shape for i, o in train):
        return []
    found = []
    for kind in ("cell", "parity", "lines", "line_colours", "cross", "square"):
        table, ok = {}, True
        for i, o in train:
            for r in range(i.shape[0]):
                for c in range(i.shape[1]):
                    k = context_features(i, r, c, kind)
                    if table.setdefault(k, int(o[r, c])) != int(o[r, c]):
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue
        preds = []
        for t in tests:
            p = t.copy()
            for r in range(t.shape[0]):
                for c in range(t.shape[1]):
                    k = context_features(t, r, c, kind)
                    if k not in table:
                        ok = False
                        break
                    p[r, c] = table[k]
                if not ok:
                    break
            if not ok:
                break
            preds.append(p)
        if ok and not any(all(np.array_equal(a, b) for a, b in zip(preds, f)) for _, f in found):
            found.append((("cellwise:" + kind,), preds))
    return found


def evaluate(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.json")))
    solved1 = solved2 = fitted = 0
    by = Counter()
    t0 = time.time()
    for f in files:
        d = json.load(open(f))
        train = [(np.array(p["input"]), np.array(p["output"])) for p in d["train"]]
        tests = [np.array(p["input"]) for p in d["test"]]
        gold = [np.array(p["output"]) for p in d["test"]]
        cands = (search(train, tests) + cellwise(train, tests))[:2]
        if cands:
            fitted += 1
        hit = [all(p.shape == g.shape and np.array_equal(p, g) for p, g in zip(preds, gold)) for _, preds in cands]
        if hit and hit[0]:
            solved1 += 1
        if any(hit):
            solved2 += 1
            by["cell-wise rule" if cands[hit.index(True)][0][0].startswith("cellwise") else f"{len(cands[hit.index(True)][0])} operation(s)"] += 1
    n = len(files)
    print(f"{os.path.basename(folder):<11} tasks {n}; a program fitted all examples in {fitted} ({100 * fitted / n:.1f}%); "
          f"solved with one guess {solved1} ({100 * solved1 / n:.1f}%), with two guesses {solved2} ({100 * solved2 / n:.1f}%); {time.time() - t0:.0f}s")
    print("            solved by:", dict(by), "; fitted the examples but wrong on the test:", fitted - solved2)


if __name__ == "__main__":
    for split in ("training", "evaluation"):
        evaluate(os.path.join(sys.argv[1], split))
    print("\nHFM's agent with a frozen language model: 0.12 at baseline, 0.79 after harness work (its own task set and budget; for scale).")
