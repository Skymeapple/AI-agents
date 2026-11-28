"""How much does cheap pre-screening raise the stable fraction of a fixed budget, in
chemical systems the screen has never seen? CPU stand-in.

Stand-in for HFM's "Crystal Generation for Unseen Chemical Systems" track, which needs a
GPU for MatterGen sampling and MatterSim relaxation. The track's diagnosis is that the
64-structure budget is spent on generation, not selection, and the proposed route is to
over-generate, pre-screen cheaply, and spend the budget on the best. This measures the
selection half on public DFT data:

  - Data: JARVIS-DFT 3D (NIST, public; about 76,000 relaxed structures with energy above
    the convex hull). Binary and ternary entries only.
  - Each chemical system (e.g. Cd-Fe-P) plays the part of an unseen target. Its JARVIS
    entries play the part of a pool of generated candidates.
  - A gradient-boosted model, trained only on other systems, predicts energy above hull
    from composition and simple structure features (space group, density, atoms per
    cell, volume per atom). It stands for a cheap single-point screen.
  - From each held-out system's pool, choose k candidates: at random, by the screen,
    by the screen with at most one pick per formula until every formula is used
    (diversity), or by the true value (oracle). Score: share of picks that are stable,
    and stable distinct formulas per system.

Two splits: "system" holds out each system's own entries; "family" also removes from
training every entry whose element set is a subset or superset of a test system, so the
screen cannot lean on Cd-P or Cd-Fe-P-O when judging Cd-Fe-P.

Usage: python screen_floor.py <jarvis json> [stable threshold eV/atom ...]
"""
import json
import sys
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ELEMENTS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
            "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
            "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu").split()
IDX = {e: i for i, e in enumerate(ELEMENTS)}
# Pauling electronegativity (0 where undefined); periodic row and group follow from the index.
CHI = dict(zip(ELEMENTS, [2.20, 0, 0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, 0, 0.93, 1.31, 1.61, 1.90, 2.19, 2.58, 3.16, 0,
                          0.82, 1.00, 1.36, 1.54, 1.63, 1.66, 1.55, 1.83, 1.88, 1.91, 1.90, 1.65, 1.81, 2.01, 2.18, 2.55, 2.96, 3.00,
                          0.82, 0.95, 1.22, 1.33, 1.6, 2.16, 1.9, 2.2, 2.28, 2.20, 1.93, 1.69, 1.78, 1.96, 2.05, 2.1, 2.66, 2.6,
                          0.79, 0.89, 1.10, 1.12, 1.13, 1.14, 1.13, 1.17, 1.2, 1.2, 1.1, 1.22, 1.23, 1.24, 1.25, 1.1, 1.27,
                          1.3, 1.5, 2.36, 1.9, 2.2, 2.20, 2.28, 2.54, 2.00, 1.62, 2.33, 2.02, 2.0, 2.2, 2.2,
                          0.7, 0.9, 1.1, 1.3, 1.5, 1.38, 1.36, 1.28]))
ROW_STARTS = (0, 2, 10, 18, 36, 54, 86)


def row_group(z):
    row = sum(z >= s for s in ROW_STARTS)
    return row, z - ROW_STARTS[row - 1]


def features(comp, spg, density, nat, vpa):
    frac = np.zeros(len(ELEMENTS))
    for e, f in comp.items():
        frac[IDX[e]] = f
    props = []
    for getter in (lambda e: CHI[e], lambda e: row_group(IDX[e])[0], lambda e: row_group(IDX[e])[1], lambda e: IDX[e]):
        v = np.array([getter(e) for e in comp])
        w = np.array(list(comp.values()))
        m = (v * w).sum()
        props += [m, np.sqrt((w * (v - m) ** 2).sum()), v.min(), v.max()]
    return np.concatenate([frac, props, [spg, density, nat, vpa, len(comp)]])


def load(path):
    raw = json.load(open(path))
    print(f"entries {len(raw)}; fields e.g. {sorted(raw[0])[:40]}", flush=True)
    rows = []
    for d in raw:
        try:
            eh = float(d["ehull"])
            els = d["atoms"]["elements"]
            lat = np.array(d["atoms"]["lattice_mat"], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
        counts = defaultdict(int)
        for e in els:
            counts[e] += 1
        if not 2 <= len(counts) <= 3 or any(e not in IDX for e in counts) or not np.isfinite(eh):
            continue
        n = len(els)
        comp = {e: c / n for e, c in counts.items()}
        g = np.gcd.reduce(list(counts.values()))
        formula = "".join(f"{e}{c // g}" for e, c in sorted(counts.items()))
        vpa = abs(np.linalg.det(lat)) / n
        spg = float(d.get("spg_number") or 0)
        dens = float(d.get("density") or 0)
        rows.append(("-".join(sorted(counts)), formula, eh, features(comp, spg, dens, n, vpa)))
    return rows


def choose(pool, k, how, rng):
    """pool: list of (formula, true, predicted). Returns chosen indices."""
    n = len(pool)
    if how == "random":
        return list(rng.choice(n, k, replace=False))
    key = 1 if how == "oracle" else 2
    order = sorted(range(n), key=lambda i: pool[i][key])
    if how != "diverse":
        return order[:k]
    picked, seen = [], set()
    for i in order:                                    # best per formula first
        if pool[i][0] not in seen:
            picked.append(i)
            seen.add(pool[i][0])
        if len(picked) == k:
            return picked
    return picked + [i for i in order if i not in picked][:k - len(picked)]


def main():
    path = sys.argv[1]
    thresholds = [float(t) for t in sys.argv[2:]] or [0.05, 0.1]
    rows = load(path)
    systems = sorted({r[0] for r in rows})
    by_sys = defaultdict(list)
    for i, r in enumerate(rows):
        by_sys[r[0]].append(i)
    X = np.stack([r[3] for r in rows])
    y = np.array([r[2] for r in rows])
    print(f"binary/ternary entries {len(rows)}, chemical systems {len(systems)}, "
          f"stable share at {thresholds[0]} eV/atom: {np.mean(y <= thresholds[0]):.3f}", flush=True)
    targets = [s for s in systems if len(by_sys[s]) >= 8]
    rng = np.random.default_rng(0)
    rng.shuffle(targets)
    folds = [targets[i::5] for i in range(5)]
    sets = {s: frozenset(s.split("-")) for s in systems}
    for split in ("system", "family"):
        pred = {}
        for f, test in enumerate(folds):
            test_sets = [sets[s] for s in test]
            if split == "system":
                banned = set(test)
            else:
                banned = {s for s in systems if any(sets[s] <= t or t <= sets[s] for t in test_sets)}
            train = [i for i, r in enumerate(rows) if r[0] not in banned]
            model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, random_state=0)
            model.fit(X[train], np.clip(y[train], 0, 2))
            for s in test:
                idx = by_sys[s]
                pred[s] = model.predict(X[idx])
            print(f"  [{split}] fold {f}: trained on {len(train)} entries, {len(test)} test systems", flush=True)
        err = np.concatenate([pred[s] - np.clip(y[by_sys[s]], 0, 2) for s in targets])
        print(f"\n===== split: {split}; screen mean absolute error {np.abs(err).mean():.3f} eV/atom on {len(err)} held-out entries =====")
        for thr in thresholds:
            for k in (1, 2, 4, 8):
                use = [s for s in targets if len(by_sys[s]) >= 2 * k]
                res = defaultdict(list)
                for s in use:
                    pool = [(rows[i][1], y[i], p) for i, p in zip(by_sys[s], pred[s])]
                    for how in ("random", "screen", "diverse", "oracle"):
                        reps = 20 if how == "random" else 1
                        hit, uniq = 0.0, 0.0
                        for _ in range(reps):
                            ch = choose(pool, k, how, rng)
                            stable = [pool[i][0] for i in ch if pool[i][1] <= thr]
                            hit += len(stable) / k
                            uniq += len(set(stable))
                        res[how].append((hit / reps, uniq / reps))
                line = "  ".join(f"{h} {np.mean([a for a, _ in v]):.3f} ({np.mean([b for _, b in v]):.2f} formulas)" for h, v in res.items())
                print(f"RESULT {split} stable<={thr} k={k} systems {len(use)}: {line}", flush=True)


if __name__ == "__main__":
    main()
