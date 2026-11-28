"""A 400-cell wireless MILP stand-in: cell sleeping with user association, solved with
SciPy's HiGHS under a time limit, against a greedy rule and large-neighbourhood search.

HFM's "Wireless 400-Cell Synthetic MILP Case Solver" has not published its instance
format or reward. This builds a problem of the kind its description and the literature
point to, so that the methods proposed for it (warm start from greedy, decomposition by
cell cluster, large-neighbourhood search around HiGHS) can be compared on equal time.

Instance (seeded): 400 cells on a 20 x 20 km square (macro and small cells), users
clustered around hotspots. Each user can attach to cells within range; its rate from a
cell depends on distance. Decisions: which cells are on (binary), and which cell serves
each user (binary). Constraints: each user is served or counted as dropped; a cell's
load (sum of demand / rate) is at most 1 if on and 0 if off. Objective: on-power of the
cells plus a penalty per dropped user (minimise).

Methods, each given the same wall-clock budget:
  greedy      switch on cells in order of users covered per watt, attach each user to
              its best active cell with room
  highs       the whole MILP in SciPy's HiGHS with the time limit
  lns         start from greedy; repeatedly free a cluster of neighbouring cells and their
              users, fix everything else, and re-solve the small MILP with HiGHS; keep
              improvements
Score: objective, and gap to the LP-relaxation bound.

Usage: python sleep_milp.py <seed> <seconds> [cells] [users]
"""
import sys
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

DROP = 30.0                                   # penalty per dropped user, in watts


def instance(seed, n_cells=400, n_users=6000):
    rng = np.random.default_rng(seed)
    macro = rng.random(n_cells) < 0.2
    cells = rng.uniform(0, 20, (n_cells, 2))
    power = np.where(macro, 400.0, 60.0)
    radius = np.where(macro, 2.5, 0.9)
    hot = rng.uniform(0, 20, (25, 2))
    near = rng.random(n_users) < 0.7
    users = np.where(near[:, None], hot[rng.integers(0, 25, n_users)] + rng.normal(0, 0.8, (n_users, 2)), rng.uniform(0, 20, (n_users, 2)))
    users = users.clip(0, 20)
    demand = rng.uniform(1, 6, n_users)                                 # Mbit/s
    d = np.linalg.norm(users[:, None] - cells[None], axis=2)
    ok = d <= radius[None]
    peak = np.where(macro, 150.0, 300.0)                                # Mbit/s at zero distance
    rate = peak[None] * np.clip(1 - d / radius[None], 0.05, 1)
    pairs = np.argwhere(ok)                                             # (user, cell)
    load = demand[pairs[:, 0]] / rate[pairs[:, 0], pairs[:, 1]]
    return dict(cells=cells, power=power, macro=macro, users=users, demand=demand, pairs=pairs, load=load)


def model(inst):
    """Variables: y (cells), x (pairs), z (dropped users). Returns c, A, lb, ub, integrality, sizes."""
    nc, nu, npair = len(inst["power"]), len(inst["demand"]), len(inst["pairs"])
    u, c = inst["pairs"][:, 0], inst["pairs"][:, 1]
    cost = np.concatenate([inst["power"], np.zeros(npair), np.full(nu, DROP)])
    # each user: sum_c x_uc + z_u = 1
    r1 = np.concatenate([u, np.arange(nu)])
    k1 = np.concatenate([nc + np.arange(npair), nc + npair + np.arange(nu)])
    A1 = coo_matrix((np.ones(len(r1)), (r1, k1)), shape=(nu, nc + npair + nu))
    # each cell: sum_u load_uc x_uc - y_c <= 0
    A2 = coo_matrix((np.concatenate([inst["load"], -np.ones(nc)]), (np.concatenate([c, np.arange(nc)]), np.concatenate([nc + np.arange(npair), np.arange(nc)]))),
                    shape=(nc, nc + npair + nu))
    # x_uc <= y_c (tightens the relaxation)
    A3 = coo_matrix((np.concatenate([np.ones(npair), -np.ones(npair)]), (np.concatenate([np.arange(npair)] * 2), np.concatenate([nc + np.arange(npair), c]))),
                    shape=(npair, nc + npair + nu))
    from scipy.sparse import vstack
    A = vstack([A1, A2, A3]).tocsr()
    lo = np.concatenate([np.ones(nu), np.full(nc, -np.inf), np.full(npair, -np.inf)])
    hi = np.concatenate([np.ones(nu), np.zeros(nc), np.zeros(npair)])
    integ = np.concatenate([np.ones(nc + npair), np.zeros(nu)])
    return cost, A, lo, hi, integ, (nc, npair, nu)


def objective(inst, on, assign):
    dropped = (assign < 0).sum()
    return inst["power"][on].sum() + DROP * dropped, dropped


def greedy(inst):
    nc, nu = len(inst["power"]), len(inst["demand"])
    u, c, load = inst["pairs"][:, 0], inst["pairs"][:, 1], inst["load"]
    by_cell = [np.where(c == k)[0] for k in range(nc)]
    on = np.zeros(nc, bool)
    room = np.ones(nc)
    assign = -np.ones(nu, int)
    while True:
        best, gain_best = None, 0.0
        for k in np.where(~on)[0]:
            idx = by_cell[k][assign[u[by_cell[k]]] < 0]
            idx = idx[np.argsort(load[idx])]
            take = idx[np.cumsum(load[idx]) <= 1.0]
            g = DROP * len(take) - inst["power"][k]
            if g > gain_best:
                best, gain_best, chosen = k, g, take
        if best is None:
            break
        on[best] = True
        assign[u[chosen]] = best
        room[best] -= load[chosen].sum()
    return on, assign


def to_vector(inst, on, assign, sizes):
    nc, npair, nu = sizes
    x = np.zeros(nc + npair + nu)
    x[:nc] = on
    key = {(a, b): i for i, (a, b) in enumerate(inst["pairs"])}
    for usr, cell in enumerate(assign):
        if cell >= 0:
            x[nc + key[(usr, cell)]] = 1
        else:
            x[nc + npair + usr] = 1
    return x


def from_vector(inst, x, sizes):
    nc, npair, nu = sizes
    on = x[:nc] > 0.5
    assign = -np.ones(nu, int)
    sel = np.where(x[nc:nc + npair] > 0.5)[0]
    assign[inst["pairs"][sel, 0]] = inst["pairs"][sel, 1]
    return on, assign


def lns(inst, cost, A, lo, hi, integ, sizes, seconds, rng, cluster=25, start=None, sub_limit=20.0):
    nc, npair, nu = sizes
    if start is None:
        on, assign = greedy(inst)
        x = to_vector(inst, on, assign, sizes)
    else:
        x = start.copy()
    best = cost @ x
    t0 = time.time()
    u_of = inst["pairs"][:, 0]
    c_of = inst["pairs"][:, 1]
    rounds = improved = 0
    while time.time() - t0 < seconds:
        seed_cell = rng.integers(nc)
        dist = np.linalg.norm(inst["cells"] - inst["cells"][seed_cell], axis=1)
        free_cells = np.argsort(dist)[:cluster]
        free_users = np.unique(u_of[np.isin(c_of, free_cells)])
        free = np.zeros(len(x), bool)
        free[free_cells] = True
        free[nc + np.where(np.isin(u_of, free_users))[0]] = True
        free[nc + npair + free_users] = True
        lb = np.where(free, 0.0, x)
        ub = np.where(free, 1.0, x)
        left = seconds - (time.time() - t0)
        if left < 1:
            break
        res = milp(cost, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=Bounds(lb, ub),
                   options={"time_limit": min(sub_limit, left), "disp": False})
        rounds += 1
        if res.x is not None and cost @ res.x < best - 1e-6:
            x, best = np.round(res.x), cost @ res.x
            improved += 1
    return x, best, rounds, improved


def main():
    seed, seconds = int(sys.argv[1]), float(sys.argv[2])
    n_cells = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    n_users = int(sys.argv[4]) if len(sys.argv) > 4 else 6000
    inst = instance(seed, n_cells, n_users)
    cost, A, lo, hi, integ, sizes = model(inst)
    nc, npair, nu = sizes
    print(f"seed {seed}: {nc} cells ({inst['macro'].sum()} macro), {nu} users, {npair} user-cell pairs, "
          f"{len(cost)} variables, {A.shape[0]} constraints", flush=True)
    t = time.time()
    lp = milp(cost, constraints=LinearConstraint(A, lo, hi), integrality=np.zeros(len(cost)), bounds=Bounds(0, 1), options={"time_limit": 600})
    bound = lp.fun
    print(f"  LP bound {bound:.1f} ({time.time() - t:.0f}s)", flush=True)
    t = time.time()
    on, assign = greedy(inst)
    g, gd = objective(inst, on, assign)
    print(f"RESULT seed {seed} greedy: {g:.1f}, gap {(g - bound) / bound:.2%}, cells on {on.sum()}, dropped {gd} ({time.time() - t:.0f}s)", flush=True)
    t = time.time()
    res = milp(cost, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=Bounds(0, 1),
               options={"time_limit": seconds, "disp": False})
    if res.x is not None:
        h = cost @ np.round(res.x)
        on_h, assign_h = from_vector(inst, np.round(res.x), sizes)
        print(f"RESULT seed {seed} highs {seconds:.0f}s: {h:.1f}, gap {(h - bound) / bound:.2%}, cells on {on_h.sum()}, "
              f"dropped {(assign_h < 0).sum()}, solver status {res.status} ({res.message[:60]}) ({time.time() - t:.0f}s)", flush=True)
    else:
        print(f"RESULT seed {seed} highs {seconds:.0f}s: no feasible solution ({res.message[:80]})", flush=True)
    t = time.time()
    x, best, rounds, improved = lns(inst, cost, A, lo, hi, integ, sizes, seconds, np.random.default_rng(seed))
    on_l, assign_l = from_vector(inst, x, sizes)
    print(f"RESULT seed {seed} lns {seconds:.0f}s: {best:.1f}, gap {(best - bound) / bound:.2%}, cells on {on_l.sum()}, "
          f"dropped {(assign_l < 0).sum()}, {rounds} rounds, {improved} improved ({time.time() - t:.0f}s)", flush=True)
    # hybrid: HiGHS for 60% of the budget, then small, fast neighbourhoods around its incumbent
    t = time.time()
    res = milp(cost, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=Bounds(0, 1),
               options={"time_limit": 0.6 * seconds, "disp": False})
    if res.x is not None:
        start = np.round(res.x)
        h0 = cost @ start
        x, best, rounds, improved = lns(inst, cost, A, lo, hi, integ, sizes, seconds - (time.time() - t), np.random.default_rng(seed + 1),
                                        cluster=12, start=start, sub_limit=4.0)
        print(f"RESULT seed {seed} highs+lns {seconds:.0f}s: {best:.1f}, gap {(best - bound) / bound:.2%} (HiGHS part alone {h0:.1f}, "
              f"gap {(h0 - bound) / bound:.2%}), {rounds} rounds, {improved} improved ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
