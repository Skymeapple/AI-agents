"""Compiled iterated local search for 1|r_j|sum w_j T_j, to see where the search plateaus.

Follow-up to tardiness.py in this folder. The pure-Python search there evaluates a
few thousand schedules per second. This version compiles the same search with
numba (about 100 times more moves per second) and runs it at several time budgets.
If ten times more search barely changes the result, the neighbourhood is at its
plateau and further gains need a different method, not more time. An independent
random multi-start descent is run as a cross-check: if it never finds anything
better, the plateau is probably close to the optimum and not just a trap.

Usage: python tardiness_fast.py [--seeds 10]
"""
from __future__ import annotations

import argparse
import statistics

import numpy as np
from numba import njit

from tardiness import best_atc, make_instance


@njit(cache=True)
def cost(seq, p, w, r, d):
    t = 0
    total = 0
    for j in seq:
        if t < r[j]:
            t = r[j]
        t += p[j]
        if t > d[j]:
            total += w[j] * (t - d[j])
    return total


@njit(cache=True)
def descend(seq, p, w, r, d, window):
    """First-improvement descent over swaps and insertions within a window."""
    n = len(seq)
    cur = cost(seq, p, w, r, d)
    improved = True
    while improved:
        improved = False
        for i in range(n):
            for j in range(max(0, i - window), min(n, i + window + 1)):
                if i == j:
                    continue
                # insertion: move job at i to position j
                job = seq[i]
                if i < j:
                    for k in range(i, j):
                        seq[k] = seq[k + 1]
                else:
                    for k in range(i, j, -1):
                        seq[k] = seq[k - 1]
                seq[j] = job
                c = cost(seq, p, w, r, d)
                if c < cur:
                    cur = c
                    improved = True
                    break
                # undo
                if i < j:
                    for k in range(j, i, -1):
                        seq[k] = seq[k - 1]
                else:
                    for k in range(j, i):
                        seq[k] = seq[k + 1]
                seq[i] = job
    return cur


@njit(cache=True)
def ils(start, p, w, r, d, rounds, seed):
    np.random.seed(seed)
    n = len(start)
    s = start.copy()
    c = descend(s, p, w, r, d, n)
    best_c = c
    for _ in range(rounds):
        cand = s.copy()
        for _ in range(np.random.randint(2, 5)):
            i, j = np.random.randint(0, n), np.random.randint(0, n)
            job = cand[i]
            if i < j:
                cand[i:j] = cand[i + 1:j + 1]
            elif i > j:
                cand[j + 1:i + 1] = cand[j:i].copy()
            cand[j] = job
        cc = descend(cand, p, w, r, d, 12)
        if cc <= c:
            s, c = cand, cc
            if cc < best_c:
                best_c = cc
    return best_c


@njit(cache=True)
def multistart(p, w, r, d, starts, seed):
    """Independent check: full descent from many random orders, sharing nothing with the ILS run."""
    np.random.seed(seed)
    n = len(p)
    best = -1
    for _ in range(starts):
        s = np.random.permutation(n)
        c = descend(s, p, w, r, d, n)
        if best < 0 or c < best:
            best = c
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    budgets = (30, 300, 3000)                # perturbation rounds
    print(f"{'n':>4} {'TF':>4} | {'tuned ATC':>10} | " + " ".join(f"{b:>7} rounds" for b in budgets) + " | gain vs tuned ATC at each budget")
    allg = {b: [] for b in budgets}
    for n in (40, 100):
        for tf, rdd in ((0.4, 0.6), (0.6, 0.4), (0.8, 0.2)):
            atc_c, res, ms = [], {b: [] for b in budgets}, []
            for s in range(args.seeds):
                p, w, r, d = (np.array(x, dtype=np.int64) for x in make_instance(n, rdd, tf, 0.5, seed=1000 * n + 100 * int(tf * 10) + s))
                c0, seq0 = best_atc(list(p), list(w), list(r), list(d))
                atc_c.append(c0)
                for b in budgets:
                    res[b].append(ils(np.array(seq0, dtype=np.int64), p, w, r, d, b, s))
                ms.append(multistart(p, w, r, d, 300 if n == 40 else 40, s))
            gains = {b: statistics.mean(1 - x / a if a else 0 for x, a in zip(res[b], atc_c)) for b in budgets}
            for b in budgets:
                allg[b].append(gains[b])
            print(f"{n:>4} {tf:>4} | {statistics.mean(atc_c):>10.0f} | " + " ".join(f"{statistics.mean(res[b]):>14.0f}" for b in budgets)
                  + " | " + "  ".join(f"{gains[b]:.1%}" for b in budgets)
                  + f" | random multi-start: {statistics.mean(ms):.0f}, beats the largest ILS budget on {sum(m < x for m, x in zip(ms, res[budgets[-1]]))}/{len(ms)} instances", flush=True)
    print("\nmean reduction vs tuned ATC by search budget: " + ", ".join(f"{b} rounds {statistics.mean(allg[b]):.2%}" for b in budgets))


if __name__ == "__main__":
    main()
