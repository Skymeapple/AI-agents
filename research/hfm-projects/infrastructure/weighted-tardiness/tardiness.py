"""Single-machine weighted tardiness with release times (1|r_j|sum w_j T_j).

Reproduces the *structure* of HFM's "Single-Machine Weighted Tardiness Scheduling
with Release Times" track on public, regenerable instances, to test whether the
route proposed in research/human-frontier-mission.md (iterated local search on top
of ATC, under the same one-second budget) beats the ATC dispatch rule HFM starts
from.

HFM's own instances and its normalized_tardiness_gain metric are private, so the
numbers here are NOT comparable to HFM's leaderboard. They test the method, not
the score.

Usage: python research/hfm-experiments/tardiness.py [--seeds 20] [--budget 1.0]
"""
from __future__ import annotations

import argparse
import math
import random
import statistics
import time


def make_instance(n: int, rdd: float, tf: float, rr: float, seed: int):
    """Standard-style generator: p~U[1,100], w~U[1,10], release r~U[0, rr*P],
    due date d = r + p + slack, slack~U[(1-tf-rdd/2)P, (1-tf+rdd/2)P] clipped at 0,
    with P = sum(p) scaled by 1/2 because release times spread the load."""
    rng = random.Random(seed)
    p = [rng.randint(1, 100) for _ in range(n)]
    w = [rng.randint(1, 10) for _ in range(n)]
    P = sum(p)
    r = [rng.randint(0, int(rr * P)) for _ in range(n)]
    lo, hi = (1 - tf - rdd / 2) * P * 0.5, (1 - tf + rdd / 2) * P * 0.5
    d = [r[j] + p[j] + max(0, int(rng.uniform(lo, hi))) for j in range(n)]
    return p, w, r, d


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


def atc(p, w, r, d, k=2.0):
    """Apparent Tardiness Cost dispatching with release times: at each decision
    point, among released jobs pick the max ATC index; if none released, jump to
    the earliest release."""
    n = len(p)
    pbar = sum(p) / n
    left = set(range(n))
    t = 0
    seq = []
    while left:
        ready = [j for j in left if r[j] <= t]
        if not ready:
            t = min(r[j] for j in left)
            continue
        best = max(ready, key=lambda j: (w[j] / p[j]) * math.exp(-max(d[j] - p[j] - t, 0) / (k * pbar)))
        seq.append(best)
        left.remove(best)
        t += p[best]
    return seq


def best_atc(p, w, r, d):
    """ATC with k tuned over a small grid (cheap, a fairer baseline)."""
    best = None
    for k in (0.5, 1.0, 2.0, 3.0, 4.0):
        s = atc(p, w, r, d, k)
        c = cost(s, p, w, r, d)
        if best is None or c < best[0]:
            best = (c, s)
    return best


def local_search(seq, p, w, r, d, deadline, window=12):
    """First-improvement descent over adjacent swaps and bounded insertions."""
    cur = cost(seq, p, w, r, d)
    n = len(seq)
    improved = True
    while improved and time.perf_counter() < deadline:
        improved = False
        for i in range(n - 1):
            seq[i], seq[i + 1] = seq[i + 1], seq[i]
            c = cost(seq, p, w, r, d)
            if c < cur:
                cur, improved = c, True
            else:
                seq[i], seq[i + 1] = seq[i + 1], seq[i]
        for i in range(n):
            if time.perf_counter() >= deadline:
                break
            job = seq[i]
            base = seq[:i] + seq[i + 1:]
            for j in range(max(0, i - window), min(n, i + window + 1)):
                if j == i:
                    continue
                cand = base[:j] + [job] + base[j:]
                c = cost(cand, p, w, r, d)
                if c < cur:
                    seq[:] = cand
                    cur, improved = c, True
                    break
    return seq, cur


def ils(p, w, r, d, budget, seed=0):
    """Iterated local search: ATC seed -> descent -> random segment perturbation
    -> descent, accepting non-worsening moves, until the time budget."""
    rng = random.Random(seed)
    deadline = time.perf_counter() + budget
    _, s = best_atc(p, w, r, d)
    s, c = local_search(s, p, w, r, d, deadline)
    best_s, best_c = s[:], c
    n = len(s)
    while time.perf_counter() < deadline:
        cand = best_s[:] if rng.random() < 0.5 else s[:]
        for _ in range(rng.randint(2, 4)):  # double-bridge-like kicks
            i, j = sorted(rng.sample(range(n), 2))
            cand.insert(j, cand.pop(i))
        cand, cc = local_search(cand, p, w, r, d, deadline)
        if cc <= c:
            s, c = cand, cc
            if cc < best_c:
                best_s, best_c = cand[:], cc
    return best_s, best_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--budget", type=float, default=1.0)
    args = ap.parse_args()
    print(f"{'n':>4} {'TF':>4} {'RDD':>4} | {'ATC(k=2)':>10} {'ATC best k':>10} {'ILS 1s':>10} | gain vs ATC(k=2)  gain vs tuned ATC")
    rows = []
    for n in (40, 100):
        for tf, rdd in ((0.4, 0.6), (0.6, 0.4), (0.8, 0.2)):
            a2, ab, il = [], [], []
            for s in range(args.seeds):
                p, w, r, d = make_instance(n, rdd, tf, 0.5, seed=1000 * n + 100 * int(tf * 10) + s)
                a2.append(cost(atc(p, w, r, d, 2.0), p, w, r, d))
                ab.append(best_atc(p, w, r, d)[0])
                il.append(ils(p, w, r, d, args.budget, seed=s)[1])
            g2 = [1 - b / a if a else 0 for a, b in zip(a2, il)]
            gb = [1 - b / a if a else 0 for a, b in zip(ab, il)]
            rows.append((n, tf, rdd, statistics.mean(g2), statistics.mean(gb)))
            print(f"{n:>4} {tf:>4} {rdd:>4} | {statistics.mean(a2):>10.0f} {statistics.mean(ab):>10.0f} {statistics.mean(il):>10.0f} |"
                  f"   {statistics.mean(g2):6.1%} (min {min(g2):.1%})    {statistics.mean(gb):6.1%} (min {min(gb):.1%})")
    print(f"\nmean weighted-tardiness reduction vs ATC(k=2): {statistics.mean(x[3] for x in rows):.1%}; vs tuned ATC: {statistics.mean(x[4] for x in rows):.1%}")


if __name__ == "__main__":
    main()
