"""Optical constants outside the measured band, by analogy with other materials.

Follow-up to optical_extrapolation.py in this folder. That script showed that a
causality-consistent fit alone is weak, because resonances outside the band leave
little trace inside it. This one tests the idea that worked for the steel project:
borrow the missing information from a library of other samples.

Method: for a target sample, find library records whose n and k inside the measured
band are closest to the target's, among records that also cover the wavelengths to
be predicted. Continue the target's curve with the change those donors show beyond
the band edge (median over the nearest donors), starting from the target's own
edge values. If no donor is close enough, hold the last value.

Two libraries are tried:
  other materials only   donors must come from a different database "book"
                         (a different material)
  any other record       donors may be another measurement of the same material,
                         by a different author or method

Same data, split and scoring as optical_extrapolation.py: refractiveindex.info
records, the middle of each wavelength range measured and both ends withheld;
settings chosen on the development half of the materials, results reported on the
other half; skill = 1 - error / error of holding the last value.

Usage: python donor_extrapolation.py <path-to-refractiveindex.info-database>
"""
from __future__ import annotations

import hashlib
import itertools
import os
import statistics
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optical_extrapolation import err, hold_last, load_samples, split  # noqa: E402


def donor_predict(obs, wl, book, library, k, tol, same_material):
    n, kk = hold_last(obs, wl)
    lo, hi = obs[0, 0], obs[-1, 0]
    for side in ("lo", "hi"):
        mask = wl < lo if side == "lo" else wl > hi
        if not mask.any():
            continue
        need_lo, need_hi = (wl[mask].min(), hi) if side == "lo" else (lo, wl[mask].max())
        edge = obs[0] if side == "lo" else obs[-1]
        cands = []
        for dbook, darr, did in library:
            if did is obs.base or (dbook == book and not same_material):
                continue
            if darr[0, 0] > need_lo or darr[-1, 0] < need_hi:
                continue
            dn = np.interp(obs[:, 0], darr[:, 0], darr[:, 1])
            dk = np.interp(obs[:, 0], darr[:, 0], darr[:, 2])
            dist = float(np.sqrt(np.mean((dn - obs[:, 1]) ** 2 + (dk - obs[:, 2]) ** 2)))
            if dist <= tol:
                cands.append((dist, darr))
        if len(cands) == 0:
            continue
        cands.sort(key=lambda c: c[0])
        dn_out, dk_out = [], []
        for _, darr in cands[:k]:
            dn_out.append(np.interp(wl[mask], darr[:, 0], darr[:, 1]) - np.interp(edge[0], darr[:, 0], darr[:, 1]))
            dk_out.append(np.interp(wl[mask], darr[:, 0], darr[:, 2]) - np.interp(edge[0], darr[:, 0], darr[:, 2]))
        n[mask] = edge[1] + np.median(dn_out, axis=0)
        kk[mask] = np.maximum(edge[2] + np.median(dk_out, axis=0), 0.0)
    return np.maximum(n, 0.01), kk


def evaluate(samples, library, k, tol, same_material):
    out, used = [], 0
    for rel, book, arr in samples:
        obs, held = split(arr)
        if len(obs) < 25 or len(held) < 10:
            continue
        ref = err(hold_last(obs, held[:, 0]), held)
        if ref < 1e-6:
            continue
        lib = [(b, a, a) for r, b, a in library if r != rel]
        pn, pk = donor_predict(obs, held[:, 0], book, lib, k, tol, same_material)
        e = err((pn, pk), held)
        used += abs(e - ref) > 1e-12
        out.append(max(1.0 - e / ref, -1.0))
    return out, used


def main():
    samples = load_samples(sys.argv[1])
    half = lambda s: int(hashlib.md5(s[1].encode()).hexdigest(), 16) % 2
    dev, test = [s for s in samples if half(s) == 0], [s for s in samples if half(s) == 1]
    print(f"{len(samples)} records; development {len(dev)}, test {len(test)} (split by material)")
    for same_material, label in ((False, "donors from other materials only"), (True, "donors may be another record of the same material")):
        best = None
        for k, tol in itertools.product((1, 3, 5), (0.05, 0.15, 0.5)):
            s, used = evaluate(dev, samples, k, tol, same_material)
            m = statistics.mean(s)
            if best is None or m > best[0]:
                best = (m, k, tol)
        _, k, tol = best
        s, used = evaluate(test, samples, k, tol, same_material)
        wins, losses = sum(v > 1e-9 for v in s), sum(v < -1e-9 for v in s)
        print(f"\n{label}: chosen on development: {k} donors, in-band distance at most {tol} (development skill {best[0]:+.3f})")
        print(f"  test: mean skill {statistics.mean(s):+.3f}, median {statistics.median(s):+.3f}; a donor was found for {used} of {len(s)} records; "
              f"better than holding the last value on {wins}, worse on {losses}")
        hit = [v for v in s if abs(v) > 1e-9]
        if hit:
            print(f"  among records with a donor: mean skill {statistics.mean(hit):+.3f}, median {statistics.median(hit):+.3f}")


if __name__ == "__main__":
    main()
