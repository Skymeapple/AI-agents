"""Predicting optical constants (n, k) outside the measured wavelength band.

Public proxy for HFM's "Predicting Optical Constants Beyond the Measured Wavelength
Band" track, built on the refractiveindex.info database (public domain, CC0):
https://github.com/polyanskiy/refractiveindex.info-database

It tests the route proposed in research/human-frontier-mission.md, section 7.5:
fit only Kramers-Kronig-consistent models. The model is a non-negative sum of
Lorentz and Drude terms on a fixed pole grid that extends beyond the band, fitted
to the complex permittivity inside the band by non-negative least squares. Because
every term is causal, the real part measured inside the band constrains absorption
outside it.

Protocol: for each tabulated (n, k) sample, the middle of the log-wavelength range
is "measured" and both ends are withheld. Reference = hold the nearest measured
value (skill 0). Skill = 1 - error / reference error. Hyper-parameters are chosen
on a development half of the materials; results are reported on the other half.
Materials are split by database "book" so one material never sits in both halves.

HFM's samples, band choices and scoring are private, so these numbers are NOT
comparable to HFM's skill_mean. They test the method on public data only.

Usage: python research/hfm-experiments/optical_extrapolation.py <path-to-database>
"""
from __future__ import annotations

import hashlib
import itertools
import os
import statistics
import sys

import numpy as np
import yaml
from scipy.optimize import nnls


def load_samples(root):
    out = []
    for dp, _, fns in os.walk(os.path.join(root, "database", "data")):
        for fn in fns:
            if not fn.endswith(".yml"):
                continue
            path = os.path.join(dp, fn)
            try:
                doc = yaml.safe_load(open(path, encoding="utf-8"))
            except Exception:
                continue
            for block in (doc or {}).get("DATA", []) or []:
                if block.get("type") != "tabulated nk":
                    continue
                try:
                    arr = np.array([[float(x) for x in ln.split()[:3]] for ln in block["data"].strip().splitlines()])
                except Exception:
                    continue
                if arr.ndim != 2 or arr.shape[1] != 3:
                    continue
                arr = arr[np.argsort(arr[:, 0])]
                _, idx = np.unique(arr[:, 0], return_index=True)
                arr = arr[idx]
                if len(arr) < 60 or arr[0, 0] <= 0 or arr[-1, 0] / arr[0, 0] < 4:
                    continue
                if (arr[:, 1] <= 0).any() or (arr[:, 2] < 0).any() or not np.isfinite(arr).all():
                    continue
                rel = os.path.relpath(path, os.path.join(root, "database", "data"))
                book = "/".join(rel.split(os.sep)[:2])
                out.append((rel, book, arr))
                break
    return out


def split(arr, frac=0.25):
    """Middle of the log-wavelength range is measured; outer `frac` on each side is withheld."""
    lw = np.log(arr[:, 0])
    a = lw[0] + frac * (lw[-1] - lw[0])
    b = lw[-1] - frac * (lw[-1] - lw[0])
    inside = (lw >= a) & (lw <= b)
    return arr[inside], arr[~inside]


def hold_last(obs, wl):
    lo, hi = obs[0], obs[-1]
    n = np.where(wl < lo[0], lo[1], hi[1])
    k = np.where(wl < lo[0], lo[2], hi[2])
    return n, k


def linear_log(obs, wl, m=5):
    """Naive baseline: extend the end slopes linearly in log-wavelength."""
    n, k = hold_last(obs, wl)
    for side, sl in (("lo", slice(0, m)), ("hi", slice(-m, None))):
        x = np.log(obs[sl, 0])
        mask = wl < obs[0, 0] if side == "lo" else wl > obs[-1, 0]
        ref = obs[0] if side == "lo" else obs[-1]
        for col, tgt in ((1, n), (2, k)):
            slope = np.polyfit(x, obs[sl, col], 1)[0]
            tgt[mask] = ref[col] + slope * (np.log(wl[mask]) - np.log(ref[0]))
    return np.maximum(n, 0.01), np.maximum(k, 0.0)


def basis(w, poles, gammas, drude):
    """Columns: constant (eps_inf - 1), Lorentz terms, Drude terms. w = 1/wavelength."""
    cols = [np.ones_like(w, dtype=complex)]
    for w0 in poles:
        for g in gammas:
            cols.append(w0 ** 2 / (w0 ** 2 - w ** 2 - 1j * g * w0 * w))
    for gd in drude:
        cols.append(-1.0 / (w ** 2 + 1j * gd * w))
    return np.stack(cols, axis=1)


def kk_fit_predict(obs, wl, per_octave=6, reach=20.0, gammas=(0.03, 0.15, 0.6), ridge=1e-3, anchor=True):
    w_obs = 1.0 / obs[:, 0]
    eps = (obs[:, 1] + 1j * obs[:, 2]) ** 2
    lo, hi = w_obs.min() / reach, w_obs.max() * reach
    poles = np.exp(np.arange(np.log(lo), np.log(hi), np.log(2) / per_octave))
    drude = [g * w_obs.min() for g in (0.1, 1.0, 10.0)]
    A = basis(w_obs, poles, gammas, drude)
    y = eps - 1.0
    scale = max(np.abs(y).mean(), 1e-9)
    norms = np.sqrt((np.abs(A) ** 2).mean(axis=0)) + 1e-12
    An = A / norms
    M = np.vstack([An.real, An.imag, np.sqrt(ridge) * np.eye(An.shape[1])])
    rhs = np.concatenate([y.real, y.imag, np.zeros(An.shape[1])]) / scale
    coef, _ = nnls(M, rhs, maxiter=20 * M.shape[1])
    coef = coef * scale / norms

    def model(wv):
        e = 1.0 + basis(1.0 / wv, poles, gammas, drude) @ coef
        nk = np.sqrt(e)
        nk = np.where(nk.imag < 0, -nk, nk)
        return nk.real, nk.imag

    n, k = model(wl)
    if anchor:  # remove the model's misfit at each band edge so predictions start from the data
        for mask, ref in ((wl < obs[0, 0], obs[0]), (wl > obs[-1, 0], obs[-1])):
            n0, k0 = model(np.array([ref[0]]))
            n[mask] += ref[1] - n0[0]
            k[mask] += ref[2] - k0[0]
    return np.maximum(n, 0.01), np.maximum(k, 0.0)


def err(pred, held):
    return float(np.sqrt(np.mean((pred[0] - held[:, 1]) ** 2 + (pred[1] - held[:, 2]) ** 2)))


def skills(samples, method):
    out = []
    for _, _, arr in samples:
        obs, held = split(arr)
        if len(obs) < 25 or len(held) < 10:
            continue
        ref = err(hold_last(obs, held[:, 0]), held)
        if ref < 1e-6:
            continue
        hn, hk = hold_last(obs, held[:, 0])
        pn, pk = method(obs, held[:, 0])
        out.append(max(1.0 - err((pn, pk), held) / ref, -1.0))  # floor at -1 so one blow-up cannot dominate
    return out


def blended(weight, **kw):
    def f(obs, wl):
        hn, hk = hold_last(obs, wl)
        n, k = kk_fit_predict(obs, wl, **kw)
        return hn + weight * (n - hn), hk + weight * (k - hk)
    return f


def summarise(name, s):
    wins = sum(1 for v in s if v > 0) / len(s)
    print(f"  {name:<42} mean skill {statistics.mean(s):+.3f}   median {statistics.median(s):+.3f}   better than reference on {wins:.0%}")


def main():
    root = sys.argv[1]
    samples = load_samples(root)
    dev = [s for s in samples if int(hashlib.md5(s[1].encode()).hexdigest(), 16) % 2 == 0]
    test = [s for s in samples if int(hashlib.md5(s[1].encode()).hexdigest(), 16) % 2 == 1]
    print(f"samples: {len(samples)} usable tabulated n,k records; dev {len(dev)}, test {len(test)} (split by material)")

    grid = list(itertools.product((1e-2, 1.0, 10.0, 100.0), (False,), (0.5, 0.75, 1.0)))
    best = None
    print("development set (hyper-parameter choice):")
    for ridge, anchor, weight in grid:
        s = skills(dev, blended(weight, ridge=ridge, anchor=anchor))
        m = statistics.mean(s)
        print(f"  ridge={ridge:<7} anchor={anchor!s:<5} blend={weight:<4} -> mean skill {m:+.3f}")
        if best is None or m > best[0]:
            best = (m, ridge, anchor, weight)
    _, ridge, anchor, weight = best
    print(f"chosen on dev: ridge={ridge}, anchor={anchor}, blend={weight}")

    print("held-out test materials:")
    summarise("hold last value (reference, skill 0)", skills(test, hold_last))
    summarise("linear extension in log-wavelength", skills(test, linear_log))
    summarise("KK-consistent fit, unblended, no anchor", skills(test, blended(1.0, ridge=ridge, anchor=False)))
    summarise("KK-consistent fit, chosen settings", skills(test, blended(weight, ridge=ridge, anchor=anchor)))


if __name__ == "__main__":
    main()
