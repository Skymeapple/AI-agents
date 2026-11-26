"""No-network floors for 64-frame Rayleigh-Benard forecasting.

Stand-in for HFM's "64-Frame Rayleigh-Benard Convection Forecasting with Walrus and
The Well" track, which needs a GPU. Data: The Well's rayleigh_benard dataset
(Polymathic AI, CC-BY 4.0). Input is six consecutive frames; the forecast covers the
next 64. Nothing here is a neural network; the point is to measure how much of the
score simple statistics already explain, and how the error grows with lead time.

Methods
  persistence          repeat the last input frame
  profile              the fitting runs' mean vertical profile at the same time index
  damped persistence   profile + a(lead) x (last frame - profile); a fitted per field
  blurred + damped     the same after a Gaussian blur whose width is fitted per lead:
                       the best a forecaster can do by "blurring towards the mean"
  mode regression      ridge regression from the last two frames' POD coefficients to
                       each lead's POD coefficients (a linear dynamical forecast)

Weights are fitted on runs from the fit file, separately for each start regime
(windows starting within 16 frames of the test start), and scored on the test file.
Error is RMSE over space, frames and test runs divided by the standard deviation of
the true target frames, per field (pressure after removing each frame's mean), then
averaged over the four fields.

Usage: python forecast_floor.py <fit.hdf5> <test.hdf5> [fit-runs]
"""
import sys

import h5py
import numpy as np
from scipy.ndimage import gaussian_filter

HIST, LEAD = 6, 64
STARTS = (0, 65, 130)
SIGMAS = (0, 1, 2, 4, 8, 16, 32)
MODES = 60
NAMES = ("buoyancy", "pressure", "velocity x", "velocity y")


def load(path, max_runs=None):
    with h5py.File(path, "r") as h:
        b = h["t0_fields/buoyancy"][:max_runs].astype(np.float32)
        p = h["t0_fields/pressure"][:max_runs].astype(np.float32)
        v = h["t1_fields/velocity"][:max_runs].astype(np.float32)
    p = p - p.mean(axis=(2, 3), keepdims=True)
    return np.stack([b, p, v[..., 0], v[..., 1]], axis=2)            # (runs, time, 4, x, y)


def blur(a, s):
    """Gaussian blur of (..., x, y): periodic in x, walls in y."""
    if s == 0:
        return a
    out = np.empty_like(a)
    for idx in np.ndindex(a.shape[:-2]):
        out[idx] = gaussian_filter(a[idx], sigma=(s, s / 4), mode=("wrap", "nearest"))
    return out


def windows(D, prof, starts):
    """Anomaly of the last input frame (n, 4, x, y) and of the 64 targets (n, 64, 4, x, y)."""
    last, tgt, t0 = [], [], []
    for s in starts:
        if s < 0 or s + HIST + LEAD > D.shape[1]:
            continue
        for r in range(D.shape[0]):
            last.append(D[r, s + HIST - 1] - prof[s + HIST - 1])
            tgt.append(D[r, s + HIST:s + HIST + LEAD] - prof[s + HIST:s + HIST + LEAD])
            t0.append(s)
    return np.stack(last), np.stack(tgt), np.array(t0)


def main():
    fit_runs = int(sys.argv[3]) if len(sys.argv) > 3 else None
    F, T = load(sys.argv[1], fit_runs), load(sys.argv[2])
    nt = min(F.shape[1], T.shape[1])
    print(f"fit runs {F.shape[0]}, test runs {T.shape[0]}, frames {nt}, grid {F.shape[3]} x {F.shape[4]}", flush=True)
    prof = F.mean(axis=(0, 3), keepdims=True)[0]                      # (time, 4, 1, y)
    prof = np.broadcast_to(prof, F.shape[1:]).copy()
    summary = {}
    for s in STARTS:
        if s + HIST + LEAD > nt:
            continue
        fl, ft, _ = windows(F, prof, range(s - 16, s + 17, 8))
        tl, tt, _ = windows(T, prof, [s])
        prev_f = np.stack([F[r, st + HIST - 2] - prof[st + HIST - 2] for st in range(s - 16, s + 17, 8) if 0 <= st and st + HIST + LEAD <= nt for r in range(F.shape[0])])
        prev_t = np.stack([T[r, s + HIST - 2] - prof[s + HIST - 2] for r in range(T.shape[0])])
        truth = tt + prof[None, s + HIST:s + HIST + LEAD]
        scale = np.array([truth[:, :, c].std() for c in range(4)])
        print(f"\n===== start frame {s}: {len(fl)} fitting windows, {len(tl)} test windows =====", flush=True)

        def score(pred_anom):
            err = np.sqrt(np.mean((pred_anom - tt) ** 2, axis=(0, 3, 4)))        # (lead, field)
            per_field = np.sqrt(np.mean((pred_anom - tt) ** 2, axis=(0, 1, 3, 4))) / scale
            return per_field, err / scale[None]

        res = {}
        res["persistence"] = score(np.repeat(tl[:, None], LEAD, axis=1))
        res["profile"] = score(np.zeros_like(tt))
        # damped persistence and blurred + damped: per lead and field, best (sigma, a) on the fitting windows
        fb = {sg: blur(fl, sg) for sg in SIGMAS}
        tb = {sg: blur(tl, sg) for sg in SIGMAS}
        pred_d, pred_b = np.zeros_like(tt), np.zeros_like(tt)
        chosen = np.zeros((LEAD, 4))
        for k in range(LEAD):
            for c in range(4):
                y = ft[:, k, c]
                best = None
                for sg in SIGMAS:
                    x = fb[sg][:, c]
                    a = float((x * y).sum() / max((x * x).sum(), 1e-30))
                    e = float(((a * x - y) ** 2).mean())
                    if sg == 0:
                        pred_d[:, k, c] = a * tl[:, c]
                    if best is None or e < best[0]:
                        best = (e, sg, a)
                pred_b[:, k, c] = best[2] * tb[best[1]][:, c]
                chosen[k, c] = best[1]
        res["damped persistence"] = score(pred_d)
        res["blurred + damped"] = score(pred_b)
        # mode regression: POD on fitting targets and inputs, ridge map from two frames of coefficients
        n = len(fl)
        fs = np.array([F[:, :, c].std() for c in range(4)])[None, :, None, None]
        snaps = np.concatenate([fl / fs, (ft[:, ::8] / fs[:, None]).reshape(-1, *fl.shape[1:])]).reshape(-1, fl[0].size)
        w, V = np.linalg.eigh(snaps @ snaps.T)
        order = np.argsort(w)[::-1][:min(MODES, len(w) - 1)]
        basis = (snaps.T @ V[:, order]) / np.sqrt(np.maximum(w[order], 1e-12))       # (features, modes)
        proj = lambda a: (a / fs).reshape(len(a), -1) @ basis
        Xf = np.hstack([proj(fl), proj(prev_f)])
        Xt = np.hstack([proj(tl), proj(prev_t)])
        runs_f = np.tile(np.arange(F.shape[0]), n // F.shape[0])
        hold = runs_f >= F.shape[0] - max(1, F.shape[0] // 5)
        Yf = np.stack([proj(ft[:, k]) for k in range(LEAD)], axis=1)                  # (n, lead, modes)
        best = None
        for lam in (1e-2, 1e-1, 1, 10, 100, 1000):
            A = Xf[~hold]
            W = np.linalg.solve(A.T @ A + lam * np.trace(A.T @ A) / A.shape[1] * np.eye(A.shape[1]), A.T @ Yf[~hold].reshape((~hold).sum(), -1))
            e = float(((Xf[hold] @ W - Yf[hold].reshape(hold.sum(), -1)) ** 2).mean())
            if best is None or e < best[0]:
                best = (e, lam)
        lam = best[1]
        W = np.linalg.solve(Xf.T @ Xf + lam * np.trace(Xf.T @ Xf) / Xf.shape[1] * np.eye(Xf.shape[1]), Xf.T @ Yf.reshape(n, -1))
        coef = (Xt @ W).reshape(len(tl), LEAD, -1)
        pred_m = ((coef @ basis.T).reshape(len(tl), LEAD, *fl.shape[1:]) * fs[:, None]).astype(np.float32)
        res["mode regression"] = score(pred_m)
        print(f"  (mode regression: {len(order)} modes, ridge {lam:g}; blur widths chosen at leads 1/8/32/64, mean over fields: "
              + ", ".join(f"{chosen[k - 1].mean():.1f}" for k in (1, 8, 32, 64)) + " grid cells)")
        print(f"  {'method':22s}" + "".join(f"{nm:>12s}" for nm in NAMES) + f"{'mean':>9s}   | mean error at lead 1, 8, 32, 64")
        for name, (pf, by_lead) in res.items():
            leads = "  ".join(f"{by_lead[k - 1].mean():.3f}" for k in (1, 8, 32, 64))
            print(f"  {name:22s}" + "".join(f"{v:12.3f}" for v in pf) + f"{pf.mean():9.3f}   | {leads}", flush=True)
            summary.setdefault(name, []).append(float(pf.mean()))
    print("\n===== mean over start frames =====")
    for name, vals in summary.items():
        print(f"RESULT {name:22s} " + "  ".join(f"{v:.3f}" for v in vals) + f"   mean {np.mean(vals):.3f}")


if __name__ == "__main__":
    main()
