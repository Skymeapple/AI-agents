"""Full-field reconstruction of Rayleigh-Benard convection from 256 point sensors.

Experiment for HFM's "Full-field reconstruction of 2D Rayleigh-Benard convection
from 256 fixed sensors" track. Data: The Well's rayleigh_benard dataset (Polymathic
AI, CC-BY 4.0, https://huggingface.co/datasets/polymathic-ai/rayleigh_benard):
512 x 128 grids, periodic in x, walls in y, fields buoyancy, pressure and velocity.

Task as in HFM: 256 fixed sensors report buoyancy and both velocity components,
not pressure, over the last eight frames. Reconstruct all four fields on the full
grid at the final frame. HFM starts from gappy POD.

For each flow setting the "valid" file (5 runs x 200 frames) is used for fitting and
the "test" file for scoring; they are different simulations. Score: RMSE of each
field divided by its standard deviation over the test frames, averaged over the
four fields (HFM's "field_nrmse" construction; its sensor layout and settings are
private, so numbers are NOT comparable).

Methods
  gappy POD            fit POD coefficients to the final frame's sensor readings
                       (HFM's starting point)
  sensor-history map   ridge regression from all eight frames of sensor readings
                       to the POD coefficients
  kernel map           the same with a Gaussian kernel (nonlinear)
  + Poisson pressure   replace the reconstructed pressure by the solution of the
                       pressure equation of the Boussinesq flow, computed from the
                       reconstructed buoyancy and velocity
                       (research/human-frontier-mission.md, section 7.9)
  combined             velocity from gappy POD, buoyancy from the history map,
                       pressure from the equation; scored on a subset of frames
The Poisson solver is first checked on the true fields; that error is printed.

Usage: python reconstruct.py <fit-file.hdf5> <test-file.hdf5> [more pairs ...]
"""
from __future__ import annotations

import sys

import h5py
import numpy as np

N_SENSORS, HISTORY, MODES, SKIP = 256, 8, int(__import__("os").environ.get("RB_MODES", 150)), 20


def load(path, max_runs=None):
    with h5py.File(path, "r") as h:
        b = h["t0_fields/buoyancy"][:max_runs].astype(np.float32)          # (runs, time, x, y)
        p = h["t0_fields/pressure"][:max_runs].astype(np.float32)
        v = h["t1_fields/velocity"][:max_runs].astype(np.float32)          # (..., 2): x and y components
        y = h["dimensions/y"][:].astype(np.float64)
        x = h["dimensions/x"][:].astype(np.float64)
        ra, pr = float(h.attrs.get("Rayleigh", 0) or 0), float(h.attrs.get("Prandtl", 0) or 0)
    fields = np.stack([b, v[..., 0], v[..., 1], p], axis=2)       # (runs, time, 4, x, y)
    nu = (pr / ra) ** 0.5 if ra > 0 else 0.0                      # viscosity in the dataset's units
    return fields[:, SKIP:], x, y, nu


def poisson_pressure(b, u, v, x, y, nu=0.0):
    """Pressure of the Boussinesq flow: lap(p) = db/dy - div(u . grad u).

    Wall condition from the vertical momentum equation with no flow through the wall:
    dp/dy = b + nu * d2v/dy2.
    """
    nx, ny = b.shape
    lx = (x[1] - x[0]) * nx
    kx = 2 * np.pi * np.fft.rfftfreq(nx, d=lx / nx)
    ddx = lambda f: np.fft.irfft(1j * kx[:, None] * np.fft.rfft(f, axis=0), n=nx, axis=0)
    ddy = lambda f: np.gradient(f, y, axis=1, edge_order=2)
    adv_u = u * ddx(u) + v * ddy(u)
    adv_v = u * ddx(v) + v * ddy(v)
    rhs = ddy(b) - ddx(adv_u) - ddy(adv_v)
    # second derivative in y on the (possibly non-uniform) grid
    D1 = np.zeros((ny, ny))
    for j in range(ny):
        e = np.zeros(ny); e[j] = 1.0
        D1[:, j] = np.gradient(e, y, edge_order=2)
    D2 = D1 @ D1
    rh = np.fft.rfft(rhs, axis=0)
    bh = np.fft.rfft(b, axis=0)
    vyy = np.fft.rfft(ddy(ddy(v)), axis=0)
    ph = np.zeros_like(rh)
    for i, k in enumerate(kx):
        A = D2 - (k ** 2) * np.eye(ny)
        r = rh[i].copy()
        A[0], A[-1] = D1[0], D1[-1]                 # wall condition dp/dy = b
        r[0], r[-1] = bh[i, 0] + nu * vyy[i, 0], bh[i, -1] + nu * vyy[i, -1]
        if i == 0:                                  # mean pressure is arbitrary: pin one value
            A[ny // 2] = 0.0; A[ny // 2, ny // 2] = 1.0; r[ny // 2] = 0.0
        ph[i] = np.linalg.solve(A, r)
    return np.fft.irfft(ph, n=nx, axis=0)


def nrmse(pred, true, scale):
    """pred, true: (samples, 4, x, y). Pressure is compared after removing each frame's mean."""
    out = []
    for c in range(4):
        a, t = pred[:, c], true[:, c]
        if c == 3:
            a = a - a.mean(axis=(1, 2), keepdims=True)
            t = t - t.mean(axis=(1, 2), keepdims=True)
        out.append(float(np.sqrt(np.mean((a - t) ** 2)) / scale[c]))
    return out


def run_pair(fit_path, test_path, rng):
    F, x, y, nu = load(fit_path, int(__import__("os").environ.get("RB_FIT_RUNS", 0)) or None)
    T, _, _, _ = load(test_path)
    stride = int(__import__("os").environ.get("RB_FIT_STRIDE", 1))
    spacing = np.diff(y)
    print(f"  y grid: {len(y)} points, spacing min {spacing.min():.5f} max {spacing.max():.5f}; viscosity {nu:.2e}; fitting runs {F.shape[0]}", flush=True)
    nx, ny = F.shape[3], F.shape[4]
    scale = np.array([T[:, :, c].std() for c in range(3)] + [(T[:, :, 3] - T[:, :, 3].mean(axis=(2, 3), keepdims=True)).std()])
    fscale = np.array([F[:, :, c].std() for c in range(4)])
    sx, sy = rng.integers(0, nx, N_SENSORS), rng.integers(2, ny - 2, N_SENSORS)

    def samples(D):
        """Final-frame fields and the eight-frame sensor history that precedes them."""
        runs, nt = D.shape[:2]
        tgt = np.concatenate([D[r, HISTORY - 1:] for r in range(runs)])                    # (n, 4, x, y)
        sens = D[:, :, :3][:, :, :, sx, sy] / fscale[None, None, :3, None]                  # (runs, time, 3, sensors)
        hist = np.concatenate([np.stack([sens[r, t - HISTORY + 1:t + 1].ravel() for t in range(HISTORY - 1, nt)]) for r in range(runs)])
        return tgt, hist, hist[:, -3 * N_SENSORS:]

    Yf, Hf, Lf = samples(F)
    if stride > 1:
        Yf, Hf, Lf = Yf[::stride], Hf[::stride], Lf[::stride]
    Yt, Ht, Lt = samples(T)
    n = len(Yf)
    X = (Yf / fscale[None, :, None, None]).reshape(n, -1)
    mean = X.mean(axis=0)
    Xc = X - mean
    w, V = np.linalg.eigh(Xc @ Xc.T)
    order = np.argsort(w)[::-1][:MODES]
    basis = ((Xc.T @ V[:, order].astype(np.float32)) / np.sqrt(np.maximum(w[order], 1e-9)).astype(np.float32))   # (features, modes)
    coef_fit = Xc @ basis
    energy = w[order].sum() / w.sum()

    def decode(coef):
        return ((coef @ basis.T + mean).reshape(-1, 4, nx, ny) * fscale[None, :, None, None]).astype(np.float32)

    # sensor rows of the basis, for gappy POD on the final frame
    grid_idx = np.concatenate([c * nx * ny + sx * ny + sy for c in range(3)])
    B, m_s = basis[grid_idx], mean[grid_idx]
    lam = 1e-2 * np.trace(B.T @ B) / MODES
    gappy = np.linalg.solve(B.T @ B + lam * np.eye(MODES), B.T @ (Lt - m_s).T).T
    # ridge map from the sensor history to the coefficients (dual form)
    hm, hs = Hf.mean(axis=0), Hf.std(axis=0) + 1e-6
    Zf, Zt = (Hf - hm) / hs, (Ht - hm) / hs
    K = Zf @ Zf.T
    alpha = np.linalg.solve(K + 1e-2 * np.trace(K) / n * np.eye(n), coef_fit)
    lin = (Zt @ Zf.T) @ alpha
    d2 = lambda A, Bm: (A ** 2).sum(1)[:, None] + (Bm ** 2).sum(1)[None] - 2 * A @ Bm.T
    Dff = d2(Zf, Zf)
    gamma = 1.0 / np.median(Dff)
    Kg = np.exp(-gamma * Dff)
    ag = np.linalg.solve(Kg + 1e-3 * np.eye(n), coef_fit)
    ker = np.exp(-gamma * d2(Zt, Zf)) @ ag

    recon = {"gappy POD (final frame)": decode(gappy), "sensor-history map (ridge)": decode(lin), "sensor-history map (kernel)": decode(ker),
             "best possible with these POD modes": decode((Yt / fscale[None, :, None, None]).reshape(len(Yt), -1) @ basis - mean @ basis)}
    res = {k: nrmse(v, Yt, scale) for k, v in recon.items()}
    # pressure from the flow equations, checked on the true fields first
    sub = np.arange(0, len(Yt), max(len(Yt) // 60, 1))
    solve = lambda Fld: np.stack([poisson_pressure(Fld[i, 0].astype(np.float64), Fld[i, 1].astype(np.float64), Fld[i, 2].astype(np.float64), x, y, nu) for i in range(len(Fld))])
    with_p = lambda Fld, P: np.concatenate([Fld[:, :3], P[:, None].astype(np.float32)], axis=1)
    chk = nrmse(with_p(Yt[sub], solve(Yt[sub])), Yt[sub], scale)[3]
    best = "gappy POD (final frame)"
    G = recon[best][sub]
    direct = nrmse(G, Yt[sub], scale)[3]
    viapoisson = nrmse(with_p(G, solve(G)), Yt[sub], scale)[3]
    # combination: buoyancy from the history map if it is better there, velocity from gappy POD, pressure from the equation
    H = recon["sensor-history map (ridge)"][sub]
    combo = G.copy()
    combo[:, 0] = H[:, 0]
    combo = with_p(combo, solve(combo))
    res["combined (subset of frames)"] = nrmse(combo, Yt[sub], scale)
    res["gappy POD (same subset)"] = nrmse(G, Yt[sub], scale)
    return res, energy, chk, best, direct, viapoisson, len(Yt)


def main():
    rng = np.random.default_rng(0)
    pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))
    agg = {}
    for fit_path, test_path in pairs:
        res, energy, chk, best, direct, viap, n = run_pair(fit_path, test_path, rng)
        name = fit_path.split("rayleigh_benard_")[-1].replace(".hdf5", "")
        print(f"\n{name}: {n} test frames; {MODES} POD modes hold {energy:.1%} of the fitting variance")
        print(f"  {'method':<38} {'buoyancy':>9} {'u':>7} {'v':>7} {'pressure':>9} {'mean':>7}")
        for k, r in res.items():
            print(f"  {k:<38} {r[0]:>9.3f} {r[1]:>7.3f} {r[2]:>7.3f} {r[3]:>9.3f} {np.mean(r):>7.3f}")
            agg.setdefault(k, []).append(np.mean(r))
        print(f"  pressure solver check on true fields: NRMSE {chk:.3f} (should be near 0 if the equation and wall condition are right)")
        print(f"  pressure from {best}: direct {direct:.3f}, from the pressure equation {viap:.3f}", flush=True)
        agg.setdefault("_poisson", []).append((direct, viap, chk))
    print("\nmean field NRMSE over settings:")
    for k, v in agg.items():
        if not k.startswith("_"):
            ref = agg["gappy POD (same subset)"] if "subset" in k else agg["gappy POD (final frame)"]
            print(f"  {k:<38} {np.mean(v):.3f}  ({np.mean(v) / np.mean(ref) - 1:+.1%} vs gappy POD; better in {sum(a < b for a, b in zip(v, ref))}/{len(v)} settings)")
    d, p, c = map(np.mean, zip(*agg["_poisson"]))
    print(f"  pressure NRMSE: direct {d:.3f}, from the pressure equation {p:.3f}; solver check on true fields {c:.3f}")


if __name__ == "__main__":
    main()
