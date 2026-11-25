"""Why does the pressure equation not reproduce the dataset's stored pressure exactly?

reconstruct.py solves lap(p) = db/dy - div(u . grad u) with dp/dy = b at the walls.
On analytic flows that solver is accurate to 0.01%, but on The Well's true fields it
differs from the stored pressure by about 0.21 (normalised RMSE). This script solves
for the advection part and the buoyancy part separately and fits the stored pressure
as a combination of the two, for both possible orderings of the velocity components.
The fitted weights show how the dataset scales each term.

Usage: python pressure_diagnostic.py <file.hdf5>
"""
import sys

import h5py
import numpy as np

from reconstruct import poisson_pressure


def main():
    with h5py.File(sys.argv[1], "r") as h:
        b = h["t0_fields/buoyancy"][0, 40:200:8].astype(np.float64)
        p = h["t0_fields/pressure"][0, 40:200:8].astype(np.float64)
        v = h["t1_fields/velocity"][0, 40:200:8].astype(np.float64)
        x, y = h["dimensions/x"][:].astype(np.float64), h["dimensions/y"][:].astype(np.float64)
        print("attrs:", {k: (v_ if np.size(v_) < 4 else "...") for k, v_ in h.attrs.items()})
        for name in ("t0_fields/buoyancy", "t0_fields/pressure", "t1_fields/velocity"):
            print(name, {k: (str(v_)[:80]) for k, v_ in h[name].attrs.items()})
    print("x range", x[0], x[-1], "y range", y[0], y[-1], "frames", len(b))
    print("buoyancy at bottom/top wall (mean):", b[:, :, 0].mean(), b[:, :, -1].mean(), "| mean |v_y| at walls for component 1:", np.abs(v[..., 1][:, :, 0]).mean(),
          "component 0:", np.abs(v[..., 0][:, :, 0]).mean())
    zero = np.zeros_like(b[0])
    for label, (iu, iv) in (("velocity stored as (x, y)", (0, 1)), ("velocity stored as (y, x)", (1, 0))):
        adv = np.stack([poisson_pressure(zero, v[i, ..., iu], v[i, ..., iv], x, y) for i in range(len(b))])
        buo = np.stack([poisson_pressure(b[i], zero, zero, x, y) for i in range(len(b))])
        c = lambda a: a - a.mean(axis=(1, 2), keepdims=True)
        A = np.column_stack([c(adv).ravel(), c(buo).ravel()])
        t = c(p).ravel()
        coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        res = np.sqrt(np.mean((A @ coef - t) ** 2)) / t.std()
        plain = np.sqrt(np.mean((A.sum(axis=1) - t) ** 2)) / t.std()
        # also allow a term linear in y times mean buoyancy (hydrostatic reference) by removing each row's x-mean
        cx = lambda a: a - a.mean(axis=1, keepdims=True)
        A2 = np.column_stack([cx(adv).ravel(), cx(buo).ravel()])
        t2 = cx(p).ravel()
        coef2, *_ = np.linalg.lstsq(A2, t2, rcond=None)
        res2 = np.sqrt(np.mean((A2 @ coef2 - t2) ** 2)) / t2.std()
        print(f"{label}: weights 1 and 1 -> error {plain:.3f}; fitted weights advection {coef[0]:.3f}, buoyancy {coef[1]:.3f} -> error {res:.3f}; "
              f"horizontal-mean removed: weights {coef2[0]:.3f}, {coef2[1]:.3f} -> error {res2:.3f}")


if __name__ == "__main__":
    main()
