"""Alignment tolerance: largest misalignment angle theta_c at which a unified
model (capacity r=30 < k_s + M k_p + (M-1) k_s = 36) still beats specialists.
Prediction (df model): sin^2 theta_c = G / ((M-1) k_s s^2 / M), with
G = (sigma^2/n) [k(d+q-k) - rho(d+Mq-rho)/M], rho = k_s + M k_p.
Run: OMP_NUM_THREADS=1 python3 exp_gap.py"""
import json, os
import numpy as np
from common import make_task, sim_losses, theory_losses
OUT = os.path.join(os.path.dirname(__file__), "..", "results")
d, q, ks, kp, M, r = 64, 16, 4, 4, 4, 30
rng = np.random.default_rng(5)
k, rho = ks + kp, ks + M * kp
thetas = np.radians(np.linspace(0, 20, 41))
rows = []
for n in [300, 1000, 3000, 10000, 30000]:
    G = (k * (d + q - k) - rho * (d + M * q - rho) / M) / n
    pred = np.degrees(np.arcsin(np.sqrt(min(1, G / ((M - 1) * ks / M)))))
    sim_sign, th_sign = [], []
    for th in thetas:
        U, S = [], []
        for rep in range(16):
            Ws = make_task(d, q, ks, kp, M, theta=th, rng=rng.integers(1 << 31))
            lu, ls = sim_losses(Ws, 1.0, n, r, rng); U.append(lu); S.append(ls)
        tu, ts = theory_losses(Ws, 1.0, n, r)
        sim_sign.append(float(np.mean(U) / np.mean(S))); th_sign.append(float(tu / ts))
    def first_cross(v):
        v = np.array(v); idx = np.where(v >= 1)[0]
        if len(idx) == 0: return None
        i = idx[0]
        if i == 0: return 0.0
        # linear interpolation in degrees
        a, b = np.degrees(thetas[i - 1]), np.degrees(thetas[i])
        return float(a + (1 - v[i - 1]) / (v[i] - v[i - 1]) * (b - a))
    row = dict(n=n, theta_c_closed_form=float(pred), theta_c_theory=first_cross(th_sign),
               theta_c_sim=first_cross(sim_sign))
    rows.append(row); print(row, flush=True)
json.dump(dict(d=d, q=q, ks=ks, kp=kp, M=M, r=r, rows=rows), open(os.path.join(OUT, "gap_results.json"), "w"), indent=1)
