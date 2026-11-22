"""E1: numerical checks of the CGF identity, the three-mode closed form and the
error-floor scaling law 1 - A* ∝ h0^kappa."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from theory import (tilted_accuracy, tilted_accuracy_direct, three_mode_peak,
                    three_mode_traj)
from universe import make_universe

R = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(R, exist_ok=True)
out = {}

# (a) identity check on random universes
ts = np.linspace(0, 200, 401)
errs = []
for s in range(50):
    w, y, f, _ = make_universe(M=400, seed=s, h0_frac=0.005 * (s % 5))
    errs.append(np.max(np.abs(tilted_accuracy(f, y, w, ts) - tilted_accuracy_direct(f, y, w, ts))))
out["identity_max_abs_err"] = float(np.max(errs))

# (b) three-mode closed form vs numerical argmax on a grid
rng = np.random.default_rng(1)
gaps_t, gaps_A = [], []
tt = np.linspace(0, 400, 200001)
for _ in range(200):
    alpha = rng.uniform(0.3, 0.9); beta = rng.uniform(0.0, alpha - 0.1)
    eta = rng.uniform(alpha + 0.02, 1.0)
    c0 = rng.uniform(0.02, 0.5); h0 = 10 ** rng.uniform(-5, -2); w0 = 1 - c0 - h0
    ts_, As_, k = three_mode_peak(c0, w0, h0, alpha, beta, eta)
    if ts_ <= 0 or ts_ > 380:
        continue
    A = three_mode_traj(c0, w0, h0, alpha, beta, eta, tt)
    i = np.argmax(A)
    gaps_t.append(abs(tt[i] - ts_) / ts_); gaps_A.append(abs(A[i] - As_))
out["three_mode_n"] = len(gaps_A)
out["three_mode_max_rel_err_tstar"] = float(np.max(gaps_t))
out["three_mode_max_abs_err_Astar"] = float(np.max(gaps_A))

# (c) error-floor scaling: log(1-A*) vs log h0, slope -> kappa
c0, alpha, beta = 0.2, 0.8, 0.1
h0s = np.logspace(-8, -2, 25)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
scal = {}
for eta in [0.85, 0.9, 0.95, 1.0]:
    floor = []
    for h0 in h0s:
        _, As_, kappa = three_mode_peak(c0, 1 - c0 - h0, h0, alpha, beta, eta)
        floor.append(1 - As_)
    floor = np.array(floor)
    slope = np.polyfit(np.log(h0s[:8]), np.log(floor[:8]), 1)[0]
    scal[f"eta={eta}"] = {"kappa_pred": kappa, "slope_fit_small_h0": float(slope)}
    ax[0].loglog(h0s, floor, label=f"η={eta} (κ={kappa:.2f})")
ax[0].set_xlabel("exploit base mass $h_0$"); ax[0].set_ylabel("residual error at peak, $1-A^*$")
ax[0].set_title("Error floor scales as $h_0^{\\kappa}$"); ax[0].legend(fontsize=8)
out["scaling"] = scal

t = np.linspace(0, 120, 2000)
for h0 in [1e-2, 1e-3, 1e-4, 1e-5, 0.0]:
    if h0 == 0:
        f = np.array([alpha, beta]); yy = np.array([1, 0], bool); ww = np.array([c0, 1 - c0])
        A = tilted_accuracy(f, yy, ww, t); lab = "no exploit"
    else:
        A = three_mode_traj(c0, 1 - c0 - h0, h0, alpha, beta, 0.95, t); lab = f"$h_0$={h0:g}"
    ax[1].plot(t, A, label=lab)
ax[1].set_xlabel("tilt t = 1/λ (≈ RL training time)"); ax[1].set_ylabel("true accuracy A(t)")
ax[1].set_title("Rise-then-fall (α=.8, β=.1, η=.95, c0=.2)"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(R, "fig1_three_mode.png"), dpi=150)

# table of t*, A* for the panel
tab = {}
for h0 in [1e-2, 1e-3, 1e-4, 1e-5]:
    ts_, As_, k = three_mode_peak(c0, 1 - c0 - h0, h0, alpha, beta, 0.95)
    tab[f"{h0:g}"] = {"t_star": ts_, "A_star": As_, "kappa": k}
out["panel_table"] = tab
json.dump(out, open(os.path.join(R, "exp1.json"), "w"), indent=2)
print(json.dumps(out, indent=2))
