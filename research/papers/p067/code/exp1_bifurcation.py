"""Exp 1: equilibrium/bifurcation diagram vs AI error eps, hysteresis loop,
team-error 'improvement paradox', and critical slowing (eigenvalue at fold)."""
import json, numpy as np
from model import *
from plotstyle import *

P = Params()
out = {"params": P.__dict__.copy()}
out["params"]["beta"] = "inf"
eps_grid = np.linspace(0.02, 0.65, 316)

# closed forms
ec, er = eps_collapse(P), eps_recover(P)
sH, sL = s_high(P), s_low(P)
eb = h(sH, P) / (1 - P.rho * sL)
jump = P.dc + ec * P.rho * (sH - sL)
out["closed_form"] = dict(s_H=sH, s_L=sL, eps_collapse=ec, eps_recover=er,
                          hysteresis_width=er - ec, eps_valley_lower=eb,
                          error_jump_at_collapse=jump,
                          err_skilled=h(sH, P), err_deskilled_at_collapse=a(sL, ec, P),
                          outage_error_gap=h(sL, P) - h(sH, P))

# numeric equilibria step + smooth
def eq_table(Pb):
    rows = []
    for e in eps_grid:
        for s, st in equilibria(e, Pb):
            rows.append((e, s, bool(st)))
    return rows

betas = [np.inf, 100, 40]
tables = {str(b): eq_table(P.with_(beta=b)) for b in betas}

# numeric bistable windows
win = {}
for b in betas:
    Pb = P.with_(beta=b)
    bi = [e for e in eps_grid if sum(st for _, st in equilibria(e, Pb)) >= 2]
    win[str(b)] = [min(bi), max(bi)] if bi else None
out["numeric_bistable_window"] = win

# hysteresis loop via slow sweep (smooth beta=100)
Pb = P.with_(beta=100)
T = 6000
dt = 0.05
n = int(T / dt)
down = np.linspace(0.65, 0.02, n)
up = down[::-1]
sd = simulate(0.9, down, Pb, dt)
su = simulate(sd[-1], up, Pb, dt)
# where jumps happen
jd = down[np.argmax(np.diff(sd) < -1e-4)] if np.any(np.diff(sd) < -1e-4) else None
ju = up[np.argmax(np.diff(su) > 1e-4)] if np.any(np.diff(su) > 1e-4) else None
out["sweep_beta100"] = dict(collapse_eps=float(jd), recover_eps=float(ju))

# eigenvalue (relaxation rate) of skilled branch as eps -> fold, smooth
eig = []
for b in [100, 40]:
    Pb = P.with_(beta=b)
    rows = []
    for e in np.linspace(0.6, 0.0, 601):
        eqs = [s for s, st in equilibria(e, Pb) if st and s > 0.75]
        if not eqs:
            break
        s0 = max(eqs)
        lam = (drift(s0 + 1e-6, e, Pb) - drift(s0 - 1e-6, e, Pb)) / 2e-6
        rows.append((e, s0, lam))
    eig.append((b, rows))
out["relaxation_rate"] = {str(b): dict(far=rows[0][2], near_fold=rows[-1][2], eps_last=rows[-1][0])
                          for b, rows in eig}

# team error along branches (step)
ers = []
for e in eps_grid:
    eqs = equilibria(e, P)
    ers.append([(s, team_error(s, e, P)) for s, st in eqs if st])

json.dump(out, open("../results/exp1_bifurcation.json", "w"), indent=2, default=float)

# ---------- figure 1 ----------
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
for b, col, lab in [(np.inf, INK, "step (beta = inf)"), (100, C1, "smooth, beta = 100"), (40, C2, "smooth, beta = 40")]:
    rows = tables[str(b)]
    st = np.array([(e, s) for e, s, k in rows if k])
    un = np.array([(e, s) for e, s, k in rows if not k]) if any(not k for _, _, k in rows) else None
    ax[0].plot(st[:, 0], st[:, 1], ".", ms=2.5, color=col, label=lab + " (stable)")
    if un is not None and len(un):
        ax[0].plot(un[:, 0], un[:, 1], "x", ms=2.5, color=col, alpha=0.6)
ax[0].axvline(ec, color=INK2, lw=0.8, ls=":"); ax[0].axvline(er, color=INK2, lw=0.8, ls=":")
ax[0].text(ec, 0.62, " eps_c", color=INK2, fontsize=8); ax[0].text(er, 0.62, " eps_r", color=INK2, fontsize=8)
ax[0].set_xlabel("AI error rate eps (lower = better AI)"); ax[0].set_ylabel("equilibrium unaided skill s*")
ax[0].set_title("(a) Equilibria (x = unstable)"); ax[0].invert_xaxis(); ax[0].legend(fontsize=7, loc="lower left")

ax[1].plot(down, sd, color=C1, label="AI improving (eps falls)")
ax[1].plot(up, su, color=C2, label="AI degrading (eps rises)")
ax[1].set_xlabel("AI error rate eps"); ax[1].set_ylabel("skill s(t)")
ax[1].set_title("(b) Hysteresis under slow sweep (beta=100)"); ax[1].invert_xaxis(); ax[1].legend(fontsize=8)

E = []
for e, lst in zip(eps_grid, ers):
    for s, err in lst:
        E.append((e, err, s > 0.75))
E = np.array(E)
m = E[:, 2] == 1
ax[2].plot(E[m, 0], E[m, 1], ".", ms=3, color=C3, label="skilled branch")
ax[2].plot(E[~m, 0], E[~m, 1], ".", ms=3, color=C2, label="deskilled branch")
ax[2].axhline(h(sH, P), color=INK2, lw=0.8, ls="--")
ax[2].text(0.6, h(sH, P) + 0.004, "unaided skilled error", fontsize=7, color=INK2)
ax[2].set_xlabel("AI error rate eps"); ax[2].set_ylabel("equilibrium task error")
ax[2].set_title("(c) Team error: improvement paradox (step)"); ax[2].invert_xaxis(); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig("../results/fig1_bifurcation.png")

fig, ax = plt.subplots(figsize=(5, 3.4))
for (b, rows), col in zip(eig, [C1, C2]):
    rows = np.array(rows)
    ax.plot(rows[:, 0], -rows[:, 2], color=col, label=f"beta = {b}")
ax.set_xlabel("AI error rate eps"); ax.set_ylabel("recovery rate -f'(s*) (1/month)")
ax.set_title("Critical slowing on skilled branch"); ax.invert_xaxis(); ax.legend()
plt.tight_layout(); plt.savefig("../results/fig2_critical_slowing.png")
print(json.dumps(out, indent=1, default=float))
