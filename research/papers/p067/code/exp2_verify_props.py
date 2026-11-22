"""Exp 2: Monte-Carlo verification of the closed-form propositions on random
parameter draws (step model), plus convergence of smooth windows to step as beta grows,
plus the practice-floor / engagement substitution frontier."""
import json, numpy as np
from model import *
from plotstyle import *

rng = np.random.default_rng(0)
N = 400
res = dict(n_draws=0, prop1_bistable_match=0, prop1_checked=0, prop3_jump_err=[], prop4_floor_ok=0,
           prop4_floor_checked=0, prop4_floor_minimal=0)
for _ in range(N):
    hl = rng.uniform(3, 13)
    P = Params(delta=LN2 / hl, alpha=rng.uniform(0.3, 2.0), kappa=rng.uniform(0, 0.6),
               h0=rng.uniform(0, 0.1), h1=rng.uniform(0.3, 0.8), rho=rng.uniform(0.1, 0.8),
               dc=rng.uniform(0.005, 0.08))
    if not condition_A(P) or s_low(P) >= s_high(P) - 1e-3:
        continue
    res["n_draws"] += 1
    ec, er = eps_collapse(P), eps_recover(P)
    # Prop 1: bistable iff ec <= eps < er  (check at random eps away from boundaries)
    for e in rng.uniform(0, 0.9, 5):
        if min(abs(e - ec), abs(e - er)) < 1e-3 or P.h1 - e * P.rho <= 0:
            continue
        nst = sum(st for _, st in equilibria(e, P, n=4001))
        pred = 2 if ec <= e < er else 1
        res["prop1_checked"] += 1
        res["prop1_bistable_match"] += int(nst == pred)
    # Prop 3: error jump at collapse equals dc + ec*rho*(sH-sL)
    sH, sL = s_high(P), s_low(P)
    if ec > 0:
        num = a(sL, ec, P) - h(sH, P)
        res["prop3_jump_err"].append(abs(num - (P.dc + ec * P.rho * (sH - sL))))
    # Prop 4: floor removes low eq, and a slightly smaller floor does not
    e = rng.uniform(max(ec, 0), er) if er > max(ec, 0) else None
    if e is not None:
        u = u_floor_star(e, P)
        if np.isfinite(u) and 0.02 < u < 0.98:
            res["prop4_floor_checked"] += 1
            eqs_hi = equilibria(e, P.with_(u_min=min(u + 1e-3, 1)), n=4001)
            eqs_lo = equilibria(e, P.with_(u_min=max(u - 0.02, 0)), n=4001)
            res["prop4_floor_ok"] += int(all(s > theta(e, P) - 1e-6 for s, st in eqs_hi if st))
            res["prop4_floor_minimal"] += int(any(s < theta(e, P) for s, st in eqs_lo if st))
res["prop3_max_abs_err"] = float(max(res["prop3_jump_err"]))
del res["prop3_jump_err"]

# convergence of bistable window with beta
P = Params()
grid = np.linspace(0.2, 0.55, 701)
conv = {}
for b in [20, 40, 100, 300, 1000, 3000, np.inf]:
    Pb = P.with_(beta=b)
    bi = [e for e in grid if sum(st for _, st in equilibria(e, Pb, n=8001)) >= 2]
    conv[str(b)] = [float(min(bi)), float(max(bi))] if bi else None
res["window_vs_beta"] = conv
res["closed_form_window"] = [eps_collapse(P), eps_recover(P)]

# practice floor needed for a skill target, vs kappa (Prop 4b)
ks = np.linspace(0, 0.6, 61)
front = {}
for sb in [0.70, 0.80, 0.85]:
    front[str(sb)] = [u_for_target(sb, P.with_(kappa=k)) for k in ks]
res["floor_frontier_kappa"] = list(ks)
res["floor_frontier"] = front
res["u_star_default"] = {str(e): u_floor_star(e, P) for e in [0.26, 0.30, 0.35, 0.40, 0.45]}
res["u_target_default"] = {str(sb): u_for_target(sb, P) for sb in [0.7, 0.8, 0.85]}
json.dump(res, open("../results/exp2_verify_props.json", "w"), indent=2, default=float)

fig, ax = plt.subplots(figsize=(5, 3.4))
for (e, ys), col in zip(front.items(), [C1, C2, C3]):
    ax.plot(ks, ys, color=col, label=f"skill target = {e}")
ax.set_xlabel("kappa: learning retained on AI-assisted tasks")
ax.set_ylabel("minimal unaided-practice floor u*")
ax.set_title("Unaided-practice floor vs. engagement design")
ax.legend(); plt.tight_layout(); plt.savefig("../results/fig3_practice_floor.png")
print(json.dumps({k: v for k, v in res.items() if k not in ("floor_frontier", "floor_frontier_kappa")}, indent=1, default=float))
