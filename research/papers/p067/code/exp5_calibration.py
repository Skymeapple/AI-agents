"""Exp 5: consistency checks of the skill-decay mechanism against published effect sizes.
(i) Budzyn et al. 2025: unassisted ADR 28.4% -> 22.4% within ~3 months after AI introduction.
    Question: can practice-driven skill decay (this model) produce a ratio of 0.789 within 3 months,
    given forgetting half-lives from the 2025 Psychological Bulletin meta-analysis (6.5-13 months)?
(ii) Shen & Tamkin 2026 (quiz 50% vs 67%) and Bastani et al. 2025 (-17% exam): implied kappa.
Assumption A1: the outcome measure is proportional to skill s. All quantities illustrative."""
import json, numpy as np
from model import LN2
from plotstyle import *

R_obs = 145 / 648 / (226 / 795)   # = 0.787 from published counts
T = 3.0                            # months of exposure


def ratio(hl, u, kappa, sH=0.9, T=T, average=True):
    delta = LN2 / hl
    alpha = delta * sH / (1 - sH)
    p = u + kappa * (1 - u)
    sp = alpha * p / (alpha * p + delta)
    lam = alpha * p + delta
    if average:   # mean of s(t) over [0,T] (ADR measured over the whole 3-month window)
        m = sp + (sH - sp) * (1 - np.exp(-lam * T)) / (lam * T)
    else:
        m = sp + (sH - sp) * np.exp(-lam * T)
    return m / sH


out = {"R_obs": R_obs}
hls = np.linspace(1, 14, 131)
us = np.linspace(0, 1, 101)
for sH in [0.8, 0.9, 0.95]:
    for avg in [True, False]:
        # minimum achievable ratio with zero practice (u=0,kappa=0)
        minr = np.array([ratio(h_, 0, 0, sH, average=avg) for h_ in hls])
        feas = hls[minr <= R_obs]
        out[f"max_halflife_months_consistent[sH={sH},{'avg' if avg else 'end'}]"] = float(feas.max()) if len(feas) else None
# with observed unassisted share ~0.47 (648 non-AI vs 734 AI-assisted exams post-exposure; assumption)
u_obs = 648 / (648 + 734)
out["u_obs_assumed"] = u_obs
for hl in [3.0, 6.5, 13.0]:
    out[f"ratio_at_u_obs_k0[hl={hl}]"] = float(ratio(hl, u_obs, 0.0))
    out[f"asymptotic_ratio_at_u_obs_k0[hl={hl}]"] = float(ratio(hl, u_obs, 0.0, T=1e6, average=False))
# required half-life given u_obs, kappa=0
req = [h_ for h_ in np.linspace(0.05, 14, 2800) if ratio(h_, u_obs, 0.0) <= R_obs]
out["max_halflife_consistent_at_u_obs_k0"] = float(max(req)) if req else None

# Shen & Tamkin; Bastani: implied kappa vs baseline fraction b
bs = np.linspace(0, 0.6, 7)
out["implied_kappa_shen_tamkin"] = {f"{b:.1f}": float((50 / 67 - b) / (1 - b)) for b in bs}
out["implied_kappa_bastani"] = {f"{b:.1f}": float((0.83 - b) / (1 - b)) for b in bs}
json.dump(out, open("../results/exp5_calibration.json", "w"), indent=2)
print(json.dumps(out, indent=1))

H, U = np.meshgrid(hls, us)
Z = np.vectorize(lambda h_, u_: ratio(h_, u_, 0.0))(H, U)
fig, ax = plt.subplots(figsize=(5.4, 3.8))
cs = ax.contourf(H, U, Z, levels=np.linspace(0.6, 1.0, 9), cmap="Blues_r")
ax.contour(H, U, Z, levels=[R_obs], colors=[C2], linewidths=2)
ax.axhline(u_obs, color=INK, ls="--", lw=1); ax.text(9, u_obs + 0.02, "assumed unassisted share", fontsize=7)
ax.axvline(6.5, color=INK2, ls=":", lw=1); ax.axvline(13, color=INK2, ls=":", lw=1)
ax.text(6.6, 0.9, "6.5 mo", fontsize=7, color=INK2); ax.text(12.0, 0.9, "13 mo", fontsize=7, color=INK2)
plt.colorbar(cs, label="predicted 3-month mean skill ratio")
ax.set_xlabel("forgetting half-life (months)"); ax.set_ylabel("unassisted share u (kappa = 0)")
ax.set_title("Can skill decay explain a 21% ADR drop in 3 months?\n(orange: observed ratio 0.79; region below/left = enough decay)", fontsize=9)
plt.tight_layout(); plt.savefig("../results/fig7_calibration.png")
