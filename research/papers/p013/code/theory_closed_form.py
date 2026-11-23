"""Closed-form continuum value of the any-model bound (Proposition 4):
L*(C) >= kappa * C^-(s-1),  kappa = ((1-1/s) G_min)^s / (zeta(s) (s-1)) * ln2^-(s-1)... (C in bits)
G(eps) = -int_0^inf ln Z'(u^-s) du,  Z'(x) = (1-eps) e^-x + eps(1-1/V) e^-(1+lam)x + eps/V.
Compares with the exact (discrete) relaxation in results/relax.json. Output results/closed_form.txt"""
import json, math, numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar
from scipy.special import zeta
def G(eps, s, H, lam):
    V = 2.0 ** H
    f = lambda u: -math.log((1 - eps) * math.exp(-u ** -s) + eps * (1 - 1 / V) * math.exp(-(1 + lam) * u ** -s) + eps / V)
    a = quad(f, 0, 1, limit=400)[0]; b = quad(f, 1, np.inf, limit=400)[0]
    return a + b
def kappa(s, H, lam):
    r = minimize_scalar(lambda le: G(10 ** le, s, H, lam), bounds=(-8, -1e-4), method='bounded', options=dict(xatol=1e-5))
    Gm = r.fun; Cn = math.log(2)          # C bits -> nats
    k = ((1 - 1 / s) * Gm) ** s / (zeta(s) * (s - 1)) * Cn ** (-(s - 1))
    return k, 10 ** r.x
R = json.load(open('../results/relax.json'))['rate']
lines = ["Prop. 4 closed form vs exact discrete relaxation (loss = 1 - J upper-bound complement), C = 1e8 and 1e10 bits",
         "s    H  lam | eps*_closed | eps*_exact(C=1e10) | loss_closed(1e8) loss_exact(1e8) | loss_closed(1e10) loss_exact(1e10)"]
for s in [1.1, 1.2, 1.5, 2.0]:
    for H in [4, 10]:
        for lam in [1, 9, 100]:
            k, e = kappa(s, H, lam)
            ex = {r['C']: r for r in R if r['s'] == s and r['H'] == H and r['lam'] == lam}
            lines.append(f"{s:<4} {H:<2} {lam:<3} | {e:.3e} | {ex[1e10]['rx_eps']:.3e} | {k*1e8**-(s-1):.4e} {ex[1e8]['rx_loss']:.4e} | "
                         f"{k*1e10**-(s-1):.4e} {ex[1e10]['rx_loss']:.4e}")
print("\n".join(lines)); open('../results/closed_form.txt', 'w').write("\n".join(lines))
