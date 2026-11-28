"""Corollary 2 (rigorous any-model hallucination floor), evaluated at the Exp C operating points.
For ANY model M with H(M) <= B bits and background false-accept rate eps (fresh key independent of M):
   h >= (q/2) * U(floor(B/delta)),  q = eps(1-1/V),  delta = q(1-ln2)/(2 ln2) bits,
   U(k) = Zipf tail mass beyond rank k (exact, Hurwitz zeta).  B is taken as 32 P (float32 parameters).
Output: results/corollary2.txt"""
import json, math
from scipy.special import zeta
S = json.load(open('../results/exp_c_summary.json'))['summary']
V = 16
lines = ["Corollary 2 floor at Exp C held-out operating points (B = 32P bits, eps = measured held-out eps*)",
         "s    h    P      lam | eps*     measured h  | Cor.2 floor on h (any model with this B and eps) | ratio measured/floor"]
for key, v in sorted(S.items(), key=lambda kv: (kv[1]['s'], kv[1]['lam'], kv[1]['P'])):
    s, P, lam, eps, h = v['s'], v['P'], v['lam'], v['mean']['eps'], v['mean']['h']
    q = eps * (1 - 1 / V); delta = q * (1 - math.log(2)) / (2 * math.log(2)); B = 32 * P
    k = math.floor(B / delta); U = float(zeta(s, k + 1.0) / zeta(s))
    fl = q / 2 * U
    lines.append(f"{s:<4} {v['h']:<4} {P:<6} {lam:<3} | {eps:.4f}  {h:.2e}   | {fl:.2e} | {h/fl:.0f}")
print("\n".join(lines)); open('../results/corollary2.txt', 'w').write("\n".join(lines) + "\n")
