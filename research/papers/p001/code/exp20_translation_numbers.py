"""Exp 20 (revision 2): small analytic quantities quoted in the text (Theorem 1 applied to matched A/B
contrasts and to the break model), on the 240-run Chinchilla refit."""
import json
import numpy as np
from scipy.stats import norm
from common import RES
import horizon as H
P = json.load(open(f"{RES}/exp1_fit.json"))["fit240"]
TH = np.array([np.log(P["A"]), np.log(P["B"]), np.log(P["E"]), P["alpha"], P["beta"]])
g = H.gam(TH); out = dict(gamma=g)
out["rho"] = {f"{c:.0e}": float(H.rho(TH, c)) for c in [1e21, 1e23, 1e25, 1e26, 5e26, 1e27, 1e28, 1e30]}
out["log_signal_of_2x_lever"] = {f"{c:.0e}": float(np.log1p(H.rho(TH, c) * (2 ** g - 1))) for c in [1e21, 1e23, 1e25, 5e26, 1e28]}
out["break_to_penalty_two_decades"] = {str(d): float(100 ** (d / g)) for d in [0.01, 0.02, 0.05, g / 2]}
out["AB_Pmin50_at_5e26"] = {str(s): float(H.p_min(TH, 5e26, s)) for s in [1e-4, 3e-4, 1e-3, 3e-3]}
out["AB_Pmin80_at_5e26"] = {str(s): float(H.p_min(TH, 5e26, s, power=0.8)) for s in [1e-4, 3e-4, 1e-3, 3e-3]}
json.dump(out, open(f"{RES}/exp20_numbers.json", "w"), indent=2); print(json.dumps(out, indent=1))
