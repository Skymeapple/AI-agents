"""Arithmetic on figures quoted from cited sources (no experiments). -> ../results/sourced_checks.json
Sources: Templeton et al. 2024 (dictionary sizes; dead fractions 'roughly 2%, 35%, 65%');
Gao et al. 2024/2025 (16M-latent SAE trained on 40B tokens)."""
import json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
M = np.array([1_048_576, 4_194_304, 33_554_432]); dead = np.array([0.02, 0.35, 0.65]); alive = M * (1 - dead)
out = dict(templeton_alive=alive.tolist(),
           alive_vs_width_exponent=dict(m1_to_m4=float(np.log(alive[1] / alive[0]) / np.log(M[1] / M[0])),
                                        m4_to_m34=float(np.log(alive[2] / alive[1]) / np.log(M[2] / M[1])),
                                        m1_to_m34=float(np.log(alive[2] / alive[0]) / np.log(M[2] / M[0]))))
try:
    mtoy = json.load(open(os.path.join(R, "extrapolation.json")))["mstar_toy_geomean"]
except Exception:
    mtoy = None
T = 40e9; n = 16e6
out["gao_16M"] = dict(tokens=T, latents=n, tokens_per_latent=T / n,
                      min_learnable_frequency_beta1={f"mstar={m:g}": m / T for m in (100, mtoy, 1000, 10000) if m},
                      tokens_between_occurrences_at_threshold={f"mstar={m:g}": T / m for m in (100, mtoy, 1000, 10000) if m})
json.dump(out, open(os.path.join(R, "sourced_checks.json"), "w"), indent=1); print(json.dumps(out, indent=1))
