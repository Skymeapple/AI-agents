"""Numerical verification of the computational items adjudicated AUDIT_CORRECT / AMBIGUOUS (CODATA-style constants)."""
import numpy as np, json, os
from common import RES
muN = 5.0507837e-27; hbar = 1.054571817e-34
g_14N = 0.40376; g_2H = 0.857438
gam_e = 1.761e11; gam_H = 2.6752e8; gam_13C = 6.7283e7
out = {}
out['680_14N_mu_sqrtII1'] = g_14N * muN * np.sqrt(2)          # item value 2.884e-27
out['680_2H_mu_sqrtII1'] = g_2H * muN * np.sqrt(2)
out['680_14N_mu_gI'] = g_14N * muN * 1
out['682_pe_over_pH'] = gam_e / gam_H                          # options 820 / 658 / 329
out['687_B1_mT'] = (np.pi / 2) / (gam_13C * 1e-6) * 1e3        # options 3.72 / 5.18 / 17.0 / 23.3
out['691_flip_angle_over_2pi'] = gam_H * 4.697e-3 * 5e-6 / (2 * np.pi)
out['692_mu_gamma_hbar_I'] = 1.569e8 * hbar * 0.5
out['692_mu_gamma_hbar_sqrtII1'] = 1.569e8 * hbar * np.sqrt(0.75)
out['697_lines_J_ll_a'] = 2 * 2 * 1 + 1; out['697_lines_J_gg_a'] = 2 * 4 * 1 + 1
json.dump({k: float(v) for k, v in out.items()}, open(os.path.join(RES, 'item_verification.json'), 'w'), indent=1)
for k, v in out.items(): print(k, v)
