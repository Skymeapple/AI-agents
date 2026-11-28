"""Small descriptive statistics quoted in the paper text. -> ../results/extras.json   Usage: python3 extras.py"""
import os, json
import numpy as np
import analyze_lib as A
out = {}
for a in (1.2, 1.5, 1.8):
    p = A.load(f"E1_a{a}_s0")["p"]
    out[f"freq_alpha{a}"] = dict(p_min=float(p.min()), p_max=float(p.max()), decades=float(np.log10(p.max() / p.min())),
                                 decades_spanned_by_rarest_two_thirds=float(np.log10(p[len(p) // 3] / p.min())))
for n in ["E1_a1.5_s0", "E3_lr0.003", "E3_lr0.0003", "E5_adam_b20.99", "E5_adam_b20.9999", "E5_sgd_lr0.1", "E5_sgd_lr0.3"]:
    d = A.load(n); T = d["T"].astype(float); rec = d["maxcos"] >= A.THR; p = d["p"]
    first = np.where(rec.any(0), rec.argmax(0), -1); ok = first > 0
    Trec = np.where(ok, np.sqrt(T[np.maximum(first, 1) - 1] * T[np.maximum(first, 1)]), np.nan)
    occ = p * Trec; third = len(p) // 3
    out[f"recovery_{n}"] = dict(frac_recovered_then_lost=float((rec.any(0) & ~rec[-1]).mean()),
                                median_occurrences_at_recovery_top8=float(np.nanmedian(occ[:8])),
                                median_occurrences_at_recovery_most_frequent_third=float(np.nanmedian(occ[:third])),
                                median_occurrences_at_recovery_rarest_third=float(np.nanmedian(occ[-third:])))
d = A.load("E4_U100000"); u = d["uniq_counts"]; rec = d["maxcos"][-1] >= A.THR
out["replay_buffer_100k"] = dict(min_unique_occurrences=int(u.min()), n_features_under_50_unique=int((u < 50).sum()),
                                 recovered_frac_under_50_unique=float(rec[u < 50].mean()), epochs=float(d["T"][-1] / 1e5))
json.dump(out, open(os.path.join(A.R, "extras.json"), "w"), indent=1); print(json.dumps(out, indent=1))
