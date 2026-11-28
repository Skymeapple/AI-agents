"""Revision checks: (1) exactness of refit on ANY Markov-equivalent DAG (generalised
Prop. 3); (2) how often twins leave the training prior's support; (3) coverage of
the original single-reversal twin sampler vs. all unidentified queries.
Output: results/rev_checks.json"""
import json, os
import numpy as np
import scm as S

n = S.N_NODES
rng = np.random.default_rng(11)
out = {}
maxdiff, n_pairs = 0.0, 0
cov_out, cov_tot = 0, 0
mec_out, mec_tot = 0, 0
unid_adj, unid_nonadj = 0, 0
for _ in range(20000):
    m = S.random_scm(rng)
    Sig = S.implied_cov(m["W"], m["s"])
    mem = S.mec_members(m["A"])
    for B in mem:
        m2 = S.refit_on(m, B)
        maxdiff = max(maxdiff, float(np.abs(S.implied_cov(m2["W"], m2["s"]) - Sig).max()))
        n_pairs += 1
        if not np.array_equal(B, m["A"]):
            mec_tot += 1; mec_out += (not S.in_prior(m2))
    for u, v in S.covered_edges(m["A"]):
        cov_tot += 1; cov_out += (not S.in_prior(S.twin_by_reversal(m, u, v)))
    Rs = np.stack([S.ancestors_matrix(B) for B in mem]); frac = Rs.mean(0)
    for a in range(n):
        for b in range(n):
            if a != b and 0 < frac[a, b] < 1:
                if m["A"][a, b] or m["A"][b, a]:
                    unid_adj += 1
                else:
                    unid_nonadj += 1
out["refit_any_mec_member_max_abs_cov_diff"] = maxdiff
out["refit_pairs_checked"] = n_pairs
out["covered_twins_out_of_prior_frac"] = cov_out / cov_tot
out["covered_twins_checked"] = cov_tot
out["mec_member_twins_out_of_prior_frac"] = mec_out / mec_tot
out["mec_member_twins_checked"] = mec_tot
out["unidentified_queries_nonadjacent_frac_under_prior"] = unid_nonadj / (unid_adj + unid_nonadj)
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "rev_checks.json")
json.dump(out, open(path, "w"), indent=1)
print(json.dumps(out, indent=1))
