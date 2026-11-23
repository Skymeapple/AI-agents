"""Numerical checks of the theoretical statements (Prop. 2 completeness, exactness
of Gaussian twins) and descriptive statistics of the twin construction."""
import json, os, collections
import numpy as np
import scm as S

out = {}
n = S.N_NODES
dags = S.all_dags(n)
key_of = [S.mec_key(A) for A in dags]
idx = {A.tobytes(): k for k, A in enumerate(dags)}

# --- Completeness: every non-identified ancestral query has a twin reachable by
# covered-edge reversals; record the minimal number of reversals needed.
def neighbours(A):
    for u, v in S.covered_edges(A):
        B = A.copy(); B[u, v] = 0; B[v, u] = 1
        yield B

min_rev = collections.Counter()
n_unident, n_fail = 0, 0
for A in dags:
    RA = S.ancestors_matrix(A)
    # BFS over the MEC via covered reversals
    dist = {A.tobytes(): 0}; frontier = [A]; reach = [(A, 0)]
    while frontier:
        nxt = []
        for B in frontier:
            for C in neighbours(B):
                if C.tobytes() not in dist:
                    dist[C.tobytes()] = dist[B.tobytes()] + 1
                    nxt.append(C); reach.append((C, dist[C.tobytes()]))
        frontier = nxt
    mec_members = [B for B, k in zip(dags, key_of) if k == S.mec_key(A)]
    assert len(reach) == len(mec_members), "covered reversals must span the MEC"
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            vals = {S.ancestors_matrix(B)[i, j] for B in mec_members}
            if len(vals) == 2:
                n_unident += 1
                ds = [d for B, d in reach if S.ancestors_matrix(B)[i, j] != RA[i, j]]
                if not ds:
                    n_fail += 1
                else:
                    min_rev[min(ds)] += 1
out["n_dags"] = len(dags)
out["n_mecs"] = len(set(key_of))
out["unidentified_dag_query_pairs"] = n_unident
out["unidentified_without_twin"] = n_fail
out["min_reversals_hist"] = dict(sorted(min_rev.items()))

# --- Exactness of Gaussian twins and selection statistics under the SCM prior
rng = np.random.default_rng(7)
maxdiff, has_cov, n_tot = 0.0, 0, 20000
frac_ident = []
mect = S.MECTable()
for _ in range(n_tot):
    m = S.random_scm(rng)
    ce = S.covered_edges(m["A"])
    for i in range(n):
        for j in range(n):
            if i != j:
                frac_ident.append(mect.query(m["A"], i, j)[0])
    if ce:
        has_cov += 1
        u, v = ce[rng.integers(len(ce))]
        m2 = S.twin_by_reversal(m, u, v)
        d = np.abs(S.implied_cov(m["W"], m["s"]) - S.implied_cov(m2["W"], m2["s"])).max()
        maxdiff = max(maxdiff, float(d))
out["frac_scms_with_covered_edge"] = has_cov / n_tot
out["max_abs_cov_diff_twins"] = maxdiff
out["frac_ancestral_queries_identified_uniform_pairs"] = float(np.mean(frac_ident))

os.makedirs("../results", exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), "..", "results", "theory_checks.json"), "w") as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))
