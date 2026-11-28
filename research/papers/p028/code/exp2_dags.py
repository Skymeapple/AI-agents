"""Exp 2: real module-dependency DAGs from installed Python packages.

(a) structural statistics of real DAGs vs. size-matched random DAGs;
(b) project success / cost under uniform vs. structure-aware allocation of a
    fixed verification budget;
(c) sweep over integration-test exposure delta and blind spot b.
"""
import json, os, sys, time
import numpy as np
import networkx as nx
from multiprocessing import Pool
from graphs import load_all
from dagsim import DagProject, simulate, waterfill
from model import latent_defect_rate

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
R = int(os.environ.get("R", 4000))
R_SWEEP = int(os.environ.get("R_SWEEP", 2000))
BASE = dict(p=0.9, b=0.02, delta=0.3, kappa=0.3, rho=0.5, a_acc=0.5, K=10)
BETA = 1.0
C0 = 0.9
E0 = -np.log(1 - C0) / BETA


def gini(x):
    x = np.sort(np.asarray(x, float))
    if x.sum() == 0: return 0.0
    n = len(x); return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def random_dag(N, E, seed):
    rng = np.random.default_rng(seed)
    D = nx.DiGraph(); D.add_nodes_from(range(N))
    allp = N * (N - 1) // 2
    picks = rng.choice(allp, size=min(E, allp), replace=False)
    # map linear index -> (i<j)
    i = (N - 2 - np.floor(np.sqrt(-8 * picks + 4 * N * (N - 1) - 7) / 2 - 0.5)).astype(int)
    j = (picks + i + 1 - N * (N - 1) // 2 + (N - i) * ((N - i) - 1) // 2).astype(int)
    D.add_edges_from(zip(i.tolist(), j.tolist()))
    return D


def stats(P):
    return dict(N=P.N, E=int(sum(len(x) for x in P.parents)),
                frac_sinks=float((P.outdeg == 0).mean()),
                frac_sources=float(np.mean([len(x) == 0 for x in P.parents])),
                max_depth=int(P.depth.max()), mean_reach=float(P.reach.mean()),
                max_reach=int(P.reach.max()), gini_reach=gini(P.reach))


def neff_frac(P, kw=BASE):
    """N_eff / N of Corollary 4 at uniform coverage C0."""
    q = latent_defect_rate(kw["p"], C0, kw["b"])
    return float(np.mean((1 - kw["delta"] * (1 - q)) ** P.outdeg * (1 - kw["a_acc"] * (1 - q))))


def policies(P, kw):
    N = P.N; B = N * E0
    surv = (1 - kw["delta"]) ** P.outdeg * (1 - kw["a_acc"])
    return {
        "uniform": np.full(N, E0),
        "reach": waterfill(1.0 + P.reach, B, BETA, kw["b"]),
        "sink-aware": waterfill(surv, B, BETA, kw["b"]),
        "combined": waterfill(surv * N + kw["rho"] * P.reach / N, B, BETA, kw["b"]),
    }


def residual(P, c, kw):
    """Proposition 4: per-module residual latent-defect probability (exact for kappa=0)."""
    q = latent_defect_rate(kw["p"], c, kw["b"])
    return q * (1 - kw["delta"] * (1 - q)) ** P.outdeg * (1 - kw["a_acc"] * (1 - q))


def job(args):
    name, P, pol, e, kw, seed = args
    c = 1 - np.exp(-BETA * e)
    r_ = R_SWEEP if "sweep" in name else R
    S, C, L = simulate(P, r_, seed, cvec=c, **kw)
    r = residual(P, c, kw)
    return dict(graph=name, policy=pol, R=r_, **{k: v for k, v in kw.items()},
                success=float(S.mean()), success_se=float(S.std() / np.sqrt(r_)),
                cost=float(C.mean()), cost_se=float(C.std() / np.sqrt(r_)),
                latent=float(L.mean()),
                success_prop4=float(np.prod(1 - r)), latent_prop4=float(r.sum()))


def main():
    t0 = time.time()
    graphs = load_all()
    projs = {k: DagProject(D) for k, (G, D, s) in graphs.items()}
    struct = {}
    for k, P in projs.items():
        st = stats(P)
        st["neff_frac"] = neff_frac(P)
        st["scc_max"] = int(max(graphs[k][2])); st["files"] = graphs[k][0].number_of_nodes()
        nullP = [DagProject(random_dag(P.N, st["E"], s)) for s in range(5)]
        nulls = [dict(stats(Pn), neff_frac=neff_frac(Pn)) for Pn in nullP]
        st["null"] = {f: float(np.mean([n[f] for n in nulls])) for f in
                      ["frac_sinks", "frac_sources", "max_depth", "mean_reach", "gini_reach", "neff_frac"]}
        struct[k] = st
    json.dump(struct, open(os.path.join(OUT, "exp2_structure.json"), "w"), indent=1)
    print("structure done", time.time() - t0, flush=True)

    jobs = []
    seed = 1000
    # (b) main configuration, all graphs, all policies; plus random-DAG null under uniform
    for k, P in projs.items():
        for pol, e in policies(P, BASE).items():
            jobs.append((k, P, pol, e, BASE, seed)); seed += 1
        Pn = DagProject(random_dag(P.N, struct[k]["E"], 99))
        jobs.append((k + "|null", Pn, "uniform", np.full(P.N, E0), BASE, seed)); seed += 1
        K0 = dict(BASE, kappa=0.0)
        jobs.append((k + "|kappa0", P, "uniform", np.full(P.N, E0), K0, seed)); seed += 1
        jobs.append((k + "|kappa0", P, "sink-aware", policies(P, K0)["sink-aware"], K0, seed)); seed += 1
    # (c) sweeps on three graphs
    for k in ["numpy", "scipy", "setuptools"]:
        P = projs[k]
        for delta in [0.0, 0.1, 0.3, 0.6, 0.9]:
            for b in [0.0, 0.02, 0.1]:
                kw = dict(BASE, delta=delta, b=b)
                for pol, e in policies(P, kw).items():
                    if pol in ("uniform", "sink-aware", "reach"):
                        jobs.append((k + "|sweep", P, pol, e, kw, seed)); seed += 1
    print("jobs", len(jobs), flush=True)
    with Pool(2) as pool:
        res = pool.map(job, jobs, chunksize=1)
    json.dump(res, open(os.path.join(OUT, "exp2_sim.json"), "w"), indent=1)
    # chain-theory reference for the main config (no integration testing):
    q = latent_defect_rate(BASE["p"], C0, BASE["b"])
    ref = {k: float((1 - q) ** P.N) for k, P in projs.items()}
    import scipy, matplotlib, sympy, pandas, setuptools, pip, platform
    vers = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                matplotlib=matplotlib.__version__, networkx=nx.__version__, sympy=sympy.__version__,
                pandas=pandas.__version__, setuptools=setuptools.__version__, pip=pip.__version__)
    json.dump(dict(q=float(q), versions=vers, chain_success=ref), open(os.path.join(OUT, "exp2_chainref.json"), "w"), indent=1)
    print("done", time.time() - t0)


if __name__ == "__main__":
    main()
