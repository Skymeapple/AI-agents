"""(1) Approval-fatigue model instantiated on AgentDojo permission structure.
(2) Out-of-sample test of static permission profiles (profiles designed on half
    of the user tasks, evaluated on the held-out half).

Model (Section 5 of the paper). A static base grant G is given to the agent;
any tool outside G requires a human approval prompt. For a hijacked agent the
human denies an out-of-grant request with probability d(n), where n is the
mean number of approval prompts per benign task (approval load) and
d(n) = d0 * exp(-n / n0)   (habituation; n0 = infinity means no fatigue).
Conditional attack feasibility given an injection attempt that the model
follows:   r(G) = s(G) + (1 - s(G)) * (1 - d(n(G)))
where s(G) is the fraction of security cases feasible inside G.
The parametric form and its constants are assumptions; we report sweeps.
"""
import json, os, itertools
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
D = json.load(open(os.path.join(RES, "agentdojo_structure.json")))
rng = np.random.default_rng(20261122)


def req(u):
    return frozenset(c["tool"] for c in u["calls"])


def feas(alts, G):
    return any(frozenset(c["tool"] for c in a) <= G for a in alts)


def suite_arrays(S):
    U = list(S["user_tasks"].values()); I = list(S["injection_tasks"].values())
    return [req(u) for u in U], [i["alternatives"] for i in I], frozenset(S["tools"])


def s_of(G, Rs, Is):
    return np.mean([feas(a, G) for _ in Rs for a in Is])


def load_of(G, Rs):
    return np.mean([len(R - G) for R in Rs])


def nested_grants(Rs, Is, tools):
    """Greedy chain from full grant down to the empty grant: at each step remove
    the tool with the best (attack-mass blocked) / (approval load added) ratio."""
    G = set(tools); chain = [frozenset(G)]
    while G:
        best = None
        s0 = s_of(frozenset(G), Rs, Is); n0 = load_of(frozenset(G), Rs)
        for t in sorted(G):
            G2 = frozenset(G - {t})
            ds = s0 - s_of(G2, Rs, Is); dn = load_of(G2, Rs) - n0
            key = (-(ds + 1e-9) / (dn + 1e-3), dn, t)
            if best is None or key < best[0]:
                best = (key, t)
        G.remove(best[1]); chain.append(frozenset(G))
    return chain


def fatigue():
    d0 = 0.9
    n0s = [np.inf, 4.0, 2.0, 1.0, 0.5, 0.25]
    out = {"d0": d0, "n0": [str(x) for x in n0s], "suites": {}}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, (sname, S) in zip(axes, D["suites"].items()):
        Rs, Is, tools = suite_arrays(S)
        chain = nested_grants(Rs, Is, tools)
        n = np.array([load_of(G, Rs) for G in chain])
        s = np.array([s_of(G, Rs, Is) for G in chain])
        rec = {"grant_size": [len(G) for G in chain], "load": n.tolist(), "s_inside": s.tolist(), "risk": {}, "argmin": {}}
        for j, n0 in enumerate(n0s):
            d = d0 * np.exp(-n / n0) if np.isfinite(n0) else d0 * np.ones_like(n)
            r = s + (1 - s) * (1 - d)
            rec["risk"][str(n0)] = r.tolist()
            k = int(np.argmin(r))
            rec["argmin"][str(n0)] = {"grant_size": len(chain[k]), "load": float(n[k]), "risk": float(r[k]),
                                      "risk_minimal_grant": float(r[-1]), "risk_full_grant": float(r[0])}
            ax.plot(n, r, "-", color=cmap(j / (len(n0s) - 1)), label=f"n0={n0:g}" if np.isfinite(n0) else "no fatigue")
            ax.plot(n[k], r[k], "o", color=cmap(j / (len(n0s) - 1)), ms=4)
        ax.set_title(sname); ax.set_xlabel("approval prompts per benign task")
        ax.grid(alpha=.3)
        out["suites"][sname] = rec
    axes[0].set_ylabel("P(attack executes | followed injection)")
    axes[-1].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Approval fatigue: risk along greedy nested base grants (d0={d0}; dots = optimum)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_approval_fatigue.png"), dpi=160)
    json.dump(out, open(os.path.join(RES, "approval_fatigue.json"), "w"), indent=1)
    for sname, rec in out["suites"].items():
        print(sname, {k: (v["grant_size"], round(v["load"], 2), round(v["risk"], 3), round(v["risk_minimal_grant"], 3), round(v["risk_full_grant"], 3)) for k, v in rec["argmin"].items()})


def design_profiles(Rs, Is, k):
    clusters = [([j], R) for j, R in enumerate(Rs)]

    def cost(G, m):
        return len(m) * sum(feas(a, G) for a in Is)
    while len(clusters) > k:
        best = None
        for a, b in itertools.combinations(range(len(clusters)), 2):
            G = clusters[a][1] | clusters[b][1]
            delta = cost(G, clusters[a][0] + clusters[b][0]) - cost(clusters[a][1], clusters[a][0]) - cost(clusters[b][1], clusters[b][0])
            key = (delta, len(G))
            if best is None or key < best[0]:
                best = (key, a, b, G)
        _, a, b, G = best
        merged = (clusters[a][0] + clusters[b][0], G)
        clusters = [c for j, c in enumerate(clusters) if j not in (a, b)] + [merged]
    return [G for _, G in clusters]


def holdout(reps=200, ks=(1, 2, 3, 4, 6, 8)):
    out = {}
    for sname, S in D["suites"].items():
        Rs, Is, tools = suite_arrays(S)
        N = len(Rs)
        res = {k: {"coverage": [], "feasible_covered": [], "feasible_all_fallback_full": []} for k in ks}
        for _ in range(reps):
            perm = rng.permutation(N); tr, te = perm[: N // 2], perm[N // 2:]
            for k in ks:
                profs = design_profiles([Rs[j] for j in tr], Is, min(k, len(tr)))
                cov = 0; fc = []; fall = []
                for j in te:
                    fits = [G for G in profs if Rs[j] <= G]
                    if fits:
                        cov += 1
                        G = min(fits, key=lambda G: (sum(feas(a, G) for a in Is), len(G)))
                        f = np.mean([feas(a, G) for a in Is]); fc.append(f); fall.append(f)
                    else:  # task needs escalation to the full grant
                        fall.append(np.mean([feas(a, tools) for a in Is]))
                res[k]["coverage"].append(cov / len(te))
                res[k]["feasible_covered"].append(np.mean(fc) if fc else np.nan)
                res[k]["feasible_all_fallback_full"].append(np.mean(fall))
        out[sname] = {str(k): {m: [float(np.nanmean(v)), float(np.nanstd(v))] for m, v in d.items()} for k, d in res.items()}
        out[sname]["in_sample_task_LP"] = float(np.mean([feas(a, R) for R in Rs for a in Is]))
        print(sname, {k: (round(out[sname][str(k)]["coverage"][0], 3), round(out[sname][str(k)]["feasible_covered"][0], 3), round(out[sname][str(k)]["feasible_all_fallback_full"][0], 3)) for k in ks}, "taskLP", round(out[sname]["in_sample_task_LP"], 3))
    json.dump(out, open(os.path.join(RES, "profile_holdout.json"), "w"), indent=1)


if __name__ == "__main__":
    fatigue()
    holdout()
