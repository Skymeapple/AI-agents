"""Run all experiments. Usage: python3 run_experiments.py [n_draws] (default 4000)."""
import sys, json, math, time
import numpy as np
from multiprocessing import Pool
from scipy.stats import spearmanr
from model import (Scenario, solve, sample_params, median_params, members, SEGMENTS, N, FULL,
                   conflict_chromatic_lower_bound, export_events, DIMENSIONS)

OUT = "../results"
SEED = 20261122

SCENARIOS = {
    "H1_baseline": dict(conflict_key="H1_CN_vs_USF"),
    "H0_no_conflict": dict(conflict_key="H0_no_conflict"),
    "H2_CN_runtime": dict(conflict_key="H1_CN_vs_USF", layer_override="H2_CN_runtime"),
    "H3_CN_vs_US_wide": dict(conflict_key="H3_CN_vs_US_wide"),
    "H4_USF_runtime": dict(conflict_key="H4_USF_runtime"),
    "I1_MR_incident": dict(conflict_key="H1_CN_vs_USF", instruments=("I1_MR_incident",)),
    "I2_harmonize_frameworks": dict(conflict_key="H1_CN_vs_USF", instruments=("I2_harmonize_frameworks",)),
    "I3_harmonize_training_summary": dict(conflict_key="H1_CN_vs_USF", instruments=("I3_harmonize_training_summary",)),
    "I4_harmonize_marking": dict(conflict_key="H1_CN_vs_USF", instruments=("I4_harmonize_marking",)),
    "I5_harmonize_visible": dict(conflict_key="H1_CN_vs_USF", instruments=("I5_harmonize_visible",)),
    "I6_harmonize_provenance": dict(conflict_key="H1_CN_vs_USF", instruments=("I6_harmonize_provenance",)),
    "I_bundle_all_MR_harmonization": dict(conflict_key="H1_CN_vs_USF", instruments=(
        "I1_MR_incident", "I2_harmonize_frameworks", "I3_harmonize_training_summary",
        "I4_harmonize_marking", "I5_harmonize_visible", "I6_harmonize_provenance")),
}
SC = {k: Scenario(**v) for k, v in SCENARIOS.items()}
CN = SEGMENTS.index("CN")


def summarize(sol, scen):
    S = sol["served"]
    nw = len(sol["wparts"])
    nr = sum(len(r) for r in sol["rparts"])
    cn_sep_w = any(W == (1 << CN) for W in sol["wparts"])
    lb_w = conflict_chromatic_lower_bound(scen, S, {"weights"})
    lb_wr = conflict_chromatic_lower_bound(scen, S, {"weights", "runtime"})
    ev = export_events(sol, scen)
    return dict(net=sol["net"], cost=sol["cost"], served=S, nw=nw, nr=nr, cn_sep_w=cn_sep_w,
                lb_w=lb_w, lb_wr=lb_wr, single_global=(nw == 1 and S == FULL),
                wparts=sol["wparts"], rparts=sol["rparts"], exports=ev)


def one_draw(k):
    rng = np.random.default_rng([SEED, k])
    p = sample_params(rng)
    res = {name: summarize(solve(sc, p), sc) for name, sc in SC.items()}
    feats = dict(F_w=p["F_w"], F_r=p["F_r"], q_D8=p["h"]["D8_political_content_norm"],
                 a_D8=p["a"]["D8_political_content_norm"], a_D3=p["a"]["D3_model_evaluation"],
                 a_D5=p["a"]["D5_training_data_provenance"], m_CN=p["m"][CN],
                 m_rest=float(p["m"].sum() - p["m"][CN]), m_USF=p["m"][SEGMENTS.index("USF")],
                 eu_inc=p["eu_incident_level"])
    return res, feats


def phase_diagram():
    """Proposition 1 check: under H0 (no conflict), CN forks its own weight variant iff
    q * M_rest exceeds the fork premium kappa (closed form below), with other params at medians."""
    p0 = median_params()
    scen = Scenario("H0_no_conflict")
    Fw_grid = np.exp(np.linspace(math.log(0.5), math.log(40), 60))
    q_grid = np.linspace(0.0, 0.4, 61)
    fork = np.zeros((len(q_grid), len(Fw_grid)), dtype=int)
    for a, q in enumerate(q_grid):
        for b, Fw in enumerate(Fw_grid):
            p = dict(p0); p["h"] = dict(p0["h"]); p["h"]["D8_political_content_norm"] = q
            p["F_w"] = Fw
            sol = solve(scen, p)
            if not (sol["served"] >> CN & 1):
                fork[a, b] = 3   # exit CN
            elif len(sol["wparts"]) == 1:
                fork[a, b] = 0
            elif any(W == (1 << CN) for W in sol["wparts"]):
                fork[a, b] = 1
            else:
                fork[a, b] = 2   # some other split (e.g. lenient bloc vs strict bloc)
    # closed-form boundary: q* = (F_w + F_r + Delta_dup) / M_rest, where Delta_dup are the
    # weight-layer costs duplicated by giving CN its own weights minus the D8 alignment cost
    # saved on the rest block, and the runtime cost change (CN already runs separately at runtime).
    from model import block_cost
    reqs = {d: scen.req(d, p0) for d in DIMENSIONS}
    cnm = 1 << CN; rest = ((1 << N) - 1) ^ cnm
    p00 = dict(p0); p00["h"] = dict(p0["h"]); p00["h"]["D8_political_content_norm"] = 0.0
    dup = (block_cost(cnm, "weights", scen, p00, reqs) + block_cost(rest, "weights", scen, p00, reqs)
           - block_cost((1 << N) - 1, "weights", scen, p00, reqs))
    M_rest = float(p0["m"].sum() - p0["m"][CN])
    # runtime: in the merged solution CN already has its own runtime variant at median params
    q_star = (Fw_grid + dup) / M_rest
    return dict(Fw_grid=Fw_grid.tolist(), q_grid=q_grid.tolist(), fork=fork.tolist(),
                q_star=q_star.tolist(), dup=dup, M_rest=M_rest)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    t0 = time.time()
    with Pool(4) as pool:
        out = pool.map(one_draw, range(n), chunksize=50)
    print(f"{n} draws in {time.time()-t0:.1f}s")
    names = list(SCENARIOS)
    summary = {"n_draws": n, "seed": SEED, "scenarios": {}}
    base = [o[0]["H1_baseline"] for o in out]
    for name in names:
        rs = [o[0][name] for o in out]
        nw = np.array([r["nw"] for r in rs]); nr = np.array([r["nr"] for r in rs])
        net = np.array([r["net"] for r in rs]); cost = np.array([r["cost"] for r in rs])
        dcost = np.array([b["cost"] - r["cost"] for b, r in zip(base, rs)])
        dnet = np.array([r["net"] - b["net"] for b, r in zip(base, rs)])
        changed_w = np.mean([sorted(b["wparts"]) != sorted(r["wparts"]) for b, r in zip(base, rs)])
        changed_r = np.mean([sorted(map(sorted, b["rparts"])) != sorted(map(sorted, r["rparts"]))
                             for b, r in zip(base, rs)])
        served_frac = {s: float(np.mean([r["served"] >> i & 1 for r in rs])) for i, s in enumerate(SEGMENTS)}
        lb_viol = int(sum(r["nw"] < r["lb_w"] for r in rs))
        summary["scenarios"][name] = dict(
            mean_nw=float(nw.mean()), mean_nr=float(nr.mean()),
            p_nw={int(k): float(np.mean(nw == k)) for k in np.unique(nw)},
            p_single_global=float(np.mean([r["single_global"] for r in rs])),
            p_cn_separate_weights=float(np.mean([r["cn_sep_w"] for r in rs])),
            mean_cost=float(cost.mean()),
            mean_cost_saving_vs_H1=float(dcost.mean()),
            cost_saving_q05_q50_q95=[float(x) for x in np.quantile(dcost, [0.05, 0.5, 0.95])],
            rel_cost_saving_mean=float(np.mean(dcost / np.array([b["cost"] for b in base]))),
            mean_net_gain_vs_H1=float(dnet.mean()),
            p_weight_structure_changed_vs_H1=float(changed_w),
            p_runtime_structure_changed_vs_H1=float(changed_r),
            served_frac=served_frac,
            chromatic_lb_violations=lb_viol,
            p_nw_eq_lb=float(np.mean([r["nw"] == r["lb_w"] for r in rs])),
        )
    # export matrices (fraction of draws in which exporter i's rule binds importer j's product)
    exp = {}
    for name in ["H1_baseline", "H0_no_conflict", "H2_CN_runtime", "I_bundle_all_MR_harmonization"]:
        M = np.zeros((N, N)); Mo = np.zeros((N, N))
        for o in out:
            seen = set(); seen_o = set()
            for i, j, d in o[0][name]["exports"]:
                seen.add((i, j))
                if DIMENSIONS[d]["kind"] == "ordered":
                    seen_o.add((i, j))
            for i, j in seen:
                M[i, j] += 1
            for i, j in seen_o:
                Mo[i, j] += 1
        perdim = {}
        for dd in ("D1_incident_reporting", "D3_model_evaluation", "D8_political_content_norm"):
            Md = np.zeros((N, N))
            for o in out:
                for i, j in {(i, j) for i, j, d in o[0][name]["exports"] if d == dd}:
                    Md[i, j] += 1
            perdim[dd] = (Md / n).tolist()
        exp[name] = dict(any=(M / n).tolist(), ordered=(Mo / n).tolist(), per_dim=perdim)
    summary["export_matrices"] = exp
    summary["segments"] = SEGMENTS
    # sensitivity: Spearman rank correlations
    feats = {k: np.array([o[1][k] for o in out]) for k in out[0][1]}
    sens = {}
    for name, metric in [("H0_no_conflict", "cn_sep_w"), ("H1_baseline", "nr"), ("H1_baseline", "nw"),
                         ("I_bundle_all_MR_harmonization", "dcost"), ("H2_CN_runtime", "dcost")]:
        if metric == "dcost":
            y = np.array([b["cost"] - o[0][name]["cost"] for b, o in zip(base, out)])
        else:
            y = np.array([float(o[0][name][metric]) for o in out])
        sens[f"{name}:{metric}"] = {k: float(spearmanr(v, y).correlation) if np.std(y) > 0 else None
                                    for k, v in feats.items()}
    summary["sensitivity_spearman"] = sens
    # EU incident-level ambiguity
    for lev in (1, 2, 3):
        idx = [k for k, o in enumerate(out) if o[1]["eu_inc"] == lev]
        summary.setdefault("eu_incident_level_split", {})[lev] = dict(
            n=len(idx), mean_cost_H1=float(np.mean([base[k]["cost"] for k in idx])))
    summary["phase"] = phase_diagram()
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    # compact table
    with open(f"{OUT}/scenario_table.csv", "w") as f:
        f.write("scenario,mean_nw,mean_nr,p_single_global,p_cn_separate_weights,mean_cost,"
                "mean_cost_saving_vs_H1,rel_saving,p_w_changed,p_r_changed,served_USF,lb_violations\n")
        for name, s in summary["scenarios"].items():
            f.write(f"{name},{s['mean_nw']:.3f},{s['mean_nr']:.3f},{s['p_single_global']:.3f},"
                    f"{s['p_cn_separate_weights']:.3f},{s['mean_cost']:.3f},{s['mean_cost_saving_vs_H1']:.3f},"
                    f"{s['rel_cost_saving_mean']:.3f},{s['p_weight_structure_changed_vs_H1']:.3f},"
                    f"{s['p_runtime_structure_changed_vs_H1']:.3f},{s['served_frac']['USF']:.3f},"
                    f"{s['chromatic_lb_violations']}\n")
    print(open(f"{OUT}/scenario_table.csv").read())


if __name__ == "__main__":
    main()
