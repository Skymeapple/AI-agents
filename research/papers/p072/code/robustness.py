"""Robustness of headline results to prior choices. Usage: python3 robustness.py"""
import json, math
import numpy as np
from multiprocessing import Pool
import obligations, model
from model import Scenario, solve, sample_params, SEGMENTS, FULL

SEED = 7
SC = {"H1": Scenario("H1_CN_vs_USF"), "H0": Scenario("H0_no_conflict"),
      "H2": Scenario("H1_CN_vs_USF", "H2_CN_runtime"), "H4": Scenario("H4_USF_runtime"),
      "BUNDLE": Scenario("H1_CN_vs_USF", instruments=("I1_MR_incident", "I2_harmonize_frameworks",
                         "I3_harmonize_training_summary", "I4_harmonize_marking", "I5_harmonize_visible",
                         "I6_harmonize_provenance")),
      "PROC": Scenario("H1_CN_vs_USF", instruments=("I1_MR_incident", "I2_harmonize_frameworks",
                       "I3_harmonize_training_summary"))}
USF = SEGMENTS.index("USF")


def draw(args):
    k, usf_scale, cost_scale, q_scale = args
    rng = np.random.default_rng([SEED, k])
    p = sample_params(rng)
    p["m"][USF] *= usf_scale
    p["F_w"] *= cost_scale; p["F_r"] *= cost_scale
    p["a"] = {d: v * cost_scale for d, v in p["a"].items()}
    p["b"] = {d: v * cost_scale for d, v in p["b"].items()}
    p["h"]["D8_political_content_norm"] = min(1.0, p["h"]["D8_political_content_norm"] * q_scale)
    out = {}
    for n, sc in SC.items():
        s = solve(sc, p)
        out[n] = (len(s["wparts"]), s["served"], s["cost"], s["net"], sorted(s["wparts"]),
                  sorted(map(sorted, s["rparts"])))
    return out


def run(usf_scale, cost_scale, q_scale, n=3000):
    with Pool(4) as pool:
        res = pool.map(draw, [(k, usf_scale, cost_scale, q_scale) for k in range(n)], chunksize=50)
    r = {}
    for nm in SC:
        r[f"{nm}_p_single_global"] = float(np.mean([o[nm][0] == 1 and o[nm][1] == FULL for o in res]))
        r[f"{nm}_p_USF_served"] = float(np.mean([o[nm][1] >> USF & 1 for o in res]))
        r[f"{nm}_net_gain_vs_H1"] = float(np.mean([o[nm][3] - o["H1"][3] for o in res]))
    r["PROC_p_any_structure_change"] = float(np.mean([(o["PROC"][4], o["PROC"][5], o["PROC"][1]) !=
                                                       (o["H1"][4], o["H1"][5], o["H1"][1]) for o in res]))
    r["BUNDLE_p_weight_change"] = float(np.mean([o["BUNDLE"][4] != o["H1"][4] for o in res]))
    r["H2_over_BUNDLE_gain_ratio"] = r["H2_net_gain_vs_H1"] / max(r["BUNDLE_net_gain_vs_H1"], 1e-9)
    return r


if __name__ == "__main__":
    grid = []
    for usf in (0.2, 1.0, 5.0, 20.0):
        grid.append(("usf_scale", usf, run(usf, 1.0, 1.0)))
    for cs in (0.25, 4.0):
        grid.append(("cost_scale", cs, run(1.0, cs, 1.0)))
    for qs in (0.1, 2.0):
        grid.append(("q_scale", qs, run(1.0, 1.0, qs)))
    json.dump([dict(param=a, value=b, **r) for a, b, r in grid], open("../results/robustness.json", "w"), indent=1)
    keys = ["H1_p_single_global", "H0_p_single_global", "H4_p_single_global", "H2_p_single_global",
            "H1_p_USF_served", "H2_net_gain_vs_H1", "BUNDLE_net_gain_vs_H1", "H4_net_gain_vs_H1",
            "PROC_p_any_structure_change", "BUNDLE_p_weight_change"]
    with open("../results/robustness.csv", "w") as f:
        f.write("param,value," + ",".join(keys) + "\n")
        for a, b, r in grid:
            f.write(f"{a},{b}," + ",".join(f"{r[k]:.3f}" for k in keys) + "\n")
    print(open("../results/robustness.csv").read())
