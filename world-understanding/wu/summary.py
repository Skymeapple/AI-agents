"""Print the headline numbers quoted in the paper, aggregated across seeds.
Usage: python -m wu.summary"""
import numpy as np

from .common import RESULTS, load_json


def ms(v, f=3):
    v = np.asarray(v, float)
    return f"{v.mean():.{f}f} ± {v.std():.{f}f}"


def e1():
    R = [load_json(p.stem) for p in sorted(RESULTS.glob("e1_seed[0-9].json"))]
    print(f"E1  ({len(R)} seeds)")
    for k in ["uniform", "best_ngram", "model", "bayes", "model_ood", "bayes_ood", "model_untrained"]:
        print(f"  nll {k:16s} {ms([r['nll'][k] for r in R], 4)}")
    print("  gap closed       ", ms([r["frac_gap_closed"] for r in R]))
    print("  excess model-bayes", ms([r["nll"]["model"] - r["nll"]["bayes"] for r in R], 4),
          " ood:", ms([r["nll"]["model_ood"] - r["nll"]["bayes_ood"] for r in R], 4))
    print("  bayes MAP ceiling", ms([r["bayes_map_acc"] for r in R]), " frac identified", ms([r["frac_identified"] for r in R]))
    for li in range(len(R[0]["probes"])):
        row = {k: ms([r["probes"][li][k] for r in R]) for k in R[0]["probes"][li] if k != "layer"}
        print(f"  probe L{li}", row, " untrained state", ms([r["probes_untrained"][li]["acc_state"] for r in R]))
    print("  chance state", 1 / R[0]["S"], "cell", 1 / R[0]["n_cells"])
    for j, iv in enumerate(R[0]["interventions"]):
        print(f"  intervene L{iv['layer']}",
              {k: ms([r["interventions"][j][k] for r in R]) for k in
               ["patch_acc_target", "random_acc_target", "none_acc_target", "patch_acc_original", "patch_p_target"]})
    print("  RSA", ms([r["rsa_spearman"] for r in R]), " ID per layer", np.round(np.mean([r["intrinsic_dim"] for r in R], 0), 2))
    print("  dynamics", [(d["step"], round(np.mean([r["dynamics"][i]["excess_nll"] for r in R]), 3),
                         round(np.mean([r["dynamics"][i]["acc_state"] for r in R]), 3)) for i, d in enumerate(R[0]["dynamics"])])
    ex = np.concatenate([[d["excess_nll"] for d in r["dynamics"]] for r in R])
    ac = np.concatenate([[d["acc_state"] for d in r["dynamics"]] for r in R])
    from scipy.stats import spearmanr
    print("  spearman(excess, probe acc)", round(spearmanr(ex, ac).correlation, 3))
    print("  ngram by n", {k: round(v, 3) for k, v in R[0]["nll"]["ngram"].items()})
    print("  train seconds", [round(r["train_seconds"]) for r in R])


def e5():
    R = load_json("e5_grokking")
    print("E5")
    for r in R["main"]:
        print("  main", r["seed"], "t_train99", r["t_train99"], "t_test99", r["t_test99"], "final", round(r["final_test"], 4),
              "fourier", round(r["final_fourier"], 3))
        h = np.asarray(r["hist"])
        i_tr = np.argmax(h[:, 2] >= 0.99)
        print("     fourier at train99", round(h[i_tr, 6], 3), "norm at train99", round(h[i_tr, 5], 1),
              "norm final", round(h[-1, 5], 1), "max norm", round(h[:, 5].max(), 1))
        it = np.argmax(h[:, 3] >= 0.5) if (h[:, 3] >= 0.5).any() else None
        if it:
            print("     step test>=0.5", int(h[it, 0]), "fourier then", round(h[it, 6], 3))
    for key in ["wd_sweep", "frac_sweep"]:
        for r in R[key]:
            print(f"  {key} wd={r['wd']} frac={r['frac']} seed={r['seed']} t_train={r['t_train99']} t_test={r['t_test99']} "
                  f"final={r['final_test']:.3f} fourier={r['final_fourier']:.3f}")


def e6():
    R = load_json("e6_incontext")
    print("E6")
    for r in R["runs"]:
        for ev in r["eval"].values():
            for k in ev:
                ev[k] = np.asarray(ev[k])
        f = r["eval"]["fresh"]
        line = f"  M={r['M']}: fresh model {np.round(f['model'][[1, 4, 8, 15]], 3)} ridge {np.round(f['ridge'][[1, 4, 8, 15]], 3)}"
        line += f" dist_ridge(tail) {np.mean(f['dist_to_ridge'][8:]):.4f}"
        if "dist_to_dmmse" in f:
            line += f" dist_dmmse(tail) {np.mean(f['dist_to_dmmse'][8:]):.4f} dmmse {np.round(f['dmmse'][[1, 4, 8, 15]], 3)}"
        if "pool" in r["eval"]:
            p = r["eval"]["pool"]
            line += f" | pool: model {np.round(p['model'][[1, 4, 8, 15]], 3)} dmmse {np.round(p['dmmse'][[1, 4, 8, 15]], 3)}"
        if "ols" in f:
            line += f" ols {np.round(f['ols'][[4, 8, 15]], 3)}"
        print(line, f"[{r['seconds']:.0f}s]")


if __name__ == "__main__":
    for fn in (e1, e5, e6):
        try:
            fn()
        except FileNotFoundError as ex:
            print("missing:", ex)
