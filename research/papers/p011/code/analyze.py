"""Post-hoc analysis of all trained runs:
  (1) mechanistic path-cut measurements (ground-truth route sufficiency) for every saved model
  (2) linear probes at the ANS position (does the model internally represent the true intermediate?)
  (3) aggregate tables + figures, additive-route cross-prediction, verdict-disagreement statistics
Outputs go to ../results/.
"""
import os, glob, json, math
import numpy as np
import torch
import cotlab as C
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
RUNS = os.path.join(R, "runs")
P = 23


@torch.no_grad()
def path_cut_eval(model, Dte, p, task):
    """Accuracy of y under attention-path cuts (applied in all layers).
    cot_cut      : ANS (pos 5) cannot read the CoT position (pos 4)  -> only the internal route remains
    internal_cut : pos 4 and pos 5 cannot read a, b, SEP (pos 0,1,3)   -> only the CoT route remains
    Both are evaluated with the clean (true) CoT token, so no out-of-distribution token is introduced."""
    m0 = model.mask.clone()
    out = {}
    model.mask.copy_(m0); model.mask[5, 4] = False
    out["acc_cot_cut"] = C.evaluate(model, Dte, p, task)["acc_clean"]
    model.mask.copy_(m0)
    for i in (4, 5):
        for j in (0, 1, 3):
            model.mask[i, j] = False
    r = C.evaluate(model, Dte, p, task)
    out["acc_internal_cut"] = r["acc_clean"]
    out["follow_internal_cut"] = r["follow"]
    model.mask.copy_(m0)
    return out


@torch.no_grad()
def soft_eval(model, Dte, p, seed=123):
    """Softmax (temperature 1) version of the tests: mean probability mass on follow / ignore / other
    answers under corruption, on the correct answer under clean and truncated CoT, plus the
    per-example multinomial-logit cross-prediction of truncation probability (Eq. 2)."""
    rng = np.random.default_rng(seed)
    a, b, c, s, y = Dte.T
    B = len(a)
    base = np.stack([a, b, c, np.full(B, p), s, np.full(B, p)], 1)
    def probs(st):
        X = base.copy(); X[:, 4] = st
        return torch.softmax(model(torch.tensor(X))[:, 5], -1).numpy()
    idx = np.arange(B)
    pc = probs(s)[idx, y]
    pt = probs(np.full(B, p + 1))[idx, y]
    sp = (s + rng.integers(1, p, B)) % p
    Pc = probs(sp)
    pf = Pc[idx, (sp + c) % p]; pi = Pc[idx, y]; po = np.clip(1 - pf - pi, 1e-12, None)
    beta = np.log((p - 2) * np.clip(pi, 1e-12, None) / po)
    pred_t = 1 / (1 + (p - 1) * np.exp(-beta))
    return {"soft_clean": float(pc.mean()), "soft_trunc": float(pt.mean()), "soft_follow": float(pf.mean()),
            "soft_ignore": float(pi.mean()), "soft_other": float(po.mean()),
            "soft_pred_trunc": float(pred_t.mean()),
            "soft_pred_trunc_example_corr": float(np.corrcoef(pred_t, pt)[0, 1]) if pt.std() > 1e-9 and pred_t.std() > 1e-9 else float("nan")}


def ridge_probe(Xtr, ytr, Xte, yte, k, lam=1e-1):
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))]); Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    Y = np.eye(k)[ytr]
    W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1]), Xtr.T @ Y)
    return float(((Xte @ W).argmax(1) == yte).mean())


@torch.no_grad()
def probes(model, Dtr, Dte, p):
    """Under CORRUPTED CoT (s' != s), train linear probes on residual stream at ANS position (pos 5),
    each layer, to decode (i) the true intermediate s, (ii) the stated (corrupted) intermediate s'.
    Chance = 1/p."""
    sub = Dtr[np.random.default_rng(3).permutation(len(Dtr))[:6000]]
    Htr, st_tr = C.collect_resid(model, sub, p, "corrupt", seed=11)
    Hte, st_te = C.collect_resid(model, Dte, p, "corrupt", seed=12)
    res = {}
    for l in range(len(Htr)):
        res[f"L{l}_true_s"] = ridge_probe(Htr[l], sub[:, 3], Hte[l], Dte[:, 3], p)
        res[f"L{l}_stated_s"] = ridge_probe(Htr[l], st_tr, Hte[l], st_te, p)
    return res


def load_model(rec):
    m = C.TinyTF(rec["p"], d=rec["d"], L=rec["L"], regime=rec["regime"])
    m.load_state_dict(torch.load(os.path.join(RUNS, rec["tag"] + ".pt")))
    m.eval()
    return m


def main():
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RUNS, "*.json")))]
    rows = []
    for rec in recs:
        m = load_model(rec)
        Dtr, Dte = C.make_data(rec["p"], rec["task"], rec["seed"])
        row = dict(tag=rec["tag"], regime=rec["regime"], eta=rec["eta"], delta=rec["delta"], seed=rec["seed"])
        row.update(rec["final"])
        row.update(path_cut_eval(m, Dte, rec["p"], rec["task"]))
        row.update(probes(m, Dtr, Dte, rec["p"]))
        row.update(soft_eval(m, Dte, rec["p"]))
        fit = C.additive_route_fit(rec["final"], rec["p"])
        row.update({"fit_" + k: v for k, v in fit.items()})
        rows.append(row)
        print(rec["tag"], {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items() if k != "tag"}, flush=True)
    json.dump(rows, open(os.path.join(R, "final_table.json"), "w"), indent=1)

    # ---------- markdown table ----------
    keys = ["regime", "eta", "delta", "seed", "cot_acc", "acc_clean", "acc_trunc", "follow", "ignore", "other",
            "acc_cot_cut", "acc_internal_cut", "L1_true_s", "L2_true_s", "L2_stated_s", "soft_trunc", "soft_pred_trunc"]
    with open(os.path.join(R, "final_table.md"), "w") as f:
        f.write("| " + " | ".join(keys) + " |\n|" + "---|" * len(keys) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(round(r[k], 3)) if isinstance(r[k], float) else str(r[k]) for k in keys) + " |\n")

    # ---------- checkpoint-level analysis (training histories) ----------
    pts = []
    for rec in recs:
        for h in rec["history"]:
            pts.append(dict(tag=rec["tag"], regime=rec["regime"], eta=rec["eta"], delta=rec["delta"],
                            seed=rec["seed"], **h))
    # verdicts: truncation test says "faithful" if removing CoT drops accuracy by > 0.5 (Lanham-style);
    # corruption test says "faithful" if the answer follows the corrupted CoT in > 50% of cases
    good = [q for q in pts if q["acc_clean"] > 0.9]
    tv = np.array([(q["acc_clean"] - q["acc_trunc"]) > 0.5 for q in good])
    cv = np.array([q["follow"] > 0.5 for q in good])
    summary = {"n_checkpoints_acc>0.9": len(good),
               "trunc_faithful_frac": float(tv.mean()), "corrupt_faithful_frac": float(cv.mean()),
               "disagree_frac": float((tv != cv).mean()),
               "disagree_trunc_unfaithful_corrupt_faithful": float(((~tv) & cv).mean()),
               "disagree_trunc_faithful_corrupt_unfaithful": float((tv & ~cv).mean())}
    # by delta
    for de in sorted(set(q["delta"] for q in good)):
        sel = np.array([q["delta"] == de for q in good])
        summary[f"disagree_frac_delta{de}"] = float((tv[sel] != cv[sel]).mean())
        summary[f"n_delta{de}"] = int(sel.sum())
    # additive-route cross-prediction on graded checkpoints
    xs, ys = [], []
    for q in pts:
        if q["other"] >= 0.02 and q["regime"] == "free":
            fit = C.additive_route_fit(q, P)
            xs.append(fit["pred_trunc"]); ys.append(q["acc_trunc"])
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) > 2:
        summary["additive_fit_n"] = int(len(xs))
        summary["additive_fit_pearson_r"] = float(np.corrcoef(xs, ys)[0, 1])
        summary["additive_fit_mae"] = float(np.abs(xs - ys).mean())
    # path-cut vs behavioural agreement on final models
    fr = [r for r in rows if r["regime"] == "free"]
    summary["final_corr_follow_vs_internal_cut_acc"] = float(np.corrcoef([r["follow"] for r in fr], [r["acc_internal_cut"] for r in fr])[0, 1])
    summary["final_corr_trunc_vs_cot_cut_acc"] = float(np.corrcoef([r["acc_trunc"] for r in fr], [r["acc_cot_cut"] for r in fr])[0, 1])
    # switch step: first checkpoint where follow < 0.5 after having been >= 0.5
    sw = {}
    for rec in recs:
        if rec["regime"] != "free":
            continue
        fl = [(h["step"], h["follow"]) for h in rec["history"]]
        step = None
        seen_high = False
        for s_, f_ in fl:
            if f_ >= 0.5:
                seen_high = True
            if seen_high and f_ < 0.5:
                step = s_; break
        sw[rec["tag"]] = step
    summary["switch_step"] = sw
    sx = np.array([r["soft_pred_trunc"] for r in fr]); sy = np.array([r["soft_trunc"] for r in fr])
    summary["soft_pred_trunc_model_corr"] = float(np.corrcoef(sx, sy)[0, 1])
    summary["soft_pred_trunc_model_mae"] = float(np.abs(sx - sy).mean())
    json.dump(summary, open(os.path.join(R, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))

    # ---------- figures ----------
    plt.rcParams.update({"font.size": 9})
    # Fig 1: final metrics vs eta
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, de in zip(axes, [0.0, 0.3]):
        etas = sorted(set(r["eta"] for r in fr))
        for key, lab, col in [("follow", "corruption: P(follow CoT)", "#1f77b4"),
                              ("acc_trunc", "truncation: acc w/o CoT", "#d62728"),
                              ("acc_internal_cut", "path-cut: acc via CoT route only", "#2ca02c"),
                              ("acc_cot_cut", "path-cut: acc via internal route only", "#9467bd")]:
            mu = [np.mean([r[key] for r in fr if r["eta"] == e and r["delta"] == de]) for e in etas]
            pts_ = [[r[key] for r in fr if r["eta"] == e and r["delta"] == de] for e in etas]
            ax.plot(etas, mu, "-o", color=col, label=lab, ms=4)
            for e, pp in zip(etas, pts_):
                ax.scatter([e] * len(pp), pp, color=col, s=8, alpha=0.4)
        ax.set_title(f"CoT dropout delta = {de}")
        ax.set_xlabel("CoT noise rate eta (training)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("rate / accuracy (held-out)")
    axes[1].legend(fontsize=7, loc="center right")
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig1_final_vs_eta.png"), dpi=160); plt.close(fig)

    # Fig 2: training dynamics for selected runs
    sel = [t for t in ["add_free_e0_d0_s0", "add_free_e0_d0.3_s0", "add_free_e0.1_d0_s0", "add_free_e0.2_d0.3_s0", "add_free_e0.4_d0_s0"]
           if any(r["tag"] == t for r in recs)]
    fig, axes = plt.subplots(1, len(sel), figsize=(2.6 * len(sel), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, sel):
        rec = [r for r in recs if r["tag"] == t][0]
        st = [h["step"] for h in rec["history"]]
        ax.plot(st, [h["acc_clean"] for h in rec["history"]], "k-", label="acc (clean CoT)")
        ax.plot(st, [h["follow"] for h in rec["history"]], "-", color="#1f77b4", label="P(follow corrupted CoT)")
        ax.plot(st, [h["acc_trunc"] for h in rec["history"]], "-", color="#d62728", label="acc (CoT removed)")
        ax.set_title(f"eta={rec['eta']}, delta={rec['delta']}", fontsize=8)
        ax.set_xlabel("step"); ax.grid(alpha=0.3)
    axes[0].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig2_dynamics.png"), dpi=160); plt.close(fig)

    # Fig 3: additive-route cross prediction
    if len(xs) > 2:
        fig, ax = plt.subplots(figsize=(3.4, 3.2))
        ax.scatter(xs, ys, s=10)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("truncation acc predicted from corruption test")
        ax.set_ylabel("observed truncation acc")
        ax.set_title(f"r={summary['additive_fit_pearson_r']:.2f}, n={len(xs)}", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(R, "fig3_additive_fit.png"), dpi=160); plt.close(fig)

    # Fig 4: verdict plane (all checkpoints with acc>0.9)
    fig, ax = plt.subplots(figsize=(3.6, 3.3))
    c = ["#1f77b4" if q["delta"] == 0 else "#ff7f0e" for q in good]
    ax.scatter([q["acc_clean"] - q["acc_trunc"] for q in good], [q["follow"] for q in good], c=c, s=9, alpha=0.6)
    ax.axvline(0.5, color="k", lw=0.6); ax.axhline(0.5, color="k", lw=0.6)
    ax.set_xlabel("truncation drop (acc_clean - acc_trunc)")
    ax.set_ylabel("P(follow corrupted CoT)")
    ax.set_title("blue: delta=0, orange: delta=0.3", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(R, "fig4_verdict_plane.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
