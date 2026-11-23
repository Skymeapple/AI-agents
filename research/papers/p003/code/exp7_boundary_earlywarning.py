"""E7: (a) critical honest discrimination J*(h0) for exploit takeover (bisection on J,
collapse fraction = 1/2), factorised and AR1 policies;  (b) an early-warning audit:
from a small gold-labelled audit of on-policy samples during the first 100 steps,
predict eventual collapse using the slope of log(false-accept share / true-accept share)."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import exp4_sequence as E

R = os.path.join(os.path.dirname(__file__), "..", "results")
BETA, ETA = 0.15, 0.95


def set_suffix(ps):
    E.P0 = np.array([0.6] * 5 + [0.5] * 3 + [ps] * 2)
    return float(E.Factorised().probs() @ E.EXPLOIT)


def collapse_frac(cls, J, seeds=8, steps=3000):
    f = E.verifier(BETA + J, BETA, ETA)
    return np.mean([E.train(cls(), f, steps=steps, seed=1000 + s, log_every=steps)[0][-1, 2] > 0.5
                    for s in range(seeds)])


def run_with_audit(cls, J, seed, steps=3000, n_audit=256, k_early=100):
    """Train, and every 10 steps during the first k_early steps draw n_audit on-policy samples,
    call the verifier once each, and use gold labels to count accepted-correct (TA) and
    accepted-incorrect (FA)."""
    rng = np.random.default_rng(seed + 777)
    f = E.verifier(BETA + J, BETA, ETA)
    pol = cls(); h, _ = None, None
    # replicate E.train but with audits
    rng_t = np.random.default_rng(seed); G, B, lr = 16, 8, 0.5
    audits = []
    for s in range(steps + 1):
        p = pol.probs()
        if s <= k_early and s % 10 == 0:
            idx = rng.choice(len(E.X), size=n_audit, p=p)
            acc = rng.random(n_audit) < f[idx]
            ta = np.sum(acc & E.CORRECT[idx]) + 0.5; fa = np.sum(acc & ~E.CORRECT[idx]) + 0.5
            audits.append((s, np.log(fa / ta)))
        idx = rng_t.choice(len(E.X), size=B * G, p=p)
        r = (rng_t.random(B * G) < f[idx]).astype(float).reshape(B, G)
        sd = r.std(1, keepdims=True)
        adv = np.where(sd > 1e-8, (r - r.mean(1, keepdims=True)) / np.maximum(sd, 1e-8), 0).ravel()
        d = lr * (adv[:, None] * pol.grad_logp(idx)).mean(0)
        if hasattr(pol, "step"): pol.step(d)
        else: pol.th += d
    p = pol.probs(); a = np.array(audits)
    slope = np.polyfit(a[:, 0], a[:, 1], 1)[0]
    # 'offline' audit statistic at step 0 only: false-accept share among accepted
    return slope, float(a[0, 1]), bool(p @ E.EXPLOIT > 0.5)


def auc(scores, labels):
    scores = np.asarray(scores); labels = np.asarray(labels, bool)
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0: return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


if __name__ == "__main__":
    out = {"boundary": [], "early_warning": {}}
    # (a) boundary
    for ps in [0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3]:
        h0 = set_suffix(ps)
        for pname, cls in [("factorised", E.Factorised), ("AR1", E.AR1)]:
            lo, hi = 0.02, 0.95
            if collapse_frac(cls, hi) >= 0.5:
                Jst = hi
            else:
                for _ in range(6):
                    mid = 0.5 * (lo + hi)
                    if collapse_frac(cls, mid) >= 0.5: lo = mid
                    else: hi = mid
                Jst = 0.5 * (lo + hi)
            out["boundary"].append({"suffix_p": ps, "h0": h0, "policy": pname, "J_star": Jst})
            print(out["boundary"][-1]); sys.stdout.flush()
    for pname in ["factorised", "AR1"]:
        b = [r for r in out["boundary"] if r["policy"] == pname and 0.02 < r["J_star"] < 0.95]
        if len(b) >= 3:
            sl, ic = np.polyfit(np.log(np.array([r["h0"] for r in b])), np.log([r["J_star"] for r in b]), 1)
            out[f"powerlaw_{pname}"] = {"exponent": float(sl), "prefactor": float(np.exp(ic)), "n": len(b)}
    print({k: v for k, v in out.items() if k.startswith("power")})

    # (b) early warning on random configurations (factorised policy)
    rng = np.random.default_rng(5)
    slopes, offl, labels, cfgs = [], [], [], []
    for i in range(60):
        ps = rng.uniform(0.05, 0.3); J = rng.uniform(0.05, 0.8)
        set_suffix(ps)
        s, o, c = run_with_audit(E.Factorised, J, seed=i)
        slopes.append(s); offl.append(o); labels.append(c); cfgs.append((ps, J))
    out["early_warning"] = {"n_runs": 60, "collapse_rate": float(np.mean(labels)),
                            "AUC_slope_first100": auc(slopes, labels),
                            "AUC_offline_false_accept_share_step0": auc(offl, labels),
                            "accuracy_rule_slope_gt_0": float(np.mean((np.array(slopes) > 0) == np.array(labels)))}
    print(out["early_warning"])
    json.dump(out, open(os.path.join(R, "exp7.json"), "w"), indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for j, pname in enumerate(["factorised", "AR1"]):
        b = [r for r in out["boundary"] if r["policy"] == pname]
        ax[0].loglog([r["h0"] for r in b], [r["J_star"] for r in b], f"C{j}-o", label=pname)
    ax[0].set_xlabel("exploit base mass h0"); ax[0].set_ylabel("critical honest discrimination J*")
    ax[0].set_title("Takeover boundary (collapse below the curve)"); ax[0].legend()
    lab = np.array(labels)
    ax[1].scatter(np.array(offl)[~lab], np.array(slopes)[~lab], label="no collapse", alpha=.7)
    ax[1].scatter(np.array(offl)[lab], np.array(slopes)[lab], label="collapse", alpha=.7, marker="x")
    ax[1].axhline(0, color="k", lw=.5)
    ax[1].set_xlabel("log(false/true accepts) at step 0 (offline audit)")
    ax[1].set_ylabel("slope of log(false/true accepts), steps 0-100")
    ax[1].set_title("Early-warning audit"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(R, "fig7_boundary_earlywarning.png"), dpi=150)
