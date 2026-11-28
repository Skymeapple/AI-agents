"""Exp 4: communication (debate) vs voting, seeded with REAL initial answers.
(a) Exact Moran / biased-voter consensus probability per item, as a function of truth advantage lambda;
    crossover lambda* where consensus accuracy equals plurality-vote accuracy.
(b) Finite-round synchronous debate simulation with stubbornness s, majority-conformity c and truth
    advantage lambda; final plurality after R rounds.
(c) Validation of the flip-ratio estimator lambda_hat = #(wrong->correct) / #(correct->wrong)."""
import json
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import load, plurality, moran_fix

d = load()
A, y, nopt, cat, models = d["A"], d["y"], d["nopt"], d["cat"], d["models"]
n, m = A.shape
C = A == y[:, None]
acc = C.mean(0)
order = np.argsort(-acc)
rng = np.random.default_rng(4)

teams = {
    "peer13 (acc .60-.80)": list(np.nonzero((acc >= 0.60) & (acc <= 0.80))[0]),
    "top5 (incl. dominant)": list(order[:5]),
    "top5 excl. dominant": list(order[1:6]),
    "weak5 (acc .35-.45)": list(np.nonzero((acc >= 0.35) & (acc <= 0.45))[0][:5]),
}
LAMS = np.exp(np.linspace(np.log(0.25), np.log(50), 120))


def consensus_acc(team, lam, idx=None):
    T = A[:, team] if idx is None else A[idx][:, team]
    yy = y if idx is None else y[idx]
    k = (T == yy[:, None]).sum(1)
    return float(moran_fix(k, len(team), lam).mean())


def lam_star(team, target, idx=None):
    f = lambda L: consensus_acc(team, L, idx) - target
    lo, hi = 1e-3, 1e4
    if f(lo) > 0:
        return lo
    if f(hi) < 0:
        return float("inf")
    return float(np.exp(brentq(lambda t: f(np.exp(t)), np.log(lo), np.log(hi), xtol=1e-8)))


out = {}
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6))
for ti, (tname, team) in enumerate(teams.items()):
    N = len(team)
    vote = float(np.mean([(plurality(A[:, team], rng) == y).mean() for _ in range(5)]))
    k = C[:, team].sum(1)
    curve = [consensus_acc(team, L) for L in LAMS]
    ls = lam_star(team, vote)
    best = float(acc[team].max())
    ls_best = lam_star(team, best)
    # per-item decomposition at lambda = 1: debate helps where k < N/2, hurts where k > N/2
    maj = k > N / 2
    neutral = moran_fix(k, N, 1.0)
    per_cat = {}
    for c in np.unique(cat):
        idx = np.nonzero(cat == c)[0]
        v = float((plurality(A[idx][:, team], rng) == y[idx]).mean())
        per_cat[str(c)] = dict(vote=v, lam_star=lam_star(team, v, idx), coverage=float((k[idx] > 0).mean()))
    out[tname] = dict(members=[models[j] for j in team], N=N, vote_acc=vote, best_member=best,
                      mean_member=float(acc[team].mean()), coverage=float((k > 0).mean()),
                      consensus_acc_lambda1=float(neutral.mean()), lam_star_vs_vote=ls, lam_star_vs_best=ls_best,
                      consensus_acc_at={str(L): consensus_acc(team, L) for L in [0.5, 0.8, 1.0, 1.25, 1.5, 2, 3, 5, 10]},
                      frac_items_majority_correct=float(maj.mean()), frac_items_minority_correct=float(((k > 0) & ~maj).mean()),
                      neutral_loss_on_majority_items=float(np.mean(np.where(maj, 1 - neutral, 0))),
                      neutral_gain_on_minority_items=float(np.mean(np.where(~maj, neutral, 0))),
                      per_category=per_cat)
    print(tname, {kk: vv for kk, vv in out[tname].items() if kk not in ("members", "per_category", "consensus_acc_at")})
    axs[0].plot(LAMS, curve, c=f"C{ti}", label=f"{tname}: consensus")
    axs[0].axhline(vote, c=f"C{ti}", ls="--", lw=1)
    axs[0].axhline(out[tname]["coverage"], c=f"C{ti}", ls=":", lw=1)
axs[0].set_xscale("log"); axs[0].axvline(1, c="k", lw=.8)
axs[0].set_xlabel(r"truth advantage $\lambda$ (persuasiveness of correct answer)")
axs[0].set_ylabel("accuracy"); axs[0].set_title("Consensus (Moran) vs plurality vote (dashed); coverage (dotted)")
axs[0].legend(fontsize=7)

# per-category lambda* for peer11
pc = out["peer13 (acc .60-.80)"]["per_category"]
cs = sorted(pc, key=lambda c: pc[c]["lam_star"])
axs[1].barh(range(len(cs)), [pc[c]["lam_star"] for c in cs], color="C0")
axs[1].set_yticks(range(len(cs))); axs[1].set_yticklabels(cs, fontsize=8)
axs[1].set_xlim(1.0, 1.05 * max(pc[c]["lam_star"] for c in cs)); axs[1].set_xlabel(r"$\lambda^*$ needed for debate consensus to match voting (peer13)")
plt.tight_layout(); plt.savefig("../results/fig5_debate_crossover.png", dpi=140)


# (b) finite-round synchronous debate simulation
def simulate(team, lam, s, c, R, reps=3, seed=0):
    g = np.random.default_rng(seed)
    T0 = A[:, team].astype(np.int16)
    N = len(team)
    accs = np.zeros(R + 1)
    wc = cw = 0
    for _ in range(reps):
        T = T0.copy()
        accs[0] += (plurality(T, g) == y).mean()
        for r in range(1, R + 1):
            corr = T == y[:, None]
            w = np.where(T < 0, 0.0, np.where(corr, lam, 1.0))          # persuasion weights
            tot = w.sum(1, keepdims=True)
            # sample a persuader for every agent
            cum = np.cumsum(w, 1) / np.maximum(tot, 1e-12)
            u = g.random((n, N))
            pick = (u[:, :, None] > cum[:, None, :]).sum(2)
            pick = np.minimum(pick, N - 1)
            copied = np.take_along_axis(T, pick, 1)
            copied = np.where(tot > 0, copied, T)
            plur = plurality(T, g)[:, None].repeat(N, 1)
            ru = g.random((n, N))
            new = np.where(ru < s, T, np.where(ru < s + (1 - s) * c, plur, copied)).astype(np.int16)
            newc = new == y[:, None]
            wc += int((~corr & newc).sum()); cw += int((corr & ~newc).sum())
            T = new
            accs[r] += (plurality(T, g) == y).mean()
    return (accs / reps).tolist(), (wc / cw if cw > 0 else float("inf"))


sim = {}
R = 8
for tname in ["peer13 (acc .60-.80)", "top5 (incl. dominant)"]:
    team = teams[tname]
    sim[tname] = {}
    for lam in [0.8, 1.0, 1.5, 3.0]:
        for c in [0.0, 0.5]:
            curve, lam_hat = simulate(team, lam, s=0.5, c=c, R=R, seed=int(lam * 10 + c * 100))
            sim[tname][f"lam={lam},c={c}"] = dict(acc_by_round=curve, lambda_hat_flip_ratio=lam_hat)
            print(tname, lam, c, [round(a, 4) for a in curve], "lam_hat", round(lam_hat, 3))
out["finite_round_sim"] = dict(s=0.5, R=R, results=sim)
json.dump(out, open("../results/exp4_debate.json", "w"), indent=2)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.3), sharey=False)
for ax, tname in zip(axs, sim):
    for i, (key, v) in enumerate(sim[tname].items()):
        ax.plot(range(R + 1), v["acc_by_round"], marker="o", ms=3, ls="-" if "c=0.0" in key else "--",
                c=f"C{i // 2}", label=key)
    ax.set_xlabel("debate round"); ax.set_ylabel("plurality accuracy after round"); ax.set_title(tname)
axs[0].legend(fontsize=7, ncol=2)
plt.tight_layout(); plt.savefig("../results/fig6_debate_rounds.png", dpi=140)
