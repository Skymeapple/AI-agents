"""E1: Knowledge items under synthetic amplification (Zipfian exposure model).

Model: K facts with Zipf(alpha) prompt frequencies p_i. A human corpus of N
samples gives Poisson(N p_i) exposures of fact i. Each exposure is independently
"stored" by the learner with probability s (per-exposure retention). A stored
fact is answered correctly; an unstored fact is guessed (1/k correct).

Synthetic amplification: a generator with retention s_gen re-expresses each human
exposure r times (fresh paraphrases, each an independent exposure for the
learner). Source-grounded generation (document in context) has s_gen = 1.
Self-recall generation (generator = the learner itself) has s_gen = s.

Proposition 2 (checked here numerically and by Monte Carlo):
  P(fact i unstored) = exp(-N p_i * [1-(1-s)^{1+r}])           (s_gen = 1)
  => amplified corpus == plain corpus of size N * M(r),
     M(r) = (1-(1-s)^{1+r}) / s  ->  1/s as r -> inf.
With a generator of retention s_gen<1 the multiplier is
  M(r) = [1 - (1-s_gen)(1-s)... ] -- computed exactly below by enumeration.
With fidelity phi<1 (a fraction 1-phi of paraphrases carry a systematic wrong
answer) accuracy is computed with a majority-of-stored-exposures readout.
"""
import json, os, numpy as np
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

K, KCLS = 100_000, 10

def zipf(alpha, K=K):
    w = np.arange(1, K + 1, dtype=float) ** (-alpha)
    return w / w.sum()

def acc_plain(N, s, p):
    return 1 - (1 - 1 / KCLS) * np.sum(p * np.exp(-s * N * p))

def acc_amp_closed(N, s, r, p, s_gen=1.0):
    # each human exposure: generator retains it w.p. s_gen; if retained it emits r
    # paraphrases. Learner stores from original (prob s) or from any paraphrase.
    # P(a given human exposure fails to lead to storage)
    fail = (1 - s) * ((1 - s_gen) + s_gen * (1 - s) ** r)
    return 1 - (1 - 1 / KCLS) * np.sum(p * np.exp(-N * p * (1 - fail))), (1 - fail) / s

def n_equiv(acc_target, s, p, N_hi=1e12):
    f = lambda logN: acc_plain(np.exp(logN), s, p) - acc_target
    lo, hi = np.log(1.0), np.log(N_hi)
    if f(hi) < 0:
        return np.inf
    if f(lo) > 0:
        return 1.0
    return float(np.exp(optimize.brentq(f, lo, hi, xtol=1e-10)))

def monte_carlo(N, s, r, p, s_gen=1.0, reps=3):
    accs = []
    for _ in range(reps):
        c = rng.poisson(N * p)
        # per human exposure: storage events
        tot = c.sum()
        idx = np.repeat(np.arange(len(p)), c)
        st0 = rng.random(tot) < s
        gen_ok = rng.random(tot) < s_gen
        # number of paraphrases stored ~ Binomial(r, s) if generator retained
        stp = (rng.binomial(r, s, size=tot) > 0) & gen_ok
        stored_exp = st0 | stp
        stored = np.zeros(len(p), bool)
        stored[idx[stored_exp]] = True
        acc = np.sum(p[stored]) + (1 / KCLS) * np.sum(p[~stored])
        accs.append(acc)
    return float(np.mean(accs)), float(np.std(accs))

def acc_fidelity(N, s, r, phi, p, reps=4):
    """Source-grounded amplification with fidelity phi (Monte Carlo, exact sampling).
    Each human exposure of fact i: original stored w.p. s (correct label); r paraphrases,
    each correct w.p. phi or carrying one systematic wrong label w.p. 1-phi, each stored
    w.p. s. Readout: label with most stored exposures; tie -> coin flip; none -> guess 1/k."""
    accs = []
    for _ in range(reps):
        c = rng.poisson(N * p)
        idx = np.repeat(np.arange(len(p)), c)
        tot = len(idx)
        orig = (rng.random(tot) < s).astype(np.int64)
        m = rng.multinomial(r, [s * phi, s * (1 - phi), 1 - s], size=tot) if r > 0 else np.zeros((tot, 3), np.int64)
        xc = np.bincount(idx, weights=orig + m[:, 0], minlength=len(p))
        xw = np.bincount(idx, weights=m[:, 1], minlength=len(p))
        acc_i = np.where(xc > xw, 1.0, 0.0) + np.where((xc == xw) & (xc > 0), 0.5, 0.0) \
            + np.where((xc == 0) & (xw == 0), 1.0 / KCLS, 0.0)
        accs.append(float(np.sum(p * acc_i)))
    return float(np.mean(accs))

results = {}
# ---- (a) closed form vs Monte Carlo, multiplier M(r) ----
p = zipf(1.1)
N = 2e5
rows = []
for s in [0.05, 0.2, 0.5]:
    for r in [0, 1, 4, 16, 64]:
        a_cf, M = acc_amp_closed(N, s, r, p)
        a_mc, sd = monte_carlo(N, s, r, p)
        Neq = n_equiv(a_cf, s, p)
        rows.append(dict(s=s, r=r, acc_closed=a_cf, acc_mc=a_mc, mc_sd=sd,
                         M_theory=M, M_from_Nequiv=Neq / N))
results["closed_vs_mc"] = rows

# ---- (b) M is independent of alpha and N (check) ----
inv = []
for alpha in [0.8, 1.1, 1.5]:
    pp = zipf(alpha)
    for NN in [1e4, 1e6]:
        for s in [0.1]:
            a_cf, M = acc_amp_closed(NN, s, 16, pp)
            inv.append(dict(alpha=alpha, N=NN, s=s, r=16, M_theory=M,
                            M_from_Nequiv=n_equiv(a_cf, s, pp) / NN))
results["invariance"] = inv

# ---- (c) generator retention s_gen (self-recall vs teacher vs source-grounded) ----
gen = []
s = 0.1
for s_gen in [0.1, 0.3, 0.6, 1.0]:
    for r in [1, 4, 16, 64, 256]:
        a_cf, M = acc_amp_closed(N, s, r, p, s_gen=s_gen)
        gen.append(dict(s=s, s_gen=s_gen, r=r, M=M, M_limit=(1 - (1 - s) * (1 - s_gen)) / s))
results["generator_retention"] = gen
# Self-recall check: generator = learner itself paraphrasing only facts it already
# stores adds NOTHING to accuracy (stored facts already answered correctly).
results["self_recall_note"] = ("When the generator is the learner recalling its own stored facts, "
                               "only already-stored facts are re-exposed, so accuracy is unchanged: M=1.")

# ---- (d) fidelity ----
fid = []
s = 0.1
for phi in [1.0, 0.9, 0.7, 0.5, 0.4]:
    for r in [0, 1, 4, 16, 64, 256]:
        a = acc_fidelity(N, s, r, phi, p)
        fid.append(dict(s=s, phi=phi, r=r, acc=a, M_eff=n_equiv(a, s, p) / N))
results["fidelity"] = fid
results["ceilings"] = dict(N=N, alpha=1.1, plain_s0_1=acc_plain(N, 0.1, p),
                           coverage_ceiling=acc_plain(N, 1.0, p))

with open(os.path.join(OUT, "e1_knowledge.json"), "w") as f:
    json.dump(results, f, indent=1)

# ---- figures ----
fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for s in [0.05, 0.2, 0.5]:
    rr = [x for x in rows if x["s"] == s]
    ax[0].plot([x["r"] + 1 for x in rr], [x["M_from_Nequiv"] for x in rr], "o", label=f"s={s} (N-equiv.)")
    rg = np.logspace(0, np.log10(65), 50)
    ax[0].plot(rg, (1 - (1 - s) ** rg) / s, "-", color=ax[0].lines[-1].get_color(), alpha=.6)
    ax[0].axhline(1 / s, ls=":", color=ax[0].lines[-1].get_color())
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("1 + r (exposures per human exposure)"); ax[0].set_ylabel("data multiplier M")
ax[0].set_title("(a) source-grounded amplification\nlines: theory, dotted: 1/s ceiling"); ax[0].legend(fontsize=7)
for s_gen in [0.1, 0.3, 0.6, 1.0]:
    gg = [x for x in gen if x["s_gen"] == s_gen]
    ax[1].plot([x["r"] for x in gg], [x["M"] for x in gg], "o-", label=f"s_gen={s_gen}")
ax[1].set_xscale("log"); ax[1].set_xlabel("r"); ax[1].set_ylabel("M (learner s=0.1)")
ax[1].set_title("(b) generator retention caps M"); ax[1].legend(fontsize=7)
for phi in [1.0, 0.9, 0.7, 0.5, 0.4]:
    ff = [x for x in fid if x["phi"] == phi]
    ax[2].plot([x["r"] + 1 for x in ff], [x["acc"] for x in ff], "o-", label=f"phi={phi}")
ax[2].axhline(results["ceilings"]["coverage_ceiling"], ls=":", c="k", label="coverage ceiling")
ax[2].axhline(results["ceilings"]["plain_s0_1"], ls="--", c="gray", label="no synthetic")
ax[2].set_xscale("log"); ax[2].set_xlabel("1 + r"); ax[2].set_ylabel("accuracy (Zipf-weighted)")
ax[2].set_title("(c) fidelity: hallucinated paraphrases"); ax[2].legend(fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig1_knowledge.png"), dpi=150)
print(json.dumps(results["closed_vs_mc"], indent=0)[:3000])
print(json.dumps(results["invariance"]))
print(json.dumps(results["ceilings"]))
for x in fid:
    print(x)
