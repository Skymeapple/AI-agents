"""E6: Verifier design experiments (addresses confounds in E2).

Part A (decorrelated grid). a in {0.25,0.5,1} x (b/a) in {0,0.2,0.5,0.8}: verifier
information, b/a, a-b and precision are no longer monotone in one variable.
Which predictor ranks accuracy gain? Information is computed two ways:
  - 'marginal'   : per round, I(C;V) at the round's empirical correctness rate q
  - 'conditional': per proposal, I(C_j;V_j) at q_j = learner's own probability of
                   the sampled label (a proxy for Thm. 2's conditional q_j);
                   by concavity this is <= the marginal version.
Part B (verifier cost / coverage). Each verifier is given a description budget
in bits.
  - knowledge: a LOOKUP verifier storing the labels of a random fraction kappa of
    the 400 items (3 bits each); exact on covered items, rejects otherwise.
  - skill: a PROGRAM verifier = the teacher (W*, b*) quantized to beta bits per
    parameter (168 parameters); accepts iff the quantized teacher agrees.
Same learner, loop and budgets as E2.
"""
import json, os, sys, time, numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import spearmanr
import e2_verified_selftraining as E

OUT = os.path.join(os.path.dirname(__file__), "..", "results")

def quantize(W, beta):
    if beta is None: return W
    m = np.abs(W).max()
    if beta == 1: return np.sign(W) * np.abs(W).mean()
    lv = 2 ** beta - 1
    return np.round((W / m + 1) / 2 * lv) / lv * 2 * m - m

def run_v(kind, n0, seed, vtype, p1, p2, rounds=E.ROUNDS, m=E.M_PROP):
    """vtype='ab': p1=a, p2=b.  vtype='cover': p1=kappa.  vtype='quant': p1=beta."""
    r = np.random.default_rng(seed)
    task = E.Task(kind, seed)
    Xt, yt, wt = task.test_set(np.random.default_rng(7 + seed))
    Xh = task.prompts(n0, r, human=True); yh = task.label(Xh)
    Wb = E.fit(Xh, yh); accs = [E.accuracy(Wb, Xt, yt, wt)]
    Xs = np.zeros((0, task.dim)); ys = np.zeros(0, int)
    bits_m = bits_c = 0.0; n_acc = n_acc_c = 0; acc_c = acc_w = n_c = n_w = 0
    vr = np.random.default_rng(10_000 + seed)
    if vtype == "cover":
        covered = np.zeros(E.KITEMS, bool); covered[vr.permutation(E.KITEMS)[:int(round(p1 * E.KITEMS))]] = True
    if vtype == "quant":
        Wq = quantize(task.W, p1); bq = quantize(task.b0, p1)
    for t in range(rounds):
        Xp = task.prompts(m, r, human=False)
        P = E.predict_proba(Wb, Xp)
        u = r.random((m, 1)); yp = np.minimum((u > P.cumsum(1)).sum(1), E.KC - 1)
        correct = yp == task.label(Xp)
        qj = P[np.arange(m), yp]
        if vtype == "ab":
            a, b = p1, p2
            accm = np.where(correct, r.random(m) < a, r.random(m) < b)
            bits_m += m * E.mutual_info_cv(correct.mean(), a, b)
            bits_c += E.mutual_info_cv(qj, a, b).sum()
        elif vtype == "cover":
            cov = covered[np.argmax(Xp, 1)]
            accm = cov & correct
            bits_m += cov.sum() * E.mutual_info_cv(correct[cov].mean() if cov.any() else 0.5, 1, 0)
            bits_c += E.mutual_info_cv(qj[cov], 1, 0).sum()
        else:
            accm = yp == np.argmax(Xp @ Wq + bq, 1)
            # verifier is deterministic given x; its information about C is I(C;V) of the empirical channel
            a_e = accm[correct].mean() if correct.any() else 0; b_e = accm[~correct].mean() if (~correct).any() else 0
            bits_m += m * E.mutual_info_cv(correct.mean(), a_e, b_e)
            bits_c += E.mutual_info_cv(qj, a_e, b_e).sum()
        acc_c += (accm & correct).sum(); acc_w += (accm & ~correct).sum(); n_c += correct.sum(); n_w += (~correct).sum()
        n_acc += accm.sum(); n_acc_c += (accm & correct).sum()
        Xs = np.vstack([Xs, Xp[accm]]); ys = np.concatenate([ys, yp[accm]])
        Wb = E.fit(np.vstack([Xh, Xs]), np.concatenate([yh, ys]))
        accs.append(E.accuracy(Wb, Xt, yt, wt))
    out = dict(kind=kind, n0=n0, seed=seed, vtype=vtype, p1=p1, p2=p2, accs=accs, bits_marg=bits_m, bits_cond=bits_c,
               n_accepted=int(n_acc), precision=float(n_acc_c / max(n_acc, 1)),
               a_emp=float(acc_c / max(n_c, 1)), b_emp=float(acc_w / max(n_w, 1)))
    if vtype == "cover":
        seen = np.zeros(E.KITEMS, bool); seen[np.argmax(Xh, 1)] = True
        known = seen | covered
        out["coverage_bound"] = float(np.sum(task.pz * known) + np.sum(task.pz * ~known) / E.KC)
        out["verifier_bits_cost"] = float(covered.sum() * np.log2(E.KC))
    if vtype == "quant":
        out["verifier_bits_cost"] = float((E.D + 1) * E.KC * (p1 if p1 else 32))
        Xq = np.random.default_rng(99).choice([-1.0, 1.0], size=(20000, E.D))
        out["verifier_agreement"] = float(np.mean(np.argmax(Xq @ Wq + bq, 1) == task.label(Xq)))
    return out

def _star(a): return run_v(*a)

if __name__ == "__main__":
    t0 = time.time()
    SEEDS = [0, 1, 2, 3, 4]
    jobs = []
    for kind, n0 in [("skill", 64), ("knowledge", 200)]:
        for a in [0.25, 0.5, 1.0]:
            for ratio in [0.0, 0.2, 0.5, 0.8]:
                for s in SEEDS: jobs.append((kind, n0, s, "ab", a, round(a * ratio, 4)))
    for kappa in [0.0, 0.125, 0.25, 0.5, 1.0]:
        for s in SEEDS: jobs.append(("knowledge", 200, s, "cover", kappa, None))
    for beta in [1, 2, 3, 4, None]:
        for s in SEEDS: jobs.append(("skill", 64, s, "quant", beta, None))
    jobs.sort(key=lambda j: j[0] != "knowledge")
    with ProcessPoolExecutor(4) as ex:
        res = list(ex.map(_star, jobs, chunksize=1))
    json.dump(dict(runs=res), open(os.path.join(OUT, "e6_verifier_design.json"), "w"), indent=1)
    print("done %.1fs, %d runs" % (time.time() - t0, len(res)))
