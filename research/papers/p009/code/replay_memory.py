"""Revision (round 1), Experiment D': does the capacity law plus measured demand predict when an ideal
fixed-state memory fails on REAL token streams (Zipfian keys, repeated writes, corpus-sized vocabulary)?

For random windows of T tokens of a real document we build the additive outer-product state
  M = sum_writes u_b k_a^T,  k_a random unit vector in R^{d_k} per token type, u_b random unit code in R^{d_v}
  per token type (codes for the WHOLE corpus vocabulary V are the competitors), d_v = 64, S = 64 d_k,
under two write policies:
  every-token : every bigram (a, b) of the window is written, with multiplicity (an unselective recurrence)
  oracle      : each context-specific recurring bigram (the N_ctx set, definition D) is written once
Queries: the N_ctx set of the window (bigrams occurring >= 2 times with a context-specific successor),
evaluated at the end of the window (up to 200 sampled per window). A query (a, b) succeeds if b scores higher
than every code that is NOT an in-window successor of a (legitimate ambiguity between several in-window
successors of a is not counted as failure; an exact memory scores 1.0 under both policies).
We sweep S = 2^10 .. 2^19 and find S_half(T), the state at which recall of the N_ctx set reaches 50%, and
compare it with the prediction of Eq. (4): S_req = 2 ln V * D(T), D = T (every-token) or D = |N_ctx set|
(oracle).  Definitions: D0 (in-sample top-1, original) and D3 (held-out smoothed bigram p < 0.01).
Output: results/replay_memory.json
"""
import os, json, collections, numpy as np
from recall_demand import load_prose, load_code
from recall_demand_v2 import Stats

ROOT = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(ROOT, "..", "results")
rng = np.random.default_rng(3)
DV = 64
SS = [2 ** e for e in range(10, 21)]
SHAPES = {"dv64": {S: (S // 64, 64) for S in SS},                          # value width of one head
          "square": {S: (int(round(S ** .5)),) * 2 for S in SS if int(round(S ** .5)) ** 2 == S}}  # d_k = d_v


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


ZG = np.linspace(-8, 8, 801)
PHI = np.exp(-ZG ** 2 / 2) / np.sqrt(2 * np.pi)


def theory_vec(sig, sc2, sw2, V):
    """Generalised Eq. (1) per query: signal sig, correct-score variance sc2, competitor variance sw2."""
    from scipy.special import log_ndtr
    x = (sig[:, None] + np.sqrt(sc2)[:, None] * ZG[None, :]) / np.sqrt(sw2)[:, None]
    return np.trapezoid(PHI[None, :] * np.exp((V - 1) * log_ndtr(x)), ZG, axis=1)


def run_corpus(docs, name, Ts, nwin):
    lens = np.array([len(d) for d in docs])
    types = sorted(set(t for d in docs for t in d)); V = len(types); idx = {t: i for i, t in enumerate(types)}
    order = np.argsort(-lens); fold = np.zeros(len(docs), int); fold[order[1::2]] = 1
    st = [Stats([d for d, f in zip(docs, fold) if f == k]) for k in (0, 1)]; ins = Stats(docs)
    dvs = sorted(set(dv for sh in SHAPES.values() for (_, dv) in sh.values()))
    Ud = {dv: unit(rng.standard_normal((V, dv)).astype(np.float32)) for dv in dvs}
    nq = 200 if V < 60000 else 100
    out = []
    for T in Ts:
        elig = np.where(lens >= T)[0]
        for w_i in range(nwin):
            di = int(rng.choice(elig)); d = docs[di]; s0 = int(rng.integers(0, len(d) - T + 1))
            w = [idx[t] for t in d[s0:s0 + T]]; wt = d[s0:s0 + T]
            bg = list(zip(w, w[1:])); c = collections.Counter(bg)
            succ = collections.defaultdict(set)
            for a, b in c:
                succ[a].add(b)
            ho = st[1 - fold[di]]
            sets = {}
            tokpair = {}
            for (a, b), n in c.items():
                if n >= 2:
                    ta, tb = types[a], types[b]
                    tokpair[(a, b)] = (ins.top1.get(ta) != tb, ho.p(ta, tb, V) < 0.01)
            sets["D0"] = [p for p, f in tokpair.items() if f[0]]
            sets["D3"] = [p for p, f in tokpair.items() if f[1]]
            wtypes = sorted(set(w)); kpos = {t: i for i, t in enumerate(wtypes)}
            # per-key weight structure for the generalised Eq. (1): every-token weights = counts, oracle weights = 1
            def struct(weights):
                tot = sum(x * x for x in weights.values()); per = collections.defaultdict(float)
                for (a, b), x in weights.items():
                    per[a] += x * x
                return tot, per
            st_every = struct(c)
            A_all = np.array([kpos[a] for a, b in bg]); B_all = np.array([b for a, b in bg])
            for shape, S, (dk, dv) in [(sh, S, x) for sh in SHAPES for S, x in SHAPES[sh].items()]:
                U = Ud[dv]
                Kw = unit(rng.standard_normal((len(wtypes), dk)).astype(np.float32))
                M_every = (U[B_all].T @ Kw[A_all]) if len(bg) else None       # d_v x d_k, with multiplicity
                rec = {"T": T, "doc": di, "S": S, "dk": dk, "dv": dv, "shape": shape, "V": V}
                import time as _t; _t0 = _t.time()
                for D in ("D0", "D3"):
                    Q = sets[D]
                    rec[f"n_ctx_{D}"] = len(Q)
                    if not Q:
                        continue
                    qa = np.array([kpos[a] for a, b in Q]); qb = np.array([b for a, b in Q])
                    M_or = U[qb].T @ Kw[qa]
                    st_or = struct({p_: 1.0 for p_ in Q})
                    sel = rng.permutation(len(Q))[:nq]
                    for pol, M in (("every", M_every), ("oracle", M_or)):
                        tot, per = st_every if pol == "every" else st_or
                        wq = np.array([(c[Q[i]] if pol == "every" else 1.0) for i in sel], float)
                        own = np.array([per[Q[i][0]] for i in sel]) - wq ** 2        # same-key other successors
                        G = tot - np.array([per[Q[i][0]] for i in sel])               # other keys
                        sc2 = own / dv + G / (dk * dv) + 1e-12
                        sw2 = wq ** 2 / dv + own / dv + G / (dk * dv)
                        rec[f"theory_{pol}_{D}"] = float(theory_vec(wq, sc2, sw2, V).mean())
                        R = Kw[qa[sel]] @ M.T                                  # q x d_v
                        sc = R @ U.T                                           # q x V
                        ok = 0
                        for r_i, q_i in enumerate(sel):
                            a_t = Q[q_i][0]; b = qb[q_i]
                            row = sc[r_i]
                            others = [x for x in succ[a_t] if x != b]
                            if others:
                                row = row.copy(); row[others] = -np.inf
                            ok += int(np.argmax(row) == b)
                        rec[f"acc_{pol}_{D}"] = ok / len(sel)
                rec["sec"] = _t.time() - _t0
                out.append(rec)
            last = [r for r in out if r["T"] == T and r["doc"] == di]
            print(name, T, w_i, {k: v for k, v in out[-1].items() if k.startswith("n_ctx")},
                  [(r["shape"][0], r["S"], round(r.get("acc_every_D3", -1), 2), round(r.get("theory_every_D3", -1), 2), round(r.get("acc_oracle_D3", -1), 2), round(r.get("theory_oracle_D3", -1), 2)) for r in last], round(sum(r["sec"] for r in last)), flush=True)
    return {"V": V, "records": out}


def s_half(Ss, accs):
    pts = [(s_, a_) for s_, a_ in zip(Ss, accs) if np.isfinite(a_)]
    for (s0, a0), (s1, a1) in zip(pts, pts[1:]):
        if a0 < 0.5 <= a1:
            f = (0.5 - a0) / (a1 - a0)
            return float(np.exp(np.log(s0) + f * (np.log(s1) - np.log(s0))))
    return None


def summarise(res):
    V = res["V"]; lnV = np.log(V); summ = []
    recs = res["records"]
    for T in sorted(set(r["T"] for r in recs)):
        for shape in SHAPES:
          Ss = sorted(SHAPES[shape])
          for D in ("D0", "D3"):
            for pol in ("every", "oracle"):
                accs = [np.mean([r[f"acc_{pol}_{D}"] for r in recs if r["T"] == T and r["S"] == S and r["shape"] == shape
                                 and f"acc_{pol}_{D}" in r] or [np.nan]) for S in Ss]
                nctx = np.mean([r[f"n_ctx_{D}"] for r in recs if r["T"] == T and r["S"] == Ss[0] and r["shape"] == shape])
                pred = 2 * lnV * (T - 1 if pol == "every" else nctx)
                sh = s_half(Ss, accs)
                th = [np.mean([r[f"theory_{pol}_{D}"] for r in recs if r["T"] == T and r["S"] == S and r["shape"] == shape
                               and f"theory_{pol}_{D}" in r] or [np.nan]) for S in Ss]
                summ.append({"T": T, "shape": shape, "def": D, "policy": pol, "acc_by_S": dict(zip(Ss, map(float, accs))),
                             "theory_by_S": dict(zip(Ss, map(float, th))),
                             "mean_abs_err_theory": float(np.nanmean(np.abs(np.array(accs) - np.array(th)))),
                             "acc_max": float(np.nanmax(accs)),
                             "mean_n_ctx": float(nctx), "S_half_measured": sh, "S_req_predicted_eq4": float(pred),
                             "ratio_measured_over_predicted": (sh / pred) if sh else None})
    return summ


def main():
    out = {}
    for name, docs in (("prose", load_prose()), ("code", load_code())):
        r = run_corpus(docs, name, Ts=[1024, 4096, 16384], nwin=5)
        out[name] = {"V": r["V"], "records": r["records"], "summary": summarise(r)}
        for s in out[name]["summary"]:
            print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k != "acc_by_S"}, flush=True)
        json.dump(out, open(os.path.join(RES, "replay_memory.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
