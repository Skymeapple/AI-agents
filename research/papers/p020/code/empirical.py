"""Empirical analysis of authorship information in human vs LLM idea populations.

Data: Nath, Dayan & Stevenson (ICCC 2024), github.com/surabhisnath/Creative_Process
  csvs/data_humans.csv (220 Dutch-speaking adults; responses translated to English)
  csvs/data_LLMs.csv   (8 LLMs x 11 temperatures x 5 repetitions = 440 sequences)
Tasks: 1 = verbal fluency (animals; control), 2 = AUT brick, 3 = AUT paperclip.
Ideas are represented by (a) the shared semantic category released with the data,
(b) the cleaned response string (finer, phrasing-sensitive).

Outputs -> ../results/
"""
import json, os, time, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from infodecomp import plugin_mi, excess_mi, cond_mi, H_counts, permute_within, list_perm_mi
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)
SEED = 7
rng = np.random.default_rng(SEED)
TASKS = {1: "fluency (animals)", 2: "AUT brick", 3: "AUT paperclip"}
NPERM = 500

H = pd.read_csv(os.path.join(DATA, "data_humans.csv"))
L = pd.read_csv(os.path.join(DATA, "data_LLMs.csv"))
H["agent"] = "H" + H.pid.astype(str)
L["agent"] = L.model + "|" + L.temp.astype(str) + "|" + L.rep.astype(str)
L["mshort"] = L.model.str.split("/").str[-1]
for D in (H, L):
    D["idea_str"] = D.response.astype(str).str.lower().str.strip().str.replace(r"\s+", " ", regex=True)
    D.sort_values(["agent", "order"], inplace=True)

log = {}


def first_k(D, k, random_k=False, seed=0):
    """Keep agents with >=k responses; take their first k (or a random k)."""
    g = D.groupby("agent")
    keep = g.size()[lambda s: s >= k].index
    D = D[D.agent.isin(keep)]
    if random_k:
        return D.groupby("agent").sample(n=k, random_state=seed).sort_values(["agent", "order"])
    return D.groupby("agent", group_keys=False).head(k)


def encode(*cols):
    allv = pd.concat(cols).astype(str)
    cats = {v: i for i, v in enumerate(sorted(allv.unique()))}
    return [c.astype(str).map(cats).to_numpy() for c in cols], len(cats)


def mi_block(D, col, K=None, n_perm=NPERM, strata=None, seed=0):
    (x,), K0 = encode(D[col])
    res = excess_mi(x, D.agent.to_numpy(), K0, n_perm=n_perm, rng=seed,
                    strata=None if strata is None else pd.factorize(D[strata])[0])
    res.update(n_agents=int(D.agent.nunique()), N=int(len(D)), K=int(K0))
    return res


# ------------------------------------------------------------------ A. naive estimator (pitfall)
# i.i.d.-response permutation null; INVALID under within-list dependence. Reported to show the pitfall.
t0 = time.time()
K_PER = 10
naive = []
for task in (1, 2, 3):
    h = first_k(H[H.task == task], K_PER)
    l = first_k(L[L.task == task], K_PER)
    nH = h.agent.nunique()
    for level in ("category", "idea_str"):
        r = mi_block(h, level, seed=1); r.update(task=task, pop="Humans", level=level); naive.append(r)
        reps = []
        for s_ in range(20):
            sub = rng.choice(l.agent.unique(), size=nH, replace=False)
            reps.append(mi_block(l[l.agent.isin(sub)], level, n_perm=200, seed=100 + s_))
        r = {k: float(np.mean([q[k] for q in reps])) for k in ("I", "null_mean", "I_star", "H", "pi_star", "p")}
        r.update(n_agents=nH, task=task, pop="LLM ensemble", level=level); naive.append(r)
    # identical replicates: T=0 lists of the same model (true between-replicate information is 0)
    l0 = l[l.temp == 0.0]
    (x,), K0 = encode(l0.category)
    r = excess_mi(x, pd.factorize(l0.agent)[0], K0, n_perm=NPERM, rng=3, strata=pd.factorize(l0.model)[0])
    r.update(task=task, pop="LLM T=0 replicates (truth: 0)", level="category", n_agents=int(l0.agent.nunique()))
    naive.append(r)
pd.DataFrame(naive).to_csv(os.path.join(OUT, "A_naive_iid_permutation.csv"), index=False)
print("A done", round(time.time() - t0), "s")


def list_table(D, col):
    """Encode responses; return x, list index per response, and a per-list dataframe."""
    (x,), K0 = encode(D[col])
    lid, lists = pd.factorize(D.agent)
    meta = D.drop_duplicates("agent").set_index("agent").loc[lists]
    return x, lid, meta.reset_index(), K0

# ------------------------------------------------------------------ B. LLM hierarchy, list-level nulls
hier = []
for task in (1, 2, 3):
    l = first_k(L[L.task == task], K_PER)
    for level in ("category", "idea_str"):
        x, lid, meta, K0 = list_table(l, level)
        mod = pd.factorize(meta.model)[0]
        cond = pd.factorize(meta.model + "|" + meta.temp.astype(str))[0]
        seq = np.arange(len(meta))
        Hx = H_counts(np.bincount(x, minlength=K0))
        rows = {
            "model": list_perm_mi(x, lid, mod, K0, n_perm=NPERM, rng=11),
            "temperature|model": list_perm_mi(x, lid, cond, K0, n_perm=NPERM, rng=12, strata_of_list=mod),
        }
        # replicate term: no list-level null exists (lists are the units); report plug-in only
        rows["replicate|model,temp (plug-in floor)"] = dict(I=cond_mi(x, seq[lid], cond[lid], K0), H=Hx)
        rows["total (list)"] = dict(I=plugin_mi(x, lid, K0, len(meta))[0], H=Hx)
        G = Hx - rows["total (list)"]["I"]
        for k_, v in rows.items():
            v.update(task=task, level=level, term=k_, K=K0, N=len(x), n_lists=len(meta), G_within=G)
            hier.append(v)
pd.DataFrame(hier).to_csv(os.path.join(OUT, "B_llm_hierarchy_listperm.csv"), index=False)
print("B done", round(time.time() - t0), "s")

# ------------------------------------------------------------------ C. temperature sweep
temp_rows = []
for task in (2, 3):
    l = first_k(L[L.task == task], K_PER)
    (xall,), K0 = encode(l.category)
    l = l.assign(x=xall)
    for T, lt in l.groupby("temp"):
        hs = lt.groupby("agent").x.apply(lambda v: H_counts(np.bincount(v, minlength=K0)))
        ds = lt.groupby("agent").x.nunique()
        # replicate floor: plug-in JSD among the replicate lists of each model, averaged over models
        floors = []
        for m, lm in lt.groupby("model"):
            if lm.agent.nunique() >= 2:
                floors.append(plugin_mi(lm.x.to_numpy(), pd.factorize(lm.agent)[0], K0, lm.agent.nunique())[0])
        xx, lid, meta, _ = list_table(lt.assign(category=lt.x), "category")
        r_mod = list_perm_mi(lt.x.to_numpy(), lid, pd.factorize(meta.model)[0], K0, n_perm=300, rng=22)
        temp_rows.append(dict(task=task, temp=T, orig_mean=float(lt.originality.mean()),
                              G_within=float(hs.mean()), distinct_within=float(ds.mean()),
                              replicate_floor=float(np.mean(floors)), I_model_star=r_mod["I_star"],
                              I_model_p=r_mod["p"], n_lists=int(lt.agent.nunique())))
TT = pd.DataFrame(temp_rows)
TT.to_csv(os.path.join(OUT, "C_temperature_sweep.csv"), index=False)
cor = {}
for task in (2, 3):
    t = TT[TT.task == task]
    cor[task] = {f"r_T_{c}": float(stats.spearmanr(t.temp, t[c])[0]) for c in
                 ("orig_mean", "G_within", "distinct_within", "replicate_floor", "I_model_star")}
    cor[task].update({f"p_T_{c}": float(stats.spearmanr(t.temp, t[c])[1]) for c in
                      ("orig_mean", "G_within", "distinct_within", "replicate_floor", "I_model_star")})
log["C_spearman_across_temperature"] = cor
print("C done", round(time.time() - t0), "s")

# ------------------------------------------------------------------ D. list-divergence ladder (main)
def jsd_lists(xs_list, K0):
    """Plug-in JSD (bits) among equal-length lists = H(pooled) - mean H(list)."""
    pooled = np.bincount(np.concatenate(xs_list), minlength=K0)
    return H_counts(pooled) - np.mean([H_counts(np.bincount(v, minlength=K0)) for v in xs_list])


ladder = []
ladder_tests = []
for task in (1, 2, 3):
    for k in (8, 10, 12):
        h = first_k(H[H.task == task], k)
        l = first_k(L[L.task == task], k)
        for level in ("category", "idea_str"):
            (xh, xl), K0 = encode(h[level], l[level])
            hl = {a: xh[(h.agent == a).to_numpy()] for a in h.agent.unique()}
            ll = {a: xl[(l.agent == a).to_numpy()] for a in l.agent.unique()}
            lmeta = l.drop_duplicates("agent").set_index("agent")[["model", "temp", "rep"]]
            for g in (3, 5):
                draws = {}
                ha = list(hl)
                draws["Humans: g different people"] = [jsd_lists([hl[a] for a in rng.choice(ha, g, replace=False)], K0)
                                                       for _ in range(1500)]
                # LLM same condition (replicates)
                conds = lmeta.groupby(["model", "temp"]).groups
                same = [list(v) for v in conds.values() if len(v) >= g]
                for Tset, name in (((0.0,), "LLM: replicates, T=0"), (tuple(np.round(np.arange(0.1, 1.01, 0.1), 1)),
                                    "LLM: replicates, T in 0.1-1.0"), ((1.0,), "LLM: replicates, T=1.0")):
                    pool = [v for v in same if lmeta.loc[v[0], "temp"] in Tset]
                    draws[name] = [jsd_lists([ll[a] for a in rng.choice(pool[rng.integers(len(pool))], g, replace=False)], K0)
                                   for _ in range(1500)]
                # same model, different temperatures
                out = []
                models = lmeta.model.unique()
                for _ in range(1500):
                    m = models[rng.integers(len(models))]
                    lm = lmeta[lmeta.model == m]
                    temps = rng.choice(lm.temp.unique(), g, replace=False)
                    out.append(jsd_lists([ll[rng.choice(lm.index[lm.temp == t])] for t in temps], K0))
                draws["LLM: same model, g temperatures"] = out
                # same temperature, different models
                out = []
                for _ in range(1500):
                    T = lmeta.temp.unique()[rng.integers(lmeta.temp.nunique())]
                    lt = lmeta[lmeta.temp == T]
                    ms = rng.choice(lt.model.unique(), g, replace=False)
                    out.append(jsd_lists([ll[rng.choice(lt.index[lt.model == m])] for m in ms], K0))
                draws["LLM: same T, g models"] = out
                # different models and temperatures
                out = []
                for _ in range(1500):
                    ms = rng.choice(models, g, replace=False)
                    out.append(jsd_lists([ll[rng.choice(lmeta.index[lmeta.model == m])] for m in ms], K0))
                draws["LLM: g models, any T"] = out
                for name, v in draws.items():
                    v = np.array(v)
                    ladder.append(dict(task=task, k=k, level=level, g=g, structure=name, mean=v.mean(),
                                       q025=np.percentile(v, 2.5), q975=np.percentile(v, 97.5)))
                # disjoint-group test: humans vs LLM g-models (most diverse LLM structure)
                if g == 5:
                    perm = rng.permutation(ha)
                    hgroups = [jsd_lists([hl[a] for a in perm[i:i + g]], K0) for i in range(0, len(perm) - g + 1, g)]
                    # disjoint LLM groups: each list used once; one list per model per group
                    byM = {m: list(rng.permutation(lmeta.index[lmeta.model == m])) for m in models}
                    lgroups = []
                    while True:
                        avail = [m for m in models if byM[m]]
                        if len(avail) < g:
                            break
                        ms = rng.choice(avail, g, replace=False)
                        lgroups.append(jsd_lists([ll[byM[m].pop()] for m in ms], K0))
                    u = stats.mannwhitneyu(hgroups, lgroups)
                    ladder_tests.append(dict(task=task, k=k, level=level, n_h_groups=len(hgroups),
                                             n_l_groups=len(lgroups), human_mean=np.mean(hgroups),
                                             human_se=stats.sem(hgroups), llm_mean=np.mean(lgroups),
                                             llm_se=stats.sem(lgroups), U=u.statistic, p=u.pvalue))
pd.DataFrame(ladder).to_csv(os.path.join(OUT, "D_list_divergence_ladder.csv"), index=False)
pd.DataFrame(ladder_tests).to_csv(os.path.join(OUT, "D2_ladder_disjoint_tests.csv"), index=False)
print("D done", round(time.time() - t0), "s")

# ------------------------------------------------------------------ E. peer-normed originality
peer = []
R = 100
for task in (2, 3):
    h = first_k(H[H.task == task], K_PER)
    l = first_k(L[L.task == task], K_PER)
    (xh, xl), K0 = encode(h.category, l.category)
    h = h.assign(x=xh); l = l.assign(x=xl)
    hA = {a: d for a, d in h.groupby("agent")}
    lA = {a: d for a, d in l.groupby("agent")}
    hkeys, lkeys = list(hA), list(lA)

    def score(target, ref_keys, ref_dict):
        ref = pd.concat([ref_dict[k] for k in ref_keys])
        cnt = np.bincount(ref.x, minlength=K0) + 0.5
        p = cnt / cnt.sum()
        surpr = float(-np.log2(p[target.x]).mean())
        strs = set(ref.idea_str)
        uniq = float(np.mean([s_ not in strs for s_ in target.idea_str]))
        return surpr, uniq

    for a in hkeys:
        for nref, lab in ((R, "human peers (100)"), (54, "human peers (54)")):
            refs = rng.choice([k for k in hkeys if k != a], nref, replace=False)
            s_, u = score(hA[a], refs, hA)
            peer.append(dict(task=task, agent=a, pop="Human", norm=lab, surprisal=s_, unique=u,
                             rating=float(hA[a].originality.mean())))
    for a in lkeys:
        refs = rng.choice(hkeys, R, replace=False)
        s_, u = score(lA[a], refs, hA)
        peer.append(dict(task=task, agent=a, pop="LLM", norm="human peers (100)", surprisal=s_, unique=u,
                         rating=float(lA[a].originality.mean())))
        refs = rng.choice([k for k in lkeys if k != a], R, replace=False)
        s_, u = score(lA[a], refs, lA)
        peer.append(dict(task=task, agent=a, pop="LLM", norm="LLM-ensemble peers (100)", surprisal=s_, unique=u,
                         rating=float(lA[a].originality.mean())))
        m = lA[a].model.iloc[0]
        same = [k for k in lkeys if k != a and lA[k].model.iloc[0] == m]
        same = list(rng.choice(same, min(54, len(same)), replace=False))
        s_, u = score(lA[a], same, lA)
        peer.append(dict(task=task, agent=a, pop="LLM", norm="same-model peers (<=54)", surprisal=s_, unique=u,
                         rating=float(lA[a].originality.mean())))
P = pd.DataFrame(peer)
P.to_csv(os.path.join(OUT, "E_peer_normed_originality_agents.csv"), index=False)
PS = P.groupby(["task", "pop", "norm"])[["surprisal", "unique", "rating"]].agg(["mean", "sem", "count"]).round(4)
PS.to_csv(os.path.join(OUT, "E_peer_normed_originality_summary.csv"))
# rating comparison at agent (list) level
rt = {}
for task in (2, 3):
    a = P[(P.task == task) & (P.norm == "human peers (100)")]
    hr, lr = a[a["pop"] == "Human"].rating, a[a["pop"] == "LLM"].rating
    w = stats.ttest_ind(lr, hr, equal_var=False)
    rt[task] = dict(human=float(hr.mean()), llm=float(lr.mean()), t=float(w.statistic), p=float(w.pvalue))
log["E_rating_welch"] = rt
print("E done", round(time.time() - t0), "s")

# ------------------------------------------------------------------ F. breadth-vs-depth coverage (descriptive)
cov = []
for task in (2, 3):
    for popname, D in (("Humans", first_k(H[H.task == task], K_PER)),
                       ("LLM ensemble", first_k(L[L.task == task], K_PER))):
        agents = D.agent.unique()
        if len(agents) > 220:
            D = D[D.agent.isin(rng.choice(agents, 220, replace=False))]
        for level in ("category", "idea_str"):
            (x,), K0 = encode(D[level])
            a = pd.factorize(D.agent)[0]
            n = a.max() + 1
            byA = [x[a == i] for i in range(n)]
            for M in [m for m in (10, 20, 50, 100, 150) if m <= n]:
                dA, dB = [], []
                for _ in range(300):
                    ags = rng.choice(n, M, replace=False)
                    dA.append(len({byA[i][rng.integers(len(byA[i]))] for i in ags}))
                    agb = rng.choice(n, M // K_PER, replace=False)
                    dB.append(len(set(np.concatenate([byA[i] for i in agb]).tolist())))
                cov.append(dict(task=task, pop=popname, level=level, M=M,
                                distinct_breadth=float(np.mean(dA)), distinct_depth=float(np.mean(dB)),
                                exchange_ratio=float(np.mean(dA) / np.mean(dB))))
pd.DataFrame(cov).to_csv(os.path.join(OUT, "F_breadth_depth_coverage.csv"), index=False)
print("F done", round(time.time() - t0), "s")

# ------------------------------------------------------------------ descriptive
desc = {}
for task in (1, 2, 3):
    desc[task] = dict(
        human_agents=int(H[H.task == task].agent.nunique()),
        llm_sequences=int(L[L.task == task].agent.nunique()),
        human_mean_len=float(H[H.task == task].groupby("agent").size().mean()),
        llm_mean_len=float(L[L.task == task].groupby("agent").size().mean()),
        human_agents_ge10=int((H[H.task == task].groupby("agent").size() >= 10).sum()),
        llm_seq_ge10=int((L[L.task == task].groupby("agent").size() >= 10).sum()),
        n_categories=int(pd.concat([H[H.task == task].category, L[L.task == task].category]).nunique()),
        human_orig=float(H[H.task == task].originality.mean()) if task > 1 else None,
        llm_orig=float(L[L.task == task].originality.mean()) if task > 1 else None)
log["descriptive"] = desc
json.dump(log, open(os.path.join(OUT, "log.json"), "w"), indent=1, default=float)
print("all done", round(time.time() - t0), "s")
