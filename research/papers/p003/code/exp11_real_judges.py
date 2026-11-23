"""E11 (revision): a real-verifier anchor.  Response to review W3.

We measure honest discrimination and an upper-tail (inversion) proxy for real reward
models and LLM judges, separately for non-math knowledge domains, reasoning, math and code,
using the public judgment files released with JudgeBench (Tan et al., 2024;
github.com/ScalerLab/JudgeBench, commit e2c52c28).  No model is run here: we analyse the
published scores/verdicts.  The data are not redistributed; clone the repository and pass
its path as argv[1].

Each JudgeBench item is a question with two GPT-4o responses, one correct and one incorrect
under the source dataset's gold answer (MMLU-Pro, LiveBench, LiveCodeBench).

Per (judge, domain) we report, with 95% bootstrap CIs over items:
  pair_acc     P(judge prefers the correct response of a pair)   [within-prompt]
  J_pair       2*pair_acc - 1  (the within-prompt discrimination that drives a
               group-normalised update; 0 = chance)
for reward models (scalar scores) additionally, pooling responses within a domain:
  J_med        TPR - FPR of the binary verifier 'score > domain median'
  inv50, inv90 share of incorrect responses scoring above the 50th / 90th percentile of
               correct responses (upper-tail inversion: wrong answers the verifier rates
               above typical / top-decile correct answers)
  conf_inv     share of pairs in which the wrong response beats the correct one by more
               than one within-domain SD of the score difference (confident inversions)."""
import json, os, sys, glob, re
import numpy as np

R = os.path.join(os.path.dirname(__file__), "..", "results")
ROOT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-0/dl/JudgeBench"


def domain(src):
    if src in ("livebench-math", "mmlu-pro-math"): return "math"
    if src == "livecodebench": return "code"
    if src == "livebench-reasoning": return "reasoning"
    if src.startswith("mmlu-pro-"):
        sub = src[len("mmlu-pro-"):]
        if sub in ("physics", "chemistry", "engineering", "computer science"): return "knowledge-STEM"
        return "knowledge-humanities/social/bio"
    return "other"


def boot(fn, n, nb=2000, seed=0):
    rng = np.random.default_rng(seed)
    v = [fn(rng.integers(0, n, n)) for _ in range(nb)]
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]


def load(path, kind):
    items = []
    for line in open(path):
        d = json.loads(line)
        lab = d["label"]
        if lab not in ("A>B", "B>A"): continue
        corrA = lab == "A>B"
        js = d["judgments"]
        rec = {"source": d["source"], "domain": domain(d["source"])}
        if kind == "reward_model":
            js = [j for j in js if j]
            sc = js[0]["judgment"]["scores"]  # [score(A), score(B)] in the original order
            if len(js) > 1:  # second game has swapped order
                sc2 = js[1]["judgment"]["scores"]; sc = [(sc[0] + sc2[1]) / 2, (sc[1] + sc2[0]) / 2]
            rec["s_c"], rec["s_w"] = (sc[0], sc[1]) if corrA else (sc[1], sc[0])
            rec["pref"] = 1.0 if rec["s_c"] > rec["s_w"] else (0.5 if rec["s_c"] == rec["s_w"] else 0.0)
        else:
            # two games: original order and swapped order; map decisions to 'correct preferred'
            vals = []
            for g, j in enumerate(js):
                dec = j.get("decision") if j else None
                if dec is None: continue
                dec = {"B<A": "A>B", "A<B": "B>A"}.get(dec, dec)
                a_better = dec.startswith("A>")
                b_better = dec.startswith("B>")
                # game 0: A is response_A; game 1: positions swapped
                corr_pos_A = corrA if g == 0 else (not corrA)
                if a_better: vals.append(1.0 if corr_pos_A else 0.0)
                elif b_better: vals.append(0.0 if corr_pos_A else 1.0)
                else: vals.append(0.5)
            if not vals: continue
            rec["pref"] = float(np.mean(vals))
        items.append(rec)
    return items


def rm_metrics(it):
    sc = np.array([r["s_c"] for r in it]); sw = np.array([r["s_w"] for r in it]); n = len(it)
    def jmed(i):
        a = np.concatenate([sc[i], sw[i]]); t = np.median(a)
        return np.mean(sc[i] > t) - np.mean(sw[i] > t)
    def inv(q):
        return lambda i: np.mean(sw[i] > np.percentile(sc[i], q))
    sdd = np.std(sc - sw)
    def conf(i): return np.mean(sw[i] - sc[i] > sdd)
    out = {}
    for name, fn in [("J_med", jmed), ("inv50", inv(50)), ("inv90", inv(90)), ("conf_inv", conf)]:
        out[name] = [float(fn(np.arange(n)))] + boot(fn, n)
    return out


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(ROOT, "outputs", "dataset=judgebench,response_model=gpt-4o-2024-05-13,*.jsonl")))
    res = {"source": "JudgeBench (Tan et al., 2024), judgment files at commit e2c52c28; GPT-4o response pairs",
           "judges": {}}
    doms = ["knowledge-humanities/social/bio", "knowledge-STEM", "reasoning", "math", "code"]
    for fpath in files:
        m = re.search(r"judge_name=([^,]+),judge_model=(.+)\.jsonl", fpath)
        jname = f"{m.group(1)}:{m.group(2)}"
        items = load(fpath, m.group(1))
        rj = {}
        for d in doms + ["non-math knowledge (all)"]:
            it = [r for r in items if (r["domain"] == d) or (d.startswith("non-math") and r["domain"].startswith("knowledge"))]
            if not it: continue
            pref = np.array([r["pref"] for r in it])
            e = {"n": len(it), "pair_acc": [float(pref.mean())] + boot(lambda i: pref[i].mean(), len(it))}
            e["J_pair"] = [2 * x - 1 for x in e["pair_acc"]]
            if "s_c" in it[0]: e.update(rm_metrics(it))
            rj[d] = e
        res["judges"][jname] = rj
        print(jname, {d: (v["n"], round(v["pair_acc"][0], 3), round(v.get("J_med", [np.nan])[0], 3),
                          round(v.get("inv90", [np.nan])[0], 3)) for d, v in rj.items()}, flush=True)
    # pooled over judges: mean J_pair per domain (judge-level spread)
    pooled = {}
    for d in doms + ["non-math knowledge (all)"]:
        v = [rj[d]["J_pair"][0] for rj in res["judges"].values() if d in rj]
        vr = [rj[d]["J_pair"][0] for k, rj in res["judges"].items() if d in rj and k.startswith("reward_model")]
        pooled[d] = {"n_judges": len(v), "median_J_pair_all_judges": float(np.median(v)),
                     "min": float(np.min(v)), "max": float(np.max(v)),
                     "median_J_pair_reward_models": float(np.median(vr)) if vr else None}
    res["pooled"] = pooled
    from scipy.stats import wilcoxon, binomtest
    pairs = {}
    for a in ["knowledge-humanities/social/bio", "non-math knowledge (all)", "reasoning", "code"]:
        ks = [k for k in res["judges"] if a in res["judges"][k] and "math" in res["judges"][k]]
        dm = np.array([res["judges"][k]["math"]["J_pair"][0] - res["judges"][k][a]["J_pair"][0] for k in ks])
        pairs[f"math_minus_{a}"] = {"n_judges": len(ks), "median_diff": float(np.median(dm)),
                                    "n_math_higher": int(np.sum(dm > 0)),
                                    "sign_test_p": float(binomtest(int(np.sum(dm > 0)), int(np.sum(dm != 0))).pvalue),
                                    "wilcoxon_p": float(wilcoxon(dm).pvalue)}
    res["paired_math_vs_other"] = pairs
    print(json.dumps(pairs, indent=1))
    print(json.dumps(pooled, indent=1))
    json.dump(res, open(os.path.join(R, "exp11.json"), "w"), indent=1)
