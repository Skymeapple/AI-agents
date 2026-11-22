"""Core library: occupation statistics from task vectors, and task->job aggregation rules."""
import os
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")

MEASURES = ["gpt_alpha", "gpt_beta", "gpt_gamma", "hum_alpha", "hum_beta", "hum_gamma", "gpt_auto"]
WEIGHTS = ["eq", "core", "im", "imrt"]


def task_weight(L, scheme):
    if scheme == "eq":
        w = np.ones(len(L))
    elif scheme == "core":
        w = L["core"].values.copy()
        # occupations with no core-labelled task fall back to all tasks
        nocore = L.groupby("onet")["core"].transform("sum").values == 0
        w[nocore] = 1.0
    elif scheme == "im":
        w = L["IM"].values.astype(float)
    elif scheme == "imrt":
        w = (L["IM"] * L["RT"] / 100.0).values.astype(float)
    else:
        raise ValueError(scheme)
    w = pd.Series(w, index=L.index)
    # normalise within O*NET occupation, then give each O*NET occupation equal weight in its OES code
    w = w / w.groupby(L["onet"]).transform("sum")
    n_onet = L.groupby("oes")["onet"].transform("nunique")
    return w / n_onet


def occ_stats(L, measure, scheme):
    """Return DataFrame indexed by OES code with s, a, g, oring, frechet_lo, sd, bottleneck."""
    e = L[measure].astype(float)
    w = task_weight(L, scheme)
    auto = L["autolabel"]
    df = pd.DataFrame({"oes": L["oes"], "onet": L["onet"], "w": w, "e": e, "we": w * e,
                       "wa": w * e * auto, "core": L["core"]})
    g = df.groupby("oes")
    out = pd.DataFrame({"s": g["we"].sum(), "a": g["wa"].sum()})
    out["g"] = out["s"] - out["a"]
    # weighted SD of task exposure within occupation
    df["m"] = df["oes"].map(out["s"])
    df["wd2"] = df["w"] * (df["e"] - df["m"]) ** 2
    out["sd"] = np.sqrt(df.groupby("oes")["wd2"].sum())
    # O-ring statistics on core tasks, per O*NET occupation then averaged within OES code
    c = df[df["core"] == 1]
    po = c.groupby(["oes", "onet"])["e"].agg(
        prod=lambda x: float(np.prod(x)),
        flo=lambda x: float(max(0.0, x.sum() - (len(x) - 1))),
        mn="min",
        bott=lambda x: float((x == 0).mean()))
    po = po.groupby("oes").mean()
    out = out.join(po)
    for col, fill in [("prod", "s"), ("flo", "s"), ("mn", "s")]:
        out[col] = out[col].fillna(out[fill])
    out["bott"] = out["bott"].fillna(1 - out["s"])
    return out


# ---------------------------------------------------------------- aggregation rules
def ces_ratio(a, g, sigma, eps, theta):
    """Employment ratio L'/L in the CES task model (closed form, see paper Sec. 3).

    a: cost share of tasks whose labour is *replaced* by AI (automation)
    g: cost share of tasks where labour stays but becomes 1/(1-theta) times as productive
    sigma: elasticity of substitution between tasks; eps: price elasticity of demand
    theta: proportional cost reduction on exposed tasks.
    """
    a = np.asarray(a, float); g = np.asarray(g, float)
    s = a + g
    if abs(sigma - 1) < 1e-12:
        lnP = s * np.log(1 - theta)
        return (1 - s + g) * np.exp((sigma - eps) * lnP)
    Phi = (1 - s) + s * (1 - theta) ** (1 - sigma)
    lnP = np.log(Phi) / (1 - sigma)
    labour_cost_share = (1 - s) + g * (1 - theta) ** (1 - sigma)
    return labour_cost_share * np.exp((sigma - eps) * lnP)


def rule_table():
    """List of (name, family, function(stats)->displacement share d in (-inf,1])."""
    rules = [("linear", "linear", lambda S: S["s"].values)]
    for tau in [0.3, 0.5, 0.7]:
        rules.append((f"threshold_{tau}", "threshold", lambda S, t=tau: (S["s"].values >= t).astype(float)))
    rules.append(("oring_indep", "oring", lambda S: S["prod"].values))
    rules.append(("oring_frechet_lo", "oring", lambda S: S["flo"].values))
    for mode in ["automate", "augment", "labelled"]:
        for sig in [0.5, 1.0, 2.0]:
            for eps in [0.5, 1.0, 1.5]:
                for th in [0.5, 0.9]:
                    def f(S, mode=mode, sig=sig, eps=eps, th=th):
                        s = S["s"].values
                        if mode == "automate":
                            a, g = s, 0 * s
                        elif mode == "augment":
                            a, g = 0 * s, s
                        else:
                            a, g = S["a"].values, S["g"].values
                        return 1 - ces_ratio(a, g, sig, eps, th)
                    rules.append((f"ces_{mode}_s{sig}_e{eps}_t{th}", f"ces_{mode}", f))
    return rules


def eps_star(s, sigma, theta):
    """Employment-neutral demand elasticity under pure automation (Proposition 2)."""
    s = np.asarray(s, float)
    if abs(sigma - 1) < 1e-12:
        lnP = s * np.log(1 - theta)
    else:
        lnP = np.log((1 - s) + s * (1 - theta) ** (1 - sigma)) / (1 - sigma)
    return sigma + np.log(1 - s) / lnP
