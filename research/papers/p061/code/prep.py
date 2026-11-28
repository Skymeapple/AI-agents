"""Data preparation: task-level exposure labels -> occupation-level task vectors,
merged with O*NET task weights and BLS OEWS employment (national & by NAICS sector).

Inputs (data/):
  full_labelset.tsv      Eloundou et al. task-level labels (github.com/openai/GPTs-are-GPTs)
  task_ratings.txt       O*NET 30.0 Task Ratings (IM importance, RT relevance)
  national_M20{21..24}_dl.csv  BLS OEWS national estimates (May of each year)
  natsector_M2024_dl.xlsx      BLS OEWS May 2024 national, by NAICS sector
Outputs (results/):
  tasks_merged.parquet-like csv, occ_tasks.pkl, emp_national.csv, emp_sector.csv
"""
import os, pickle
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "data"); R = os.path.join(ROOT, "results")
os.makedirs(R, exist_ok=True)

EMAP = {"alpha": {"E0": 0., "E1": 1., "E2": 0.},
        "beta": {"E0": 0., "E1": 1., "E2": .5},
        "gamma": {"E0": 0., "E1": 1., "E2": 1.}}


def load_tasks():
    L = pd.read_csv(os.path.join(D, "full_labelset.tsv"), sep="\t")
    L = L.rename(columns={"O*NET-SOC Code": "onet", "Task ID": "task_id", "Task Type": "ttype"})
    L["task_id"] = L["task_id"].astype(int)
    L["core"] = (L["ttype"].fillna("Core") == "Core").astype(float)
    for k, m in EMAP.items():
        L[f"gpt_{k}"] = L["gpt4_exposure"].map(m)
        L[f"hum_{k}"] = L["human_exposure_agg"].map(m)
    L["gpt_auto"] = L["automation"].astype(float)          # T0..T4 -> 0..1
    L["autolabel"] = (L["automation"] >= 0.5).astype(float)  # T2+ : task can be (mostly) automated
    # O*NET task ratings
    TR = pd.read_csv(os.path.join(D, "task_ratings.txt"), sep="\t", dtype={"O*NET-SOC Code": str})
    TR = TR.rename(columns={"O*NET-SOC Code": "onet", "Task ID": "task_id"})
    im = TR[TR["Scale ID"] == "IM"][["onet", "task_id", "Data Value"]].rename(columns={"Data Value": "IM"})
    rt = TR[TR["Scale ID"] == "RT"][["onet", "task_id", "Data Value"]].rename(columns={"Data Value": "RT"})
    L = L.merge(im, on=["onet", "task_id"], how="left").merge(rt, on=["onet", "task_id"], how="left")
    match = L["IM"].notna().mean()
    # fallback: match on task_id only (task ids are unique across occupations in O*NET)
    im2 = im.drop_duplicates("task_id").set_index("task_id")["IM"]
    rt2 = rt.drop_duplicates("task_id").set_index("task_id")["RT"]
    L["IM"] = L["IM"].fillna(L["task_id"].map(im2))
    L["RT"] = L["RT"].fillna(L["task_id"].map(rt2))
    match2 = L["IM"].notna().mean()
    # remaining missing: occupation median
    for c in ["IM", "RT"]:
        L[c] = L[c].fillna(L.groupby("onet")[c].transform("median")).fillna(L[c].median())
    L["soc"] = L["onet"].str[:7]
    return L, match, match2


def load_oes(year):
    f = os.path.join(D, f"national_M{year}_dl.csv")
    X = pd.read_csv(f, dtype=str, encoding="utf-8-sig")
    X = X[X["O_GROUP"] == "detailed"].copy()
    X["emp"] = pd.to_numeric(X["TOT_EMP"].str.replace(",", ""), errors="coerce")
    X["wage"] = pd.to_numeric(X["A_MEAN"].str.replace(",", ""), errors="coerce")
    return X[["OCC_CODE", "OCC_TITLE", "emp", "wage"]].rename(columns={"OCC_CODE": "soc"})


def main():
    L, m1, m2 = load_tasks()
    oes = {y: load_oes(y) for y in [2021, 2022, 2023, 2024]}
    codes24 = set(oes[2024]["soc"])
    # map O*NET SOC6 -> OES detailed code (exact, else broad code xx-xxx0)
    def to_oes(s):
        if s in codes24: return s
        b = s[:6] + "0"
        return b if b in codes24 else None
    L["oes"] = L["soc"].map(to_oes)
    info = {"n_tasks": len(L), "n_onet_occ": L["onet"].nunique(),
            "im_match_exact": m1, "im_match_any": m2,
            "share_tasks_mapped_to_oes": L["oes"].notna().mean()}
    L = L[L["oes"].notna()].copy()
    # within each OES code, tasks from several O*NET occupations are pooled; each O*NET
    # occupation receives equal total weight (we normalise weights within O*NET occ first).
    L.to_csv(os.path.join(R, "tasks_merged.csv"), index=False)
    E = pd.DataFrame({"soc": sorted(codes24)})
    for y in oes:
        E = E.merge(oes[y][["soc", "emp"]].rename(columns={"emp": f"emp{y}"}), on="soc", how="left")
    E = E.merge(oes[2024][["soc", "OCC_TITLE", "wage"]], on="soc", how="left")
    E["has_tasks"] = E["soc"].isin(set(L["oes"]))
    tot24 = E["emp2024"].sum()
    info["oes_detailed_occ_2024"] = len(E)
    info["oes_occ_with_tasks"] = int(E["has_tasks"].sum())
    info["emp_coverage_2024"] = E.loc[E.has_tasks, "emp2024"].sum() / tot24
    info["total_emp_2024_detailed"] = tot24
    E.to_csv(os.path.join(R, "emp_national.csv"), index=False)
    S = pd.read_excel(os.path.join(D, "natsector_M2024_dl.xlsx"), dtype=str)
    S = S[S["O_GROUP"] == "detailed"].copy()
    S["emp"] = pd.to_numeric(S["TOT_EMP"].str.replace(",", ""), errors="coerce")
    S = S[["NAICS", "NAICS_TITLE", "OCC_CODE", "emp"]].rename(columns={"OCC_CODE": "soc"})
    S.to_csv(os.path.join(R, "emp_sector.csv"), index=False)
    info["sector_emp_missing_share"] = S["emp"].isna().mean()
    pd.Series(info).to_csv(os.path.join(R, "prep_info.csv"))
    print(pd.Series(info))


if __name__ == "__main__":
    main()
