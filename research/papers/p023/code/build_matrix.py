"""Build the item x agent answer matrix from the public MMLU-Pro model outputs.

Source: https://github.com/TIGER-AI-Lab/MMLU-Pro  (eval_results/*.zip), commit f418b116.
Usage:  python build_matrix.py <dir_with_unzipped_model_folders> ../results/answers.npz

Each unzipped folder contains JSON lists of records with fields
question_id, answer, pred, category, src.  We keep the `pred` field exactly as
released by the benchmark authors (their regex extraction); a missing / None pred
is coded -1 (abstention), which the aggregation code counts as wrong.
"""
import glob, json, os, sys
import numpy as np

LET = "ABCDEFGHIJ"
src_dir, out = sys.argv[1], sys.argv[2]
# Excluded: Meta-Llama-3-70B-Instruct (released file has 12,187 records and 9.6% accuracy,
# far below chance-adjusted expectations and its published ~56% score -> misaligned file);
# DeepSeek-Coder-V2 (only 10,351 of 12,032 items present, would shrink the common item set).
EXCLUDE = {"Meta-Llama-3-70B-Instruct_5shots", "DeepSeek-Coder-V2_5shots"}

records = {}          # model -> {qid: pred_idx}
meta = {}             # qid -> (answer_idx, category, n_options)
for d in sorted(glob.glob(os.path.join(src_dir, "*/"))):
    name = os.path.basename(d.rstrip("/"))
    name = name.replace(".json", "")
    if name in EXCLUDE:
        continue
    preds = {}
    for f in glob.glob(d + "**/*.json", recursive=True):
        x = json.load(open(f))
        if not isinstance(x, list):
            continue  # summary files
        for e in x:
            if not isinstance(e, dict):
                continue  # a few malformed entries (bare strings) in one file
            q = str(e["question_id"])
            p = e.get("pred")
            preds[q] = LET.index(p) if isinstance(p, str) and p in LET else -1
            if q not in meta:
                opts = e.get("options")
                if isinstance(opts, str):
                    try:
                        import ast
                        opts = ast.literal_eval(opts)
                    except Exception:
                        opts = None
                nopt = len(opts) if isinstance(opts, list) else 10
                meta[q] = (LET.index(e["answer"]), e["category"], nopt)
    records[name] = preds
    print(f"{name:45s} n={len(preds)}")

# keep items answered (present) by every model
qids = sorted(set.intersection(*[set(v) for v in records.values()]), key=int)
models = sorted(records)
A = np.array([[records[m][q] for m in models] for q in qids], dtype=np.int8)
y = np.array([meta[q][0] for q in qids], dtype=np.int8)
cat = np.array([meta[q][1] for q in qids])
nopt = np.array([meta[q][2] for q in qids], dtype=np.int8)
np.savez_compressed(out, A=A, y=y, cat=cat, nopt=nopt, models=np.array(models), qids=np.array(qids))
print("items", len(qids), "models", len(models))
acc = (A == y[:, None]).mean(0)
for m, a, ab in sorted(zip(models, acc, (A == -1).mean(0)), key=lambda t: -t[1]):
    print(f"{m:45s} acc={a:.4f} abstain={ab:.4f}")
