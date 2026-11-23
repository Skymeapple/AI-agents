"""Build the Best-of-N jailbreaking first-success table from the public raw attempt logs.
Source: data/raw_data/best_of_n_jailbreaking/*_text_t1.0_n10000.jsonl in the public code release of
Schaeffer et al. (2025), github.com/RylanSchaeffer/KoyejoLab-Large-How-Do-Language-Monkey-Power-Get-Their-Power
(commit cd078ae), which redistributes the text-modality BoN runs of Hughes et al. (2024) on 159 HarmBench prompts.
Each line is one augmented attempt {i, flagged, n, behavior_id}.  Attempts were run in batches of 60 and stopped
after the first batch containing a success (cap 10,020 attempts; 6,000 for cygnet), so the first 60 attempts are recorded for EVERY
prompt (checked below) and give genuine fixed-n Binomial counts for n <= 60, while T = index of the first success
gives an unbiased indicator 1{T <= k} of each prompt's pass@k for every k <= 10,020 (a capped first-success design).
Usage: python3 build_bon.py /path/to/best_of_n_jailbreaking   -> data/bon_text_first_success.csv"""
import os, sys, json, glob, collections
import pandas as pd

src = sys.argv[1]
out = []
for f in sorted(glob.glob(os.path.join(src, "*_text_t1.0_n10000.jsonl"))):
    model = os.path.basename(f).split("_text_")[0]
    by = collections.defaultdict(list)
    for line in open(f):
        r = json.loads(line)
        by[r["behavior_id"]].append((int(r["n"]), bool(r["flagged"])))
    for b, v in by.items():
        v.sort()
        ns = [a for a, _ in v]
        assert ns == list(range(len(ns))), (model, b)      # contiguous attempts 0..N-1
        assert len(ns) >= 60 and len(ns) % 60 == 0, (model, b)
        fl = [a for a, s in v if s]
        first = min(fl) + 1 if fl else None
        # stopping rule: the run ends with the batch containing a success (judged at run time); a prompt with no
        # recorded success must have reached the cap, so 1{T <= k} is known for every k <= 10,020.
        if not fl:
            assert len(ns) in (10020, 6000), (model, b)   # cap: 10,020 (6,000 for cygnet)
        stop_mismatch = bool(fl) and (min(fl) // 60 != (len(ns) - 1) // 60)
        first60 = "".join("1" if s else "0" for _, s in v[:60])
        out.append(dict(model=model, behavior=b, n_recorded=len(ns), first_success=first, n_success=len(fl), stop_mismatch=stop_mismatch,
                        first60=first60))
d = pd.DataFrame(out)
dst = os.path.join(os.path.dirname(__file__), "..", "data", "bon_text_first_success.csv")
d.to_csv(dst, index=False)
print(d.groupby("model").agg(M=("behavior", "size"), asr_60=("first60", lambda s: (s.str.count("1") > 0).mean()),
                             asr_cap=("first_success", lambda s: s.notna().mean()),
                             stop_mismatch=("stop_mismatch", "sum")).to_string())
