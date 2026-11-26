"""Model-free floor on miniF2F: how many competition theorems fall to one built-in
Lean/Mathlib tactic, with no language model and no search.

Stand-in for HFM's "Agent for Formal Theorem Proving in Lean" track, which needs a
paid language-model API. Statements: the public miniF2F Lean 4 port
(github.com/yangky11/miniF2F-lean4, MIT; 244 validation and 244 test theorems).
For each tactic, every theorem's `sorry` is replaced by that tactic and the file is
checked once; `#print axioms` after each theorem confirms it was accepted without
`sorry`. Each declaration has a fixed heartbeat budget so slow tactics fail cleanly.

Usage: python lean_floor.py <miniF2F-lean4 checkout> <out.json> <valid|test>
       python lean_floor.py summary <valid.json> <test.json>
"""
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

TACTICS = ["rfl", "norm_num", "simp", "ring", "linarith", "nlinarith", "omega", "positivity", "aesop", "simp_all", "grind", "bound",
           "intros; norm_num [*]", "intros; nlinarith [sq_nonneg (0:ℝ)]", "norm_num at *; linarith", "field_simp; ring", "intros; omega",
           "intros; simp_all; linarith", "norm_num [Nat.Prime, Finset.sum_range_succ, Finset.prod_range_succ, Nat.factorial]"]
HEARTBEATS = 200000
CHUNK = 41
HEADER = "import Mathlib\nset_option maxHeartbeats %d\nset_option maxRecDepth 2000\nopen BigOperators Real Nat Topology Rat\n\n" % HEARTBEATS


def statements(folder):
    out = {}
    for f in sorted(os.listdir(folder)):
        text = open(os.path.join(folder, f), encoding="utf-8").read()
        i = text.find("theorem ")
        if i < 0:
            continue
        body = text[i:].rstrip()
        name = re.match(r"theorem\s+(\S+)", body).group(1)
        if body.endswith("sorry"):
            out[name] = body[:-len("sorry")]
    return out


def check(repo, split, k, part, items):
    """Check one chunk of theorems with one tactic; returns the names accepted and a status word."""
    path = os.path.join(repo, f"floor_{split}_{k}_{part}.lean")
    with open(path, "w", encoding="utf-8") as fo:
        fo.write(HEADER)
        for name, s in items:
            fo.write(f"{s}{TACTICS[k]}\n#print axioms {name}\n\n")
    status = "ok"
    try:
        r = subprocess.run(["lake", "env", "lean", "-M", "5000", os.path.basename(path)], cwd=repo, capture_output=True, text=True, timeout=1500)
        out = r.stdout + r.stderr
        if r.returncode not in (0, 1) or r.returncode < 0:
            status = f"exit {r.returncode}"
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        status = "timeout"
    proved = set()
    for m in re.finditer(r"'([^']+)' depends on axioms: \[(.*?)\]", out, re.S):
        if "sorryAx" not in m.group(2):
            proved.add(m.group(1))
    proved |= {m.group(1) for m in re.finditer(r"'([^']+)' does not depend on any axioms", out)}
    return proved & {n for n, _ in items}, status, out[:120]


def run(repo, split, stmts, k):
    items = list(stmts.items())
    proved, notes = set(), []
    for part in range(0, len(items), CHUNK):
        got, status, head = check(repo, split, k, part, items[part:part + CHUNK])
        proved |= got
        if status != "ok":
            notes.append(f"chunk {part}: {status}")
    print(f"{split:5s} {TACTICS[k]:60s} proves {len(proved):3d} {notes if notes else ''} | {head!r}", flush=True)
    return split, TACTICS[k], sorted(proved)


def watchdog(limit_kb=5_500_000):
    """Kill any lean process whose resident memory passes the limit, so one runaway
    tactic cannot take the whole machine down. The chunk it was checking is lost."""
    import time
    while True:
        try:
            for line in subprocess.run(["ps", "-eo", "pid,rss,comm"], capture_output=True, text=True).stdout.split("\n")[1:]:
                f = line.split()
                if len(f) == 3 and f[2] == "lean" and int(f[1]) > limit_kb:
                    os.kill(int(f[0]), 9)
                    print(f"watchdog: killed lean process using {int(f[1]) // 1000} MB", flush=True)
        except Exception as e:
            print("watchdog error", e, flush=True)
        time.sleep(1)


def main():
    if sys.argv[1] == "summary":
        return summary(sum((json.load(open(f)) for f in sys.argv[2:4]), []))
    repo, outp, which = sys.argv[1:4]
    sets = {which: statements(os.path.join(repo, "MiniF2F", which.capitalize()))}
    print({k: len(v) for k, v in sets.items()}, flush=True)
    import threading
    threading.Thread(target=watchdog, daemon=True).start()
    jobs = [(repo, sp, sets[sp], k) for k in range(len(TACTICS)) for sp in sets]
    with ThreadPoolExecutor(2) as ex:
        res = list(ex.map(lambda a: run(*a), jobs))
    json.dump([{"split": s, "tactic": t, "proved": p} for s, t, p in res], open(outp, "w"), indent=0)
    for sp in sets:
        n = len(sets[sp])
        by = {t: set(p) for s, t, p in res if s == sp}
        union = set().union(*by.values())
        print(f"\nRESULT {sp}: {n} theorems; any single tactic proves {len(union)} ({len(union) / n:.1%})")
        got = set()
        for t in sorted(by, key=lambda t: -len(by[t])):
            print(f"RESULT   {t:60s} {len(by[t]):3d}   new beyond the tactics above: {len(by[t] - got)}")
            got |= by[t]
        kinds = {}
        for name in sets[sp]:
            kind = name.split("_")[0]
            a = kinds.setdefault(kind, [0, 0])
            a[0] += name in union
            a[1] += 1
        print("RESULT   by source: " + ", ".join(f"{k} {a}/{b}" for k, (a, b) in sorted(kinds.items())))


def summary(rows):
    n_test = 244
    # tactics chosen on valid, scored on test: greedy cover of the validation set
    by_v = {t: set(p) for r in rows for s, t, p in [(r["split"], r["tactic"], r["proved"])] if s == "valid"}
    by_t = {t: set(p) for r in rows for s, t, p in [(r["split"], r["tactic"], r["proved"])] if s == "test"}
    chosen, cov = [], set()
    for _ in range(5):
        t = max(by_v, key=lambda t: len(by_v[t] - cov))
        if not by_v[t] - cov:
            break
        chosen.append(t)
        cov |= by_v[t]
        tc = set().union(*[by_t[c] for c in chosen])
        print(f"RESULT best {len(chosen)} tactics picked on valid ({', '.join(chosen)}): valid {len(cov)}, test {len(tc)} ({len(tc) / n_test:.1%})")


if __name__ == "__main__":
    main()
