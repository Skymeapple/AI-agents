"""Strict re-check of every proof this project has counted.

Why: the earlier scripts (lean_floor.py, prover_llm.py, prover_repair.py) accepted a
theorem when `#print axioms` showed no `sorryAx`. That is not enough. When the kernel
rejects a proof ("declaration has metavariables"), Lean adds the theorem as an axiom of
itself, so `#print axioms` lists the theorem's own name and no `sorryAx`; and theorems
whose elaboration failed can print "does not depend on any axioms". recheck.py found both.

Strict criterion here: a candidate is accepted only if
  (1) no Lean error message points to a line inside that candidate's theorem, and
  (2) `#print axioms` for it either says it depends on no axioms or lists only the
      standard ones (propext, Classical.choice, Quot.sound).
Each candidate gets its own uniquely named copy of the statement. If Lean is cut short
(timeout or crash), the candidate it stopped on is counted as failed and the rest are
re-run.

Candidates: every (theorem, tactic) pair any floor.json file claims, and every proof in
the prover_llm.py / prover_repair.py result files.

Usage: python audit.py <miniF2F-lean4 checkout> <out.json> <shard>/<shards> <result files ...>
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_floor as lf

STANDARD = {"propext", "Classical.choice", "Quot.sound"}
CHUNK = 25


def candidates(files, stmts):
    out = []
    index = {t: k for k, t in enumerate(lf.TACTICS)}
    for f in files:
        src = os.path.basename(os.path.dirname(f))
        for r in json.load(open(f)):
            if "tactic" in r:                                          # floor.json
                k = index[r["tactic"]]
                hb = lf.HEARTBEATS * (2 if k >= lf.ROUND3 else 1)
                for name in r["proved"]:
                    s = stmts[r["split"]][name]
                    out.append(dict(split=r["split"], name=name, label=r["tactic"], source=src, hb=hb,
                                    body=f"{s}{lf.tactic_for(k, s)}"))
            elif r.get("proved") and r.get("proof"):                   # prover_llm.py
                s = stmts["test"][r["name"]]
                out.append(dict(split="test", name=r["name"], label="model", source=src, hb=400000, body=f"{s}{r['proof'].rstrip()}"))
            elif r.get("proved") and r.get("wins"):                    # prover_repair.py: every accepted variant
                s = stmts["test"][r["name"]]
                for w in r["wins"]:
                    out.append(dict(split="test", name=r["name"], label=f"model {w['variant']}", source=src, hb=400000,
                                    body=f"{s.rstrip()}\n{w['proof'].rstrip()}"))
    return out


def check(repo, cands, tag):
    """Returns {index: (ok, reason)} for the candidates given as (index, candidate)."""
    res, todo, tries = {}, list(cands), 0
    while todo:
        path = os.path.join(repo, f"audit_{tag}_{tries}.lean")
        hb = max(c["hb"] for _, c in todo)
        lines = (lf.HEADER % hb).split("\n")
        spans = []
        for i, c in todo:
            uid = f"audit_c{i}"
            body = re.sub(r"theorem\s+" + re.escape(c["name"]) + r"\b", f"theorem {uid}", c["body"], count=1)
            start = len(lines) + 1
            lines += body.split("\n")
            end = len(lines)
            lines += ["", f"#print axioms {uid}", ""]
            spans.append((i, uid, start, end))
        open(path, "w", encoding="utf-8").write("\n".join(lines))
        out, status = lf.lean(repo, path, 900)
        os.remove(path)
        tries += 1
        err_lines = [int(m.group(1)) for m in re.finditer(r"^[^\n:]*\.lean:(\d+):\d+: error", out, re.M)]
        seen = set()
        for i, uid, start, end in spans:
            m = re.search(rf"'{uid}' depends on axioms: \[(.*?)\]", out, re.S)
            n = re.search(rf"'{uid}' does not depend on any axioms", out)
            if not (m or n):
                continue
            seen.add(i)
            errs = [e for e in err_lines if start <= e <= end]
            if errs:
                res[i] = (False, f"error at line {errs[0] - start + 1} of the theorem")
            elif m:
                ax = {a.strip() for a in m.group(1).replace("\n", " ").split(",") if a.strip()}
                bad = ax - STANDARD
                res[i] = (not bad, "ok" if not bad else f"non-standard axioms {sorted(bad)}")
            else:
                res[i] = (True, "ok (no axioms)")
        if status == "ok":
            for i, _, _, _ in spans:
                res.setdefault(i, (False, "no axiom report"))
            break
        rest = [(i, c) for i, c in todo if i not in seen]
        if not rest:
            break
        res[rest[0][0]] = (False, f"lean {status}")
        todo = rest[1:]
    return res


def main():
    repo, outp, shard = sys.argv[1:4]
    k_shard, n_shards = (int(x) for x in shard.split("/"))
    stmts = {sp: lf.statements(os.path.join(repo, "MiniF2F", sp.capitalize())) for sp in ("valid", "test")}
    cands = candidates(sys.argv[4:], stmts)
    mine = [(i, c) for i, c in enumerate(cands) if i % n_shards == k_shard]
    print(f"{len(cands)} candidates in all; this shard checks {len(mine)}", flush=True)
    import threading
    threading.Thread(target=lf.watchdog, args=(4_000_000,), daemon=True).start()
    results = []
    for j in range(0, len(mine), CHUNK):
        part = mine[j:j + CHUNK]
        res = check(repo, part, f"{k_shard}_{j}")
        for i, c in part:
            ok, why = res.get(i, (False, "not checked"))
            results.append({k: c[k] for k in ("split", "name", "label", "source")} | {"ok": ok, "why": why})
        json.dump(results, open(outp, "w"), indent=0)
        print(f"  {j + len(part)}/{len(mine)} checked; accepted so far {sum(r['ok'] for r in results)}", flush=True)
    print(f"RESULT shard {k_shard}/{n_shards}: {sum(r['ok'] for r in results)} of {len(results)} candidates pass the strict check")


if __name__ == "__main__":
    main()
