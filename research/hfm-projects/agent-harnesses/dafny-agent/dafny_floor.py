"""Model-free floors on DafnyBench: how many programs verify with no hints, and how
many more a classical guess-and-prune of loop bounds (Houdini) recovers.

Stand-in for HFM's "Agent for Verified Programming with Dafny" track, which needs a
paid language-model API. HFM's task is to implement methods from specifications; the
public DafnyBench task (Loughridge et al., Apache 2.0) is to restore the removed
invariants and assertions so each program verifies. No language model is used here.

Stage 1  verify every hints-removed program unchanged.
Stage 2  for each failing program, add candidate bound invariants read off each
         while-loop guard (for `while i < n`: 0 <= i, i <= n, ...), then repeatedly
         delete every candidate the verifier complains about until the program
         verifies or nothing more can be deleted. Program logic is never edited.

Usage: python dafny_floor.py <dafny-binary> <hints_removed dir> <out.jsonl> [limit]
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

TIME_LIMIT = 30
MARK = "// candidate"


def verify(dafny, text):
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "p.dfy")
        open(path, "w").write(text)
        try:
            r = subprocess.run([dafny, "verify", "--verification-time-limit", str(TIME_LIMIT), "--cores", "1", path],
                               capture_output=True, text=True, timeout=300)
            out = r.stdout + r.stderr
        except subprocess.TimeoutExpired:
            return False, "timeout", set()
    ok = re.search(r"verified, 0 errors", out) is not None and "time out" not in out and "resolution/type errors" not in out
    lines = {int(m.group(1)) for m in re.finditer(r"p\.dfy\((\d+),\d+\)", out)}
    return ok, out, lines


def candidates(guard):
    """Bound invariants suggested by a loop guard such as `i < n` or `j >= 0`."""
    out = []
    for part in re.split(r"&&|\|\|", guard):
        m = re.fullmatch(r"\s*\(?\s*([A-Za-z_]\w*)\s*(<=|<|>=|>|!=)\s*(.+?)\s*\)?\s*", part)
        if not m:
            continue
        v, op, bound = m.groups()
        if op in ("<", "<=", "!="):
            out += [f"0 <= {v}", f"{v} <= {bound}"]
            if op == "<=":
                out.append(f"{v} <= {bound} + 1")
        if op in (">", ">=", "!="):
            out += [f"{v} >= {bound}", f"{v} >= 0"]
            if op == ">=":
                out += [f"{v} >= {bound} - 1", f"{v} >= -1"]
    return list(dict.fromkeys(out))


def add_candidates(text):
    out, n = [], 0
    for line in text.split("\n"):
        m = re.match(r"^(\s*)while\s*(.*?)\s*(\{?)\s*$", line)
        if m and m.group(2) and not m.group(2).startswith("*"):
            indent, guard, brace = m.groups()
            guard = guard.strip()
            if guard.startswith("(") and guard.endswith(")"):
                guard = guard[1:-1]
            cands = candidates(guard)
            if cands:
                out.append(f"{indent}while {m.group(2)}")
                for c in cands:
                    out.append(f"{indent}  invariant {c} {MARK}")
                    n += 1
                if brace:
                    out.append(f"{indent}{{")
                continue
        out.append(line)
    return "\n".join(out), n


def houdini(dafny, text):
    cur, n = add_candidates(text)
    if n == 0:
        return False, 0, 0
    for _ in range(8):
        ok, out, bad = verify(dafny, cur)
        kept = sum(MARK in l for l in cur.split("\n"))
        if ok:
            return True, n, kept
        lines = cur.split("\n")
        drop = [i for i in bad if 0 < i <= len(lines) and MARK in lines[i - 1]]
        if not drop:
            return False, n, kept
        cur = "\n".join(l for i, l in enumerate(lines, 1) if i not in drop)
    return False, n, kept


def one(args):
    dafny, path = args
    text = open(path, encoding="utf-8", errors="replace").read()
    ok, out, _ = verify(dafny, text)
    rec = {"file": os.path.basename(path), "lines": text.count("\n") + 1, "loops": len(re.findall(r"^\s*while\b", text, re.M)),
           "plain": ok, "timeout": out == "timeout" or "time out" in out, "tail": out.strip()[-300:]}
    if not ok and rec["loops"]:
        rec["houdini"], rec["candidates"], rec["kept"] = houdini(dafny, text)
    return rec


def main():
    dafny, folder, outp = sys.argv[1:4]
    files = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".dfy"))
    if len(sys.argv) > 4:
        files = files[:int(sys.argv[4])]
    recs = []
    with ThreadPoolExecutor(4) as ex, open(outp, "w") as fo:
        for i, rec in enumerate(ex.map(one, [(dafny, f) for f in files])):
            recs.append(rec)
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            if i < 3 or i % 50 == 0:
                print(f"[{i}] {rec['file']}: plain {rec['plain']}, houdini {rec.get('houdini')} | {rec['tail'][-120:]!r}", flush=True)
    n = len(recs)
    plain = sum(r["plain"] for r in recs)
    hou = sum(bool(r.get("houdini")) for r in recs)
    noloop = [r for r in recs if r["loops"] == 0]
    loop = [r for r in recs if r["loops"] > 0]
    print(f"\nRESULT programs {n}")
    print(f"RESULT verify with no hints: {plain} ({plain / n:.1%})")
    print(f"RESULT   programs without loops: {sum(r['plain'] for r in noloop)} of {len(noloop)}; with loops: {sum(r['plain'] for r in loop)} of {len(loop)}")
    print(f"RESULT recovered by loop-bound guess-and-prune: {hou} more -> {plain + hou} ({(plain + hou) / n:.1%})")
    print(f"RESULT timeouts among failures: {sum(r['timeout'] and not r['plain'] for r in recs)}")
    short = [r for r in recs if r["lines"] <= 40]
    print(f"RESULT programs of 40 lines or fewer: {len(short)}, verify with no hints {sum(r['plain'] for r in short)}, after guess-and-prune {sum(r['plain'] or bool(r.get('houdini')) for r in short)}")


if __name__ == "__main__":
    main()
