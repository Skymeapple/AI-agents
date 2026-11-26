"""Model-free floor on miniF2F: how many competition theorems fall to one built-in
Lean/Mathlib tactic, with no language model and no search.

Stand-in for HFM's "Agent for Formal Theorem Proving in Lean" track, which needs a
paid language-model API. Statements: the public miniF2F Lean 4 port
(github.com/yangky11/miniF2F-lean4, MIT; 244 validation and 244 test theorems).
For each tactic, every theorem's `sorry` is replaced by that tactic and the file is
checked once; `#print axioms` after each theorem confirms it was accepted without
`sorry`. Each declaration has a fixed heartbeat budget so slow tactics fail cleanly; a
theorem that still hangs or crashes Lean is dropped for that tactic and the rest of its
batch is rerun.

Usage: python lean_floor.py <miniF2F-lean4 checkout> <out.json> <valid|test> [first:last tactic index]
       python lean_floor.py summary <result.json> [<result.json> ...]
"""
import json
import os
import pty
import re
import select
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

TACTICS = ["rfl", "norm_num", "simp", "ring", "linarith", "nlinarith", "omega", "positivity", "aesop", "simp_all", "grind", "bound",
           "intros; norm_num [*]", "intros; nlinarith [sq_nonneg (0:ℝ)]", "norm_num at *; linarith", "field_simp; ring", "intros; omega",
           "intros; simp_all; linarith", "norm_num [Nat.Prime, Finset.sum_range_succ, Finset.prod_range_succ, Nat.factorial]",
           # second round: short compositions around the strongest tactics of the first round
           "simp_all <;> grind", "norm_num at * <;> grind", "(try field_simp at *) <;> grind", "intros; constructor <;> grind",
           "constructor <;> intros <;> grind", "grind [Nat.ModEq, Int.ModEq]", "decide", "norm_num [Nat.ModEq, Int.ModEq] at * <;> omega",
           "simp_all [Finset.sum_range_succ, Finset.prod_range_succ] <;> grind", "norm_num at * <;> bound",
           "(try field_simp at *) <;> nlinarith", "induction' n with n ih <;> simp_all [Finset.sum_range_succ] <;> grind", "exact?",
           "nlinarith [SQUARES]", "intros; nlinarith [SQUARES]"]
HEARTBEATS = 200000
CHUNK = 41
CHUNK_SECONDS = 600


def tactic_for(k, statement):
    """The tactic text for one theorem. SQUARES expands to the squares of the theorem's
    real variables and of their pairwise sums and differences (at most four variables)."""
    t = TACTICS[k]
    if "SQUARES" not in t:
        return t
    names = []
    for m in re.finditer(r"\(([^():]+):\s*ℝ\s*\)", statement):
        names += m.group(1).split()
    names = list(dict.fromkeys(names))[:4]
    hints = [f"sq_nonneg {v}" for v in names]
    for i, u in enumerate(names):
        for v in names[i + 1:]:
            hints += [f"sq_nonneg ({u} - {v})", f"sq_nonneg ({u} + {v})", f"mul_self_nonneg ({u} * {v})"]
    return t.replace("SQUARES", ", ".join(hints) if hints else "sq_nonneg (0:ℝ)")
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


def lean(repo, path, seconds):
    """Run lean on one file with its output on a pseudo-terminal, so that every message
    is flushed as it is produced and survives a timeout. Returns (output, status)."""
    master, slave = pty.openpty()
    proc = subprocess.Popen(["lake", "env", "lean", "-M", "5000", os.path.basename(path)], cwd=repo, stdout=slave, stderr=slave,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    os.close(slave)
    buf, start, status = [], time.time(), "ok"
    while True:
        left = seconds - (time.time() - start)
        if left <= 0:
            status = "timeout"
            break
        ready, _, _ = select.select([master], [], [], min(left, 5))
        if ready:
            try:
                data = os.read(master, 65536)
            except OSError:
                data = b""
            if not data:
                break
            if sum(map(len, buf)) < 20_000_000:
                buf.append(data)
        elif proc.poll() is not None:
            break
    try:
        os.killpg(proc.pid, signal.SIGKILL)       # also stops the lean process that lake started
    except OSError:
        pass
    code = proc.wait()
    os.close(master)
    if status == "ok" and code not in (0, 1):
        status = f"exit {code}"
    return b"".join(buf).decode("utf-8", "replace").replace("\r\n", "\n"), status


def check(repo, split, k, part, items):
    """Check a batch of theorems with one tactic. If the batch is cut short (timeout or
    crash), keep what was decided, drop the theorem it stopped on, and carry on with
    the rest. Returns the names accepted, notes on what was dropped, and an output sample."""
    proved, notes, head, todo, tries = set(), [], "", list(items), 0
    while todo:
        path = os.path.join(repo, f"floor_{split}_{k}_{part}_{tries}.lean")
        with open(path, "w", encoding="utf-8") as fo:
            fo.write(HEADER)
            for name, s in todo:
                fo.write(f"{s}{tactic_for(k, s)}\n#print axioms {name}\n\n")
        out, status = lean(repo, path, CHUNK_SECONDS)
        os.remove(path)
        head = head or out[:120]
        tries += 1
        seen = set()
        for m in re.finditer(r"'([^']+)' depends on axioms: \[(.*?)\]", out, re.S):
            seen.add(m.group(1))
            if "sorryAx" not in m.group(2):
                proved.add(m.group(1))
        for m in re.finditer(r"'([^']+)' does not depend on any axioms", out):
            seen.add(m.group(1))
            proved.add(m.group(1))
        if status == "ok":
            break
        rest = [(n, s) for n, s in todo if n not in seen]
        if not rest:
            break
        notes.append(f"{rest[0][0]}: {status}")
        todo = rest[1:]
    return proved & {n for n, _ in items}, notes, head


def run(repo, split, stmts, k):
    items = list(stmts.items())
    proved, notes = set(), []
    for part in range(0, len(items), CHUNK):
        got, dropped, head = check(repo, split, k, part, items[part:part + CHUNK])
        proved |= got
        notes += dropped
    print(f"{split:5s} {TACTICS[k]:60s} proves {len(proved):3d} {notes if notes else ''} | {head!r}", flush=True)
    return split, TACTICS[k], sorted(proved)


def watchdog(limit_kb=5_000_000):
    """Kill any lean process whose private (anonymous) memory passes the limit, so one
    runaway tactic cannot take the whole machine down. Mathlib's memory-mapped files
    (about 5.6 GB, shared between processes) are not counted. The chunk is lost."""
    import time
    while True:
        try:
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                try:
                    st = open(f"/proc/{pid}/status").read()
                except OSError:
                    continue
                if not st.startswith("Name:\tlean\n"):
                    continue
                m = re.search(r"RssAnon:\s+(\d+) kB", st)
                if m and int(m.group(1)) > limit_kb:
                    os.kill(int(pid), 9)
                    print(f"watchdog: killed lean process using {int(m.group(1)) // 1000} MB of private memory", flush=True)
        except Exception as e:
            print("watchdog error", e, flush=True)
        time.sleep(1)


def main():
    if sys.argv[1] == "summary":
        return summary(sum((json.load(open(f)) for f in sys.argv[2:]), []))
    repo, outp, which = sys.argv[1:4]
    sets = {which: statements(os.path.join(repo, "MiniF2F", which.capitalize()))}
    print({k: len(v) for k, v in sets.items()}, flush=True)
    import threading
    threading.Thread(target=watchdog, daemon=True).start()
    lo, hi = (int(x) for x in sys.argv[4].split(":")) if len(sys.argv) > 4 else (0, len(TACTICS))
    jobs = [(repo, sp, sets[sp], k) for k in range(lo, hi) for sp in sets]
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
    for _ in range(8):
        t = max(by_v, key=lambda t: len(by_v[t] - cov))
        if not by_v[t] - cov:
            break
        chosen.append(t)
        cov |= by_v[t]
        tc = set().union(*[by_t[c] for c in chosen])
        print(f"RESULT best {len(chosen)} tactics picked on valid ({', '.join(chosen)}): valid {len(cov)}, test {len(tc)} ({len(tc) / n_test:.1%})")


if __name__ == "__main__":
    main()
