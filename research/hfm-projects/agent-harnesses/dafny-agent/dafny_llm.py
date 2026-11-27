"""An open-weight coding model on free CPU, on the DafnyBench programs the rule floor leaves:
the model proposes loop invariants, the verifier prunes them.

Stand-in for HFM's "Agent for Verified Programming with Dafny", which assumes a paid
frozen model. Qwen2.5-Coder-7B-Instruct (4-bit GGUF, llama.cpp, four CPU cores) is shown
a hint-stripped DafnyBench program and asked for the annotated program. Only the loop
invariants and decreases clauses it writes after each loop header are kept; they are
inserted after the matching loop header of the ORIGINAL program, together with the
rule floor's own candidates (bounds from the guard, postconditions with the bound
replaced by the counter). Dafny then verifies, and every inserted clause it reports as
failing is dropped and the rest re-verified, up to 12 rounds (Houdini pruning). If the
program still does not verify, the model gets one repair turn: the current program and
Dafny's errors, and its new clauses go through the same insertion and pruning.

The program text is never edited, only clauses are inserted after loop headers, so the
specification cannot be weakened; `decreases *` is not accepted.

Only programs that neither verify unchanged nor are recovered by the template rules of
dafny_floor.py are attempted, and files that do not parse are skipped.

Usage: python dafny_llm.py <dafny> <hints_removed folder> <floor.jsonl> <model.gguf> <out.jsonl> <shard>/<shards> [earlier llm.jsonl ...]
       DAFNY_TURNS (default 2) sets the number of turns (the first answer plus repair turns).
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dafny_floor as df

SYSTEM = ("You are an expert in Dafny. You add loop invariants, assertions and decreases clauses to Dafny programs "
          "so that they verify. You never change the code, the method signatures, or the requires/ensures clauses, "
          "and you never use assume. Reply with the complete program in one ```dafny code block.")
LOOP = re.compile(r"^(\s*)(while|for)\b\s*(.*?)\s*(\{?)\s*$")
CLAUSE = re.compile(r"^\s*(invariant|decreases)\b")
CONT = re.compile(r"^\s*(&&|\|\||==>|<==>|\+|-|\*|<|>|=)")
MAX_CHARS = 6000


def norm(x):
    return re.sub(r"\s+", "", x)


def proposals(answer):
    """Clauses written after each loop header of the model's answer: {normalised header: [clauses]}, plus a list in loop order."""
    lines = answer.split("\n")
    by_head, in_order = {}, []
    for i, ln in enumerate(lines):
        m = LOOP.match(ln)
        if not m:
            continue
        cl = []
        j = i + 1
        while j < len(lines) and (CLAUSE.match(lines[j]) or (cl and CONT.match(lines[j]))):
            if CLAUSE.match(lines[j]):
                cl.append(lines[j].strip())
            else:
                cl[-1] += " " + lines[j].strip()
            j += 1
        cl = [c.split("//")[0].strip().rstrip(";") for c in cl]
        cl = [c for c in cl if c and not re.match(r"decreases\s*\*", c)]
        by_head.setdefault(norm(f"{m.group(2)} {m.group(3)}"), []).extend(cl)
        in_order.append(cl)
    return by_head, in_order


def insert(orig, answer, floor_cands=True):
    """The original program with proposed clauses (and the floor's candidates) after each loop header."""
    by_head, in_order = proposals(answer) if answer else ({}, [])
    floor_by_loop = []
    if floor_cands:
        marked, _ = df.add_candidates(orig, post=True)
        cur = None
        for ln in marked.split("\n"):
            if LOOP.match(ln):
                cur = []
                floor_by_loop.append(cur)
            elif df.MARK in ln and cur is not None:
                cur.append(ln.replace(df.MARK, "").strip())
    out, k, n = [], 0, 0
    for ln in orig.split("\n"):
        m = LOOP.match(ln)
        if not m or not m.group(3) or m.group(3).startswith("*"):
            out.append(ln)
            continue
        indent, kw, head, brace = m.groups()
        cl = list(by_head.get(norm(f"{kw} {head}"), []))
        if not cl and k < len(in_order):
            cl = list(in_order[k])
        if k < len(floor_by_loop):
            cl += floor_by_loop[k]
        k += 1
        cl = list(dict.fromkeys(cl))
        out.append(f"{indent}{kw} {head}")
        for c in cl:
            out.append(f"{indent}  {c} {df.MARK}")
            n += 1
        if brace:
            out.append(f"{indent}{{")
    return "\n".join(out), n


def prune(dafny, text):
    """Houdini: drop inserted clauses that Dafny reports, re-verify, up to 12 rounds."""
    cur = text
    for _ in range(12):
        ok, out, bad = df.verify(dafny, cur)
        if ok:
            return True, cur, out
        lines = cur.split("\n")
        drop = [i for i in bad if 0 < i <= len(lines) and df.MARK in lines[i - 1]]
        if not drop:
            return False, cur, out
        cur = "\n".join(l for i, l in enumerate(lines, 1) if i not in drop)
    return False, cur, out


def extract(text):
    m = re.findall(r"```(?:dafny)?\s*\n(.*?)```", text, re.S)
    if m:
        return max(m, key=len)
    m = re.search(r"```(?:dafny)?\s*\n(.*)", text, re.S)          # unterminated block
    return m.group(1) if m else None


def errors(out):
    keep = [ln for ln in out.split("\n") if "Error" in ln or "error" in ln or "Related location" in ln]
    return "\n".join(keep[:12])[:1500] or out[-800:]


def main():
    dafny, folder, floor, gguf, outp, shard = sys.argv[1:7]
    k_shard, n_shards = (int(x) for x in shard.split("/"))
    turns = int(os.environ.get("DAFNY_TURNS", "2"))
    done = set()
    for prev in sys.argv[7:]:                                    # programs verified by earlier runs
        for ln in open(prev):
            r = json.loads(ln)
            if r.get("verified"):
                done.add(r["file"])
    todo = []
    for ln in open(floor):
        r = json.loads(ln)
        if r["plain"] or r.get("houdini") or r.get("houdini_post") or "parse errors" in r["tail"] or r["file"] in done:
            continue
        todo.append(r["file"])
    todo = sorted(todo)
    sizes = {f: len(open(os.path.join(folder, f), errors="replace").read()) for f in todo}
    long_ = [f for f in todo if sizes[f] > MAX_CHARS]
    todo = [f for f in todo if sizes[f] <= MAX_CHARS]
    mine = todo[k_shard::n_shards]
    print(f"{len(todo) + len(long_)} programs left by the floor (parse errors excluded); {len(long_)} longer than {MAX_CHARS} characters skipped; "
          f"this shard tries {len(mine)}", flush=True)
    from llama_cpp import Llama
    llm = Llama(model_path=gguf, n_ctx=8192, n_threads=4, verbose=False, seed=k_shard + 100 * turns)
    n_ok = n_first = 0
    with open(outp, "w") as fo:
        for i, f in enumerate(mine):
            orig = open(os.path.join(folder, f), encoding="utf-8", errors="replace").read()
            t0 = time.time()
            rec = {"file": f, "chars": len(orig), "turns": []}
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Add the loop invariants and decreases clauses this Dafny program needs to verify. Put them directly after each loop header.\n\n```dafny\n{orig}\n```"}]
            answers = ""
            for turn in range(turns):
                if sum(len(m["content"]) for m in msgs) > 20000:          # keep within the context window
                    msgs = msgs[:2] + msgs[-2:]
                text = llm.create_chat_completion(messages=msgs, temperature=0.2 + 0.2 * turn, max_tokens=2500)["choices"][0]["message"]["content"]
                ans = extract(text) or ""
                answers += "\n" + ans                          # clauses from both turns are pooled
                prog, n = insert(orig, answers)
                ok, cur, vout = prune(dafny, prog) if n else (False, prog, "")
                kept = sum(df.MARK in l for l in cur.split("\n"))
                rec["turns"].append({"proposed": n, "kept": kept, "verified": ok, "errors": "" if ok else errors(vout)[:600]})
                if ok:
                    rec["proof"] = cur
                    break
                msgs += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"With your invariants (those Dafny rejected removed) the program still fails:\n```dafny\n{cur}\n```\n"
                                                     f"Dafny says:\n{errors(vout)}\nGive the complete program again with stronger or additional loop invariants after each loop header."}]
            rec["verified"] = "proof" in rec
            rec["first_turn"] = rec["verified"] and len(rec["turns"]) == 1
            rec["secs"] = round(time.time() - t0)
            n_ok += rec["verified"]
            n_first += rec["first_turn"]
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            print(f"  [{i + 1}/{len(mine)}] {f}: " + " / ".join(f"{t['proposed']} proposed, {t['kept']} kept, {'VERIFIED' if t['verified'] else 'no'}" for t in rec["turns"])
                  + f" ({rec['secs']}s)", flush=True)
    by_turn = [0] * turns
    for ln in open(outp):
        r = json.loads(ln)
        if r["verified"]:
            by_turn[len(r["turns"]) - 1] += 1
    print(f"RESULT shard {k_shard}/{n_shards}: {n_ok} of {len(mine)} verified; by turn: {by_turn}")


if __name__ == "__main__":
    main()
