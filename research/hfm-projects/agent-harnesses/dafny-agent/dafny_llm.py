"""An open-weight coding model on free CPU, on the DafnyBench programs the rule floor leaves.

Stand-in for HFM's "Agent for Verified Programming with Dafny", which assumes a paid
frozen model. Qwen2.5-Coder-7B-Instruct (4-bit GGUF, llama.cpp, four CPU cores) is given a
hint-stripped DafnyBench program and asked to add loop invariants, assertions and
decreases clauses so that it verifies. If Dafny rejects the answer, the model gets one
repair turn with the verifier's error messages.

Only programs that neither verify unchanged nor are recovered by the template rules of
dafny_floor.py are attempted, and files that do not parse are skipped.

An answer counts only if it passes an integrity check: every non-blank line of the original
program appears in the answer, in order and unchanged apart from whitespace, and every
added line is an annotation (invariant, assert, decreases, a continuation line of one, or a comment). No
`assume`, `decreases *`, `{:verify false}`, `{:axiom}` or new `requires`/`ensures`, so the specification
cannot be weakened.

Usage: python dafny_llm.py <dafny> <hints_removed folder> <floor.jsonl> <model.gguf> <out.jsonl> <shard>/<shards>
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
ANNOT = re.compile(r"^\s*(invariant\b|assert\b|decreases\b|//|&&|\|\||==>|<==>)")
FORBID = re.compile(r"\bassume\b|\{:verify\s+false\}|\{:axiom\}|\{:extern\}|decreases\s*\*")
MAX_CHARS = 6000


def norm(s):
    return re.sub(r"\s+", " ", s.strip())


def integrity(orig, ans):
    """True if ans is orig with only annotation lines added."""
    if FORBID.search(ans) and not FORBID.search(orig):
        return False, "forbidden construct"
    o = [norm(ln) for ln in orig.split("\n") if norm(ln)]
    a = [ln for ln in ans.split("\n") if norm(ln)]
    i = 0
    for ln in a:
        if i < len(o) and norm(ln) == o[i]:
            i += 1
        elif not ANNOT.match(ln):
            return False, f"changed or added code: {ln.strip()[:80]}"
        elif re.match(r"^\s*(requires|ensures|modifies|reads)\b", ln):
            return False, "new specification clause"
    if i < len(o):
        return False, f"original line missing: {o[i][:80]}"
    return True, ""


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
    todo = []
    for ln in open(floor):
        r = json.loads(ln)
        if r["plain"] or r.get("houdini") or r.get("houdini_post") or "parse errors" in r["tail"]:
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
    llm = Llama(model_path=gguf, n_ctx=8192, n_threads=4, verbose=False, seed=k_shard)
    n_ok = n_first = 0
    with open(outp, "w") as fo:
        for i, f in enumerate(mine):
            orig = open(os.path.join(folder, f), encoding="utf-8", errors="replace").read()
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Add the annotations this Dafny program needs to verify.\n\n```dafny\n{orig}\n```"}]
            t0 = time.time()
            rec = {"file": f, "chars": len(orig), "turns": []}
            for turn in range(2):
                out = llm.create_chat_completion(messages=msgs, temperature=0.2, max_tokens=2500)
                text = out["choices"][0]["message"]["content"]
                ans = extract(text)
                if ans is None:
                    rec["turns"].append({"status": "no code block"})
                    feedback = "Reply with the complete program in one ```dafny code block."
                else:
                    good, why = integrity(orig, ans)
                    if not good:
                        rec["turns"].append({"status": "integrity", "why": why})
                        feedback = f"Your answer changed the program ({why}). Keep every original line unchanged and only add invariant, assert or decreases lines."
                    else:
                        ok, vout, _ = df.verify(dafny, ans)
                        rec["turns"].append({"status": "verified" if ok else "rejected", "errors": "" if ok else errors(vout)[:600]})
                        if ok:
                            rec["proof"] = ans
                            break
                        feedback = f"Dafny rejected it:\n{errors(vout)}\nFix the annotations and reply with the complete program."
                msgs += [{"role": "assistant", "content": text}, {"role": "user", "content": feedback}]
            rec["verified"] = "proof" in rec
            rec["first_turn"] = rec["turns"][0]["status"] == "verified"
            rec["secs"] = round(time.time() - t0)
            n_ok += rec["verified"]
            n_first += rec["first_turn"]
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            print(f"  [{i + 1}/{len(mine)}] {f}: {' / '.join(t['status'] for t in rec['turns'])} ({rec['secs']}s)", flush=True)
    print(f"RESULT shard {k_shard}/{n_shards}: {n_ok} of {len(mine)} verified ({n_first} on the first turn, {n_ok - n_first} after one repair turn)")


if __name__ == "__main__":
    main()
