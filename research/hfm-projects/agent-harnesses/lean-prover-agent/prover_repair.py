"""Model sketch, automation for the details: repair the open-weight prover's failed proofs
with the tactic portfolio, on CPU.

Round 1 and 2 of prover_llm.py showed that more whole-proof samples barely help on the
miniF2F test theorems left by the tactics (4 samples prove 18 of 134, 8 more prove 3 of
the 116 left). This tests the other route: keep the model's proof as a sketch and let the
fixed tactic portfolio of lean_floor.py fill the steps the model gets wrong.

For each remaining theorem, the model samples 4 proofs (new seed). Each proof is split into
top-level steps (a step is a line at the proof's base indentation together with its more
indented or `<;>` continuation lines). Variants, each checked by Lean:
  as-is        the sample unchanged (plain sampling, for comparison)
  hardened     every one-line `have ... := by t` becomes `have ... := by first | (t) | PORTFOLIO`
  prefix k     the first k hardened steps, then `all_goals PORTFOLIO` (k = 1 .. n-1)
A theorem counts as proved by repair only if some variant other than as-is is accepted
(no `sorry` in `#print axioms`).

Usage: python prover_repair.py <miniF2F-lean4 checkout> <model.gguf> <out.json> <shard>/<shards> <samples> <earlier results.json ...>
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_floor as lf
from prover_llm import PROMPT_HEADER

MAX_VARIANTS = 12


def steps(proof):
    lines = [ln.rstrip() for ln in proof.split("\n") if ln.strip() and not ln.strip().startswith("--")]
    if not lines:
        return []
    base = min(len(ln) - len(ln.lstrip()) for ln in lines)
    groups = []
    for ln in lines:
        ind = len(ln) - len(ln.lstrip())
        body = ln[base:]
        if ind == base and not body.lstrip().startswith(("<;>", "|")) or not groups:
            groups.append([body])
        else:
            groups[-1].append(body)
    return groups


def harden(groups, port):
    out = []
    for g in groups:
        new = []
        for i, ln in enumerate(g):
            nxt = g[i + 1] if i + 1 < len(g) else None
            ind = len(ln) - len(ln.lstrip())
            continued = nxt is not None and (len(nxt) - len(nxt.lstrip()) > ind or nxt.lstrip().startswith("<;>"))
            m = re.match(r"^(\s*have\b.*?:=\s*by)\s+(\S.*)$", ln)
            if m and not continued:
                ln = f"{m.group(1)} first | ({m.group(2)}) | {port}"
            new.append(ln)
        out.append(new)
    return out


def render(groups, tail=None):
    lines = [ln for g in groups for ln in g]
    if tail:
        lines.append(tail)
    return "\n".join("  " + ln for ln in lines)


def variants(proof, port):
    g = steps(proof)
    if not g:
        return []
    h = harden(g, port)
    vs = [("as-is", render(g)), ("hardened", render(h))]
    for k in range(len(h) - 1, 0, -1):                       # longest prefixes first
        vs.append((f"prefix {k}", render(h[:k], f"all_goals {port}")))
    seen, out = set(), []
    for tag, v in vs:
        if v not in seen:
            seen.add(v)
            out.append((tag, v))
    return out[:MAX_VARIANTS]


def accepted(out):
    ok = set()
    for m in re.finditer(r"'([^']+)' depends on axioms: \[(.*?)\]", out, re.S):
        if "sorryAx" not in m.group(2):
            ok.add(m.group(1))
    for m in re.finditer(r"'([^']+)' does not depend on any axioms", out):
        ok.add(m.group(1))
    return ok


def main():
    repo, gguf, outp, shard, samples = sys.argv[1:6]
    k_shard, n_shards = (int(x) for x in shard.split("/"))
    samples = int(samples)
    stmts = lf.statements(os.path.join(repo, "MiniF2F", "Test"))
    for f in sys.argv[6:]:
        for r in json.load(open(f)):
            if "split" in r and r["split"] == "test":
                for name in r["proved"]:
                    stmts.pop(name, None)
            elif r.get("proved") is True:
                stmts.pop(r["name"], None)
    names = sorted(stmts)[k_shard::n_shards]
    print(f"{len(stmts)} test theorems left by the tactics and the model; this shard tries {len(names)}, {samples} samples each", flush=True)
    from llama_cpp import Llama
    llm = Llama(model_path=gguf, n_ctx=2048, n_threads=4, verbose=False, seed=200 + k_shard + int(os.environ.get("PROVER_SEED", "0")))
    results = []
    for i, name in enumerate(names):
        s = stmts[name]
        port = lf.tactic_for(lf.TACTICS.index("PORTFOLIO"), s)
        prompt = f"Complete the following Lean 4 code:\n\n```lean4\n{PROMPT_HEADER}{s}"
        t0 = time.time()
        wins = []
        for a in range(samples):
            text = llm.create_completion(prompt, max_tokens=512, temperature=1.0, top_p=0.95, stop=["```"])["choices"][0]["text"]
            vs = variants(text, port)
            if not vs:
                continue
            path = os.path.join(repo, f"rep_{k_shard}_{i}_{a}.lean")
            with open(path, "w", encoding="utf-8") as fo:
                fo.write(lf.HEADER % 400000)
                for j, (tag, body) in enumerate(vs):
                    renamed = re.sub(r"theorem\s+" + re.escape(name), f"theorem {name}_s{a}_v{j}", s, count=1)
                    fo.write(f"{renamed.rstrip()}\n{body}\n\n#print axioms {name}_s{a}_v{j}\n\n")
            out, status = lf.lean(repo, path, 900)
            os.remove(path)
            ok = accepted(out)
            for j, (tag, body) in enumerate(vs):
                if f"{name}_s{a}_v{j}" in ok:
                    wins.append({"sample": a, "variant": tag, "proof": body, "lean": status})
        plain = any(w["variant"] == "as-is" for w in wins)
        repaired = any(w["variant"] != "as-is" for w in wins)
        results.append({"name": name, "proved": bool(wins), "plain": plain, "repaired": repaired, "wins": wins,
                        "secs": round(time.time() - t0)})
        tags = sorted({w["variant"] for w in wins})
        print(f"  [{i + 1}/{len(names)}] {name}: {'PROVED ' + str(tags) if wins else 'no'} ({time.time() - t0:.0f}s)", flush=True)
        json.dump(results, open(outp, "w"), indent=0)
    n = sum(r["proved"] for r in results)
    print(f"RESULT shard {k_shard}/{n_shards}: {n} of {len(results)} proved; plain sampling {sum(r['plain'] for r in results)}, "
          f"by repair variants {sum(r['repaired'] for r in results)}, only by repair {sum(r['repaired'] and not r['plain'] for r in results)}")


if __name__ == "__main__":
    main()
