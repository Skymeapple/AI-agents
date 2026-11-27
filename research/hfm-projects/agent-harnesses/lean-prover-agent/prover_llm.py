"""An open-weight prover model on free CPU, on the miniF2F theorems that no tactic proved.

Stand-in for HFM's "Agent for Formal Theorem Proving in Lean" track, which assumes a
paid language-model API. DeepSeek-Prover-V1.5-RL (7B, 4-bit GGUF, llama.cpp on four CPU
cores) writes whole proofs in its non-chain-of-thought mode; each attempt is checked by
Lean against Mathlib, and accepted only if `#print axioms` shows no `sorry`. Only the
test theorems left unproved by all 52 tactics of lean_floor.py are attempted, so the
result adds to the tactic floor.

Usage: python prover_llm.py <miniF2F-lean4 checkout> <model.gguf> <out.json> <shard>/<shards> <attempts> <earlier floor.json ...>
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_floor as lf

PROMPT_HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 400000\n\nopen BigOperators Real Nat Topology Rat\n\n"


def main():
    repo, gguf, outp, shard, attempts = sys.argv[1:6]
    k_shard, n_shards = (int(x) for x in shard.split("/"))
    attempts = int(attempts)
    stmts = lf.statements(os.path.join(repo, "MiniF2F", "Test"))
    for f in sys.argv[6:]:
        for r in json.load(open(f)):
            if r["split"] == "test":
                for name in r["proved"]:
                    stmts.pop(name, None)
    names = sorted(stmts)[k_shard::n_shards]
    print(f"{len(stmts)} test theorems left by the tactics; this shard tries {len(names)}, {attempts} attempts each", flush=True)
    from llama_cpp import Llama
    llm = Llama(model_path=gguf, n_ctx=2048, n_threads=4, verbose=False, seed=k_shard)
    results = []
    for i, name in enumerate(names):
        s = stmts[name]
        prompt = f"Complete the following Lean 4 code:\n\n```lean4\n{PROMPT_HEADER}{s}"
        t0 = time.time()
        proofs = []
        for a in range(attempts):
            out = llm.create_completion(prompt, max_tokens=512, temperature=1.0, top_p=0.95, stop=["```"])
            proofs.append(out["choices"][0]["text"])
        gen = time.time() - t0
        path = os.path.join(repo, f"llm_{k_shard}_{i}.lean")
        with open(path, "w", encoding="utf-8") as fo:
            fo.write(lf.HEADER % 400000)
            for a, p in enumerate(proofs):
                renamed = re.sub(r"theorem\s+" + re.escape(name), f"theorem {name}_a{a}", s, count=1)
                fo.write(f"{renamed}{p.rstrip()}\n\n#print axioms {name}_a{a}\n\n")
        out, status = lf.lean(repo, path, 900)
        os.remove(path)
        ok = set()
        for m in re.finditer(r"'([^']+)' depends on axioms: \[(.*?)\]", out, re.S):
            if "sorryAx" not in m.group(2):
                ok.add(m.group(1))
        for m in re.finditer(r"'([^']+)' does not depend on any axioms", out):
            ok.add(m.group(1))
        good = [a for a in range(attempts) if f"{name}_a{a}" in ok]
        results.append({"name": name, "proved": bool(good), "attempts_ok": good, "lean": status,
                        "proof": proofs[good[0]] if good else None, "gen_secs": round(gen)})
        print(f"  [{i + 1}/{len(names)}] {name}: {'PROVED by ' + str(good) if good else 'no'} (generation {gen:.0f}s, lean {status})", flush=True)
        json.dump(results, open(outp, "w"), indent=0)
    n = sum(r["proved"] for r in results)
    print(f"RESULT shard {k_shard}/{n_shards}: {n} of {len(results)} residual theorems proved with {attempts} attempts each")


if __name__ == "__main__":
    main()
