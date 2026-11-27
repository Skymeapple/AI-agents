"""Repair with Lean's own error messages: does verifier feedback help the open prover?

In the Dafny project, one repair turn with the verifier's errors verified four times as
many programs as the first answer. The automatic tactic repair tried in prover_repair.py
added nothing once proofs were checked strictly. This tests real feedback for Lean.

For every miniF2F test theorem not yet proved (by the audited results of audit.py), the
model (DeepSeek-Prover-V1.5-RL, 4-bit, CPU) samples 4 proofs. Each is checked with the
strict test of lean_floor.strict. For each failed sample (up to 4), the prompt is extended
with that attempt and Lean's first error messages as a comment, and the model writes one
new proof: 4 feedback samples, so 8 samples per theorem in all. Results are reported
separately for the fresh and the feedback samples.

Usage: python prover_feedback.py <miniF2F-lean4 checkout> <model.gguf> <out.json> <shard>/<shards> <audit.json ...>
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_floor as lf
from prover_llm import PROMPT_HEADER


def check(repo, path, name, s, proofs):
    items = [(f"{name}_x{a}", re.sub(r"theorem\s+" + re.escape(name), f"theorem {name}_x{a}", s, count=1) + p.rstrip()) for a, p in enumerate(proofs)]
    text, spans = lf.build(lf.HEADER % 400000, items)
    open(path, "w", encoding="utf-8").write(text)
    out, status = lf.lean(repo, path, 900)
    os.remove(path)
    ok, _ = lf.strict(out, spans)
    msgs = {}
    for (uid, start, end) in spans:
        errs = [m for m in re.finditer(r"^[^\n:]*\.lean:(\d+):(\d+): error: (.*?)(?=^[^\n:]*\.lean:\d+:\d+:|^'|\Z)", out, re.M | re.S)
                if start <= int(m.group(1)) <= end]
        msgs[uid] = "\n".join(f"line {int(m.group(1)) - start + 1}: {m.group(3).strip()[:300]}" for m in errs[:3]) or "the proof was not accepted"
    return [f"{name}_x{a}" in ok for a in range(len(proofs))], [msgs[f"{name}_x{a}"] for a in range(len(proofs))]


def main():
    repo, gguf, outp, shard = sys.argv[1:5]
    k_shard, n_shards = (int(x) for x in shard.split("/"))
    stmts = lf.statements(os.path.join(repo, "MiniF2F", "Test"))
    proved = {r["name"] for f in sys.argv[5:] for r in json.load(open(f)) if r["ok"] and r["split"] == "test"}
    left = sorted(n for n in stmts if n not in proved)
    names = left[k_shard::n_shards]
    print(f"{len(left)} test theorems not proved by the audited results; this shard tries {len(names)}", flush=True)
    import threading
    threading.Thread(target=lf.watchdog, args=(4_000_000,), daemon=True).start()
    from llama_cpp import Llama
    llm = Llama(model_path=gguf, n_ctx=3072, n_threads=4, verbose=False, seed=300 + k_shard)
    results = []
    for i, name in enumerate(names):
        s = stmts[name]
        base = f"Complete the following Lean 4 code:\n\n```lean4\n{PROMPT_HEADER}{s}"
        t0 = time.time()
        fresh = [llm.create_completion(base, max_tokens=512, temperature=1.0, top_p=0.95, stop=["```"])["choices"][0]["text"] for _ in range(4)]
        ok1, errs = check(repo, os.path.join(repo, f"fb_{k_shard}_{i}_a.lean"), name, s, fresh)
        fb = []
        for p, e, good in zip(fresh, errs, ok1):
            if good:
                continue
            prompt = (f"Complete the following Lean 4 code:\n\n```lean4\n{PROMPT_HEADER}{s}{p.rstrip()}\n```\n\n"
                      f"Lean rejected this proof:\n{e}\n\nWrite a corrected proof.\n\n```lean4\n{PROMPT_HEADER}{s}")
            fb.append(llm.create_completion(prompt, max_tokens=512, temperature=0.7, top_p=0.95, stop=["```"])["choices"][0]["text"])
        ok2 = check(repo, os.path.join(repo, f"fb_{k_shard}_{i}_b.lean"), name, s, fb)[0] if fb else []
        rec = {"name": name, "fresh_ok": sum(ok1), "feedback_ok": sum(ok2), "n_feedback": len(fb),
               "proved": bool(any(ok1) or any(ok2)), "by_feedback_only": bool(any(ok2) and not any(ok1)),
               "proof": next((p for p, g in zip(fresh, ok1) if g), None) or next((p for p, g in zip(fb, ok2) if g), None),
               "secs": round(time.time() - t0)}
        results.append(rec)
        json.dump(results, open(outp, "w"), indent=0)
        print(f"  [{i + 1}/{len(names)}] {name}: fresh {sum(ok1)}/4, feedback {sum(ok2)}/{len(fb)} ({rec['secs']}s)", flush=True)
    print(f"RESULT shard {k_shard}/{n_shards}: {sum(r['proved'] for r in results)} of {len(results)} proved; "
          f"by fresh samples {sum(r['fresh_ok'] > 0 for r in results)}, only by feedback {sum(r['by_feedback_only'] for r in results)}")


if __name__ == "__main__":
    main()
