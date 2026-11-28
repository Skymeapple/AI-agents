"""SynLogic easy validation with a small open model on CPU: answer directly, or write a program that searches.

HFM's track fine-tunes a 7B model on SynLogic and scores 0.378 on the easy set. That
needs a GPU. This stand-in asks a different question that a CPU can answer: with no
training, how many easy puzzles does a small model (Qwen2.5-3B, 4-bit) solve

  direct   by reasoning to an answer, with the dataset's own prompt, and
  code     by writing a Python program whose printed output is taken as the answer?

Many of the 27 task types (sudoku, cryptarithms, arithmetic targets, bracket matching)
are search problems, so the split by task shows which part of the benchmark is a
reasoning problem and which is a tool-use problem.

Scoring uses SynLogic's own verifiers (MIT licence, github.com/MiniMax-AI/SynLogic).
For the code route the program's output is offered to the verifier in three wrappings
(as is, in <answer> tags, and in a python block inside <answer> tags), because the
verifiers differ in which wrapping they parse. Correct means any wrapping verifies.

Usage:
  python synlogic_small_model.py run <model.gguf> <validation.parquet> <SynLogic repo> <direct|code> <part> <parts> <out.jsonl>
  python synlogic_small_model.py score <direct.jsonl...> -- <code.jsonl...>
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

CODE_SYSTEM = ("You solve puzzles by writing Python. Reply with one ```python code block and nothing else. "
               "The program must compute the answer itself (search, brute force or direct calculation; standard library only), "
               "finish within 15 seconds, and print only the final answer, in exactly the format the puzzle asks for.")


def extract_code(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return m[-1] if m else text


def run_code(code):
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "s.py")
        open(src, "w").write(code)
        try:
            r = subprocess.run([sys.executable, "-I", src], capture_output=True, text=True, timeout=20, cwd=d, env={"PATH": os.environ.get("PATH", "")})
            return r.stdout.strip()[:4000], ("error" if r.returncode else "ok")
        except subprocess.TimeoutExpired:
            return "", "timeout"


def verify(verifier, data, text):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return bool(verifier.verify(data, text))
    except Exception:
        return False


def run(model_path, parquet, repo, mode, part, parts, out_path):
    import pandas as pd
    sys.path.insert(0, repo)
    with contextlib.redirect_stdout(io.StringIO()):
        from task2verifier import verifier_classes
        from base.data import Data
    from llama_cpp import Llama
    df = pd.read_parquet(parquet)
    llm = Llama(model_path=model_path, n_ctx=6144, n_threads=os.cpu_count(), verbose=False)
    out = open(out_path, "a")
    n = ok = 0
    for i, r in enumerate(df.itertuples()):
        if i % parts != part:
            continue
        task = r.data_source.split("/")[-1]
        prompt = r.prompt[0]["content"]
        extra = r.extra_info if isinstance(r.extra_info, dict) else json.loads(r.extra_info)
        data = Data.from_json_str(extra["game_data_str"]) if isinstance(extra["game_data_str"], str) else Data.from_json_dict(extra["game_data_str"])
        verifier = verifier_classes[task]()
        rec = {"i": i, "task": task}
        t = time.time()
        if len(llm.tokenize(prompt.encode())) > 4800:
            rec.update(correct=False, status="too long")
        elif mode == "direct":
            a = llm.create_chat_completion(messages=[{"role": "user", "content": prompt}], max_tokens=600, temperature=0.0)
            text = a["choices"][0]["message"]["content"]
            rec.update(correct=verify(verifier, data, text), status=a["choices"][0]["finish_reason"], reply=text[-600:])
        else:
            a = llm.create_chat_completion(messages=[{"role": "system", "content": CODE_SYSTEM}, {"role": "user", "content": data.question}], max_tokens=700, temperature=0.0)
            code = extract_code(a["choices"][0]["message"]["content"])
            printed, status = run_code(code)
            good = bool(printed) and any(verify(verifier, data, w) for w in (printed, f"<answer>{printed}</answer>", f"<answer>```python\n{printed}\n```</answer>"))
            rec.update(correct=good, status=status if printed else status + ", nothing printed", reply=printed[-600:], code=code[:3000])
        n += 1; ok += rec["correct"]
        out.write(json.dumps(rec, ensure_ascii=False) + "\n"); out.flush()
        print(f"{n} done, {ok} correct; {task}: {rec['correct']} ({rec['status']}); {time.time() - t:.0f}s", flush=True)


def score(direct_paths, code_paths):
    load = lambda ps: {r["i"]: r for p in ps for r in map(json.loads, open(p))}
    d, c = load(direct_paths), load(code_paths)
    both = sorted(set(d) & set(c))
    tasks = defaultdict(list)
    for i in both:
        tasks[d[i]["task"]].append(i)
    usable = lambda r: r["status"] == "ok"
    row = lambda ids: (sum(d[i]["correct"] for i in ids), sum(c[i]["correct"] for i in ids),
                       sum((c[i] if usable(c[i]) else d[i])["correct"] for i in ids),
                       sum(d[i]["correct"] or c[i]["correct"] for i in ids), sum(usable(c[i]) for i in ids))
    print(f"puzzles answered by both routes: {len(both)} (direct {len(d)}, code {len(c)})")
    print(f"\n{'task':<34}{'n':>4}{'direct':>8}{'code':>6}{'code if it ran, else direct':>29}{'either':>8}{'code ran':>10}")
    for t in sorted(tasks, key=lambda t: -row(tasks[t])[3]):
        a = row(tasks[t])
        print(f"{t:<34}{len(tasks[t]):>4}{a[0]:>8}{a[1]:>6}{a[2]:>29}{a[3]:>8}{a[4]:>10}")
    a = row(both)
    pc = lambda x: f"{100 * x / max(len(both), 1):.1f}%"
    print(f"\n{'all':<34}{len(both):>4}{pc(a[0]):>8}{pc(a[1]):>6}{pc(a[2]):>29}{pc(a[3]):>8}{pc(a[4]):>10}")
    print("direct replies cut off at the length limit:", sum(d[i]["status"] == "length" for i in both))
    print("tasks where only the code route scores:", sorted(t for t in tasks if row(tasks[t])[1] and not row(tasks[t])[0]))
    print("tasks where only the direct route scores:", sorted(t for t in tasks if row(tasks[t])[0] and not row(tasks[t])[1]))
    print("tasks neither route scores on:", sorted(t for t in tasks if not row(tasks[t])[3]))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), int(sys.argv[7]), sys.argv[8])
    else:
        k = sys.argv.index("--")
        score(sys.argv[2:k], sys.argv[k + 1:])
