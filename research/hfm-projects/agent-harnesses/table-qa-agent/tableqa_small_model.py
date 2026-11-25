"""Table question answering with a small open-weight model on CPU: does the harness matter?

Experiment for HFM's "Agent for Questions over Real Tables" track, where the model
is frozen and only the harness around it changes (HFM: 0.37 -> 0.88). HFM's frozen
model needs a paid key. Here a much smaller open model (Qwen2.5-3B-Instruct, 4-bit,
run with llama.cpp on a free 4-core runner) stands in, on WikiTableQuestions
(Pasupat and Liang 2015, public) test questions over tables unseen in training.
Absolute scores are far below a large model's; the comparison between harnesses on
the same model is the point.

Harnesses
  direct   the table (up to 40 rows) is put in the prompt and the model answers
  code-notebook  the same as "code" below, but if the code ends with a bare expression
           its value is printed, as a notebook would (added after the first run showed
           the model almost never calls print)
  code     the model sees the column names and five rows, writes pandas code, the
           code is run on the full table and its printed output is the answer;
           on an error the model is shown the error and tries again (up to 2 retries)

Per-question outputs are written to a JSON-lines file so that combinations (use the
code answer when it ran, else the direct answer) can be scored afterwards.

Usage: python tableqa_small_model.py <model.gguf> <WikiTableQuestions dir> <direct|code|code-notebook> <questions> <out.jsonl> [questions to skip]
       python tableqa_small_model.py score <direct.jsonl> <code.jsonl> [label]
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata


def norm(x):
    x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode().lower().strip()
    x = re.sub(r"^(the|a|an)\s+", "", x)
    x = re.sub(r"[\"'$%,]", "", x).strip(" .")
    try:
        return f"{float(x):.4g}"
    except ValueError:
        return re.sub(r"\s+", " ", x)


def correct(answer, target):
    gold = {norm(t) for t in target.split("|")}
    got = {norm(t) for t in re.split(r"\||\n|;", str(answer)) if t.strip()}
    if got == gold:
        return True
    return len(gold) == 1 and norm(str(answer).strip().split("\n")[-1]) in gold


def read_table(path):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def show(rows, n, width=40):
    return "\n".join(" | ".join(c.replace("\n", " ")[:width] for c in r) for r in rows[: n + 1])


def print_last(code):
    """As a notebook does: if the code ends with a bare expression, print its value."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        call = ast.Expr(ast.Call(ast.Name("print", ast.Load()), [last.value], []))
        if not (isinstance(last.value, ast.Call) and getattr(last.value.func, "id", "") == "print"):
            tree.body[-1] = ast.copy_location(call, last)
            return ast.unparse(ast.fix_missing_locations(tree))
    return code


def run_code(code, table_path, notebook=False):
    if notebook:
        code = print_last(code)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "run.py")
        with open(src, "w") as f:
            f.write("import pandas as pd, numpy as np, re\ndf = pd.read_csv(%r, dtype=str, keep_default_na=False)\n" % table_path + code)
        try:
            r = subprocess.run([sys.executable, "-I", src], capture_output=True, text=True, timeout=20, cwd=d, env={"PATH": os.environ.get("PATH", "")})
        except subprocess.TimeoutExpired:
            return None, "timed out"
        if r.returncode != 0 or not r.stdout.strip():
            return None, (r.stderr.strip().split("\n")[-1] if r.stderr.strip() else "the code printed nothing")[:300]
        return r.stdout.strip()[:300], None


def extract_code(text):
    m = re.search(r"```(?:python)?\n(.*?)```", text, re.S)
    return (m.group(1) if m else text).replace("pd.read_csv(", "(lambda *a, **k: df)(")


def clean(a):
    """A code answer that looks like an answer: short, one line, not empty, zero, a truth value or a printed table."""
    a = str(a)
    return not ("\n" in a or "dtype" in a or len(a) > 60 or a in ("", "nan", "None", "[]", "0", "False", "True") or a.startswith("Empty"))


def score(direct_path, code_path, label="code"):
    print(f"--- {label}")
    load = lambda p: {r["id"]: r for r in map(json.loads, open(p))}
    d, c = load(direct_path), load(code_path)
    ids = sorted(set(d) & set(c))
    acc = lambda f: 100 * sum(f(i) for i in ids) / len(ids)
    print(f"{len(ids)} questions answered by both harnesses")
    print(f"  direct answer from the table in the prompt        {acc(lambda i: d[i]['ok']):.1f}%")
    print(f"  code, first attempt                               {acc(lambda i: c[i]['ok_first']):.1f}%   (code ran: {acc(lambda i: c[i]['ran_first']):.1f}%)")
    print(f"  code, with up to two retries on errors            {acc(lambda i: c[i]['ok']):.1f}%   (code ran: {acc(lambda i: c[i]['ran']):.1f}%)")
    print(f"  code when it ran, otherwise the direct answer     {acc(lambda i: c[i]['ok'] if c[i]['ran'] else d[i]['ok']):.1f}%")
    print(f"  code when its answer looks clean, else direct     {acc(lambda i: c[i]['ok'] if (c[i]['ran'] and clean(c[i]['answer'])) else d[i]['ok']):.1f}%")
    print(f"  either harness right (upper bound of a selector)  {acc(lambda i: c[i]['ok'] or d[i]['ok']):.1f}%")
    both = sum(c[i]["ok"] and d[i]["ok"] for i in ids); only_c = sum(c[i]["ok"] and not d[i]["ok"] for i in ids); only_d = sum(d[i]["ok"] and not c[i]["ok"] for i in ids)
    print(f"  right in both {both}, code only {only_c}, direct only {only_d}")
    small = [i for i in ids if d[i]["rows"] <= 40]; big = [i for i in ids if d[i]["rows"] > 40]
    for name, g in (("tables that fit in the prompt (40 rows or fewer)", small), ("longer tables", big)):
        if g:
            print(f"  {name}: {len(g)} questions, direct {100 * sum(d[i]['ok'] for i in g) / len(g):.1f}%, code {100 * sum(c[i]['ok'] for i in g) / len(g):.1f}%")
    print(f"  seconds per question: direct {sum(d[i]['secs'] for i in ids) / len(ids):.0f}, code {sum(c[i]['secs'] for i in ids) / len(ids):.0f}")


def main():
    if sys.argv[1] == "score":
        return score(*sys.argv[2:5])
    model_path, root, mode, n, out = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    from llama_cpp import Llama
    llm = Llama(model_path=model_path, n_ctx=6144, n_threads=os.cpu_count(), verbose=False)

    def ask(messages, max_tokens):
        r = llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.0)
        return r["choices"][0]["message"]["content"].strip()

    with open(os.path.join(root, "data", "pristine-unseen-tables.tsv"), encoding="utf-8") as f:
        qs = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    import random
    random.Random(0).shuffle(qs)
    skip = int(sys.argv[6]) if len(sys.argv) > 6 else 0      # a fresh set of questions, to check a rule chosen on the first set
    seen = 0
    done, n_ok = 0, 0
    with open(out, "w") as fo:
        for q in qs:
            if done >= n:
                break
            path = os.path.join(root, q["context"])
            rows = read_table(path)
            if len(rows) < 2 or len(show(rows, 40)) > 9000:
                continue
            seen += 1
            if seen <= skip:
                continue
            t0 = time.time()
            rec = dict(id=q["id"], question=q["utterance"], target=q["targetValue"], rows=len(rows) - 1)
            if mode == "direct":
                a = ask([{"role": "system", "content": "Answer the question from the table. Reply with only the answer, no explanation. If there are several answers separate them with |."},
                         {"role": "user", "content": f"Table:\n{show(rows, 40)}\n\nQuestion: {q['utterance']}\nAnswer:"}], 40)
                rec.update(answer=a, ok=correct(a, q["targetValue"]))
            else:
                msgs = [{"role": "system", "content": "You write short Python (pandas) code to answer a question about a table. The table is already loaded as the DataFrame `df`; every column is text, so convert to numbers where needed. Print only the final answer. Reply with one code block."},
                        {"role": "user", "content": f"Columns: {rows[0]}\nFirst rows:\n{show(rows, 5)}\nNumber of rows: {len(rows) - 1}\n\nQuestion: {q['utterance']}"}]
                for attempt in range(3):
                    reply = ask(msgs, 220)
                    ans, err = run_code(extract_code(reply), os.path.abspath(path), notebook=mode == "code-notebook")
                    if attempt == 0:
                        rec.update(ran_first=err is None, ok_first=err is None and correct(ans, q["targetValue"]))
                    if err is None:
                        break
                    msgs += [{"role": "assistant", "content": reply}, {"role": "user", "content": f"That failed: {err}\nFix the code. Reply with one code block."}]
                rec.update(answer=ans, ran=err is None, attempts=attempt + 1, ok=err is None and correct(ans, q["targetValue"]), code=reply[-600:])
            rec["secs"] = time.time() - t0
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            done += 1; n_ok += rec["ok"]
            if done % 10 == 0:
                print(f"{mode}: {done} questions, {100 * n_ok / done:.1f}% correct, {rec['secs']:.0f}s last", flush=True)
    print(f"{mode}: {done} questions, {100 * n_ok / done:.1f}% correct")


if __name__ == "__main__":
    main()
