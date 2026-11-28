"""Harness changes around a frozen small model on checkable shell tasks, on CPU.

Stand-in for HFM's "Agent Harness Evolution Under a Frozen API Model", which needs paid
access to a language model. The question is the same: with the model fixed, how much do
changes to the harness move the held-out pass rate? The frozen model here is
Qwen2.5-3B-Instruct (4-bit GGUF, llama.cpp on CPU), and the tasks are generated shell
tasks, each run in its own empty directory and checked by a Python function.

Tasks: 12 templates x 4 random instances. Templates 0-5 are the development set, on which
a harness would be chosen; templates 6-11 are held out (different kinds of task, not just
new instances).

Harnesses (cumulative):
  h0 minimal     the model sees the task and the output of its previous commands
                 (last 400 characters each); one command per turn; it says DONE to stop
  h1 feedback    also the exit code and stderr labelled separately, and long outputs kept
                 as head and tail (1,200 characters) instead of cut
  h2 discovery   also a listing of the directory and the first lines of each file before
                 the first turn
  h3 verify      also, the first DONE is refused once with a request to check the result,
                 and the model is shown the files it changed

Usage: python shell_harness.py <model.gguf> <harness> <out.jsonl> [max steps]
"""
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time

WORDS = "alpha beta gamma delta omega sigma kappa lambda theta zeta".split()


def T(rng):
    """Task templates: (description, setup(dir), check(dir) -> bool)."""
    tasks = []

    # 0 count matching lines
    n_err = rng.randint(3, 15)
    lines = [f"2024-05-0{rng.randint(1, 9)} {'ERROR' if i < n_err else rng.choice(['INFO', 'WARN', 'DEBUG'])} event {rng.randint(1, 999)}" for i in range(40)]
    rng.shuffle(lines)
    tasks.append(("Count how many lines of app.log contain the word ERROR and write just that number to answer.txt.",
                  lambda d, L=lines: open(f"{d}/app.log", "w").write("\n".join(L) + "\n"),
                  lambda d, n=n_err: _read(d, "answer.txt") == str(n)))
    # 1 sum a csv column
    rows = [(rng.choice(WORDS), rng.randint(1, 100), rng.randint(1, 50)) for _ in range(20)]
    tasks.append(("data.csv has a header line and columns name,price,qty. Write the sum of the qty column to answer.txt (just the number).",
                  lambda d, R=rows: open(f"{d}/data.csv", "w").write("name,price,qty\n" + "".join(f"{a},{b},{c}\n" for a, b, c in R)),
                  lambda d, s=sum(r[2] for r in rows): _read(d, "answer.txt") == str(s)))
    # 2 rename extensions
    names = rng.sample(WORDS, 4)
    tasks.append(("Rename every .txt file in the folder notes/ to have the extension .md instead. Do not change the contents.",
                  lambda d, N=names: (os.makedirs(f"{d}/notes"), [open(f"{d}/notes/{n}.txt", "w").write(n) for n in N]),
                  lambda d, N=names: sorted(os.listdir(f"{d}/notes")) == sorted(f"{n}.md" for n in N) and all(open(f"{d}/notes/{n}.md").read() == n for n in N)))
    # 3 unique sorted
    pool = [rng.choice(WORDS) for _ in range(25)]
    tasks.append(("names.txt has one name per line with repeats. Write the distinct names, sorted alphabetically, one per line, to sorted.txt.",
                  lambda d, P=pool: open(f"{d}/names.txt", "w").write("\n".join(P) + "\n"),
                  lambda d, P=pool: _read(d, "sorted.txt").split("\n") == sorted(set(P))))
    # 4 recursive count
    n_logs = rng.randint(3, 9)
    tasks.append(("Count all files ending in .log anywhere under the folder logs/ (including subfolders) and write the number to answer.txt.",
                  lambda d, n=n_logs: _make_tree(d, n, rng.randint(2, 6)),
                  lambda d, n=n_logs: _read(d, "answer.txt") == str(n)))
    # 5 replace a word
    w_old, w_new = rng.sample(WORDS, 2)
    text = " ".join(rng.choice(WORDS + [w_old] * 3) for _ in range(30))
    tasks.append((f"In story.txt replace every occurrence of the word {w_old} with {w_new}, editing the file in place.",
                  lambda d, t=text: open(f"{d}/story.txt", "w").write(t + "\n"),
                  lambda d, t=text, a=w_old, b=w_new: _read(d, "story.txt") == re.sub(rf"\b{a}\b", b, t)))
    # 6 largest file (held out)
    sizes = rng.sample(range(10, 900), 5)
    fnames = rng.sample(WORDS, 5)
    tasks.append(("Find the largest file (by size in bytes) in the folder files/ and write its file name (without the path) to answer.txt.",
                  lambda d, F=fnames, S=sizes: (os.makedirs(f"{d}/files"), [open(f"{d}/files/{f}.bin", "w").write("x" * s) for f, s in zip(F, S)]),
                  lambda d, F=fnames, S=sizes: _read(d, "answer.txt") == f"{F[S.index(max(S))]}.bin"))
    # 7 unique IPs (held out)
    ips = [f"10.0.{rng.randint(0, 3)}.{rng.randint(1, 20)}" for _ in range(30)]
    tasks.append(("access.log has one request per line, starting with the client IP address. Write the number of distinct IP addresses to answer.txt.",
                  lambda d, I=ips: open(f"{d}/access.log", "w").write("".join(f"{ip} - - GET /{rng.choice(WORDS)} 200\n" for ip in I)),
                  lambda d, I=ips: _read(d, "answer.txt") == str(len(set(I)))))
    # 8 json field (held out)
    recs = [{"id": i, "user": rng.choice(WORDS), "score": rng.randint(0, 100)} for i in range(8)]
    target = rng.choice(recs)
    tasks.append((f"records.json is a JSON list of objects with fields id, user and score. Write the score of the record with id {target['id']} to answer.txt.",
                  lambda d, R=recs: json.dump(R, open(f"{d}/records.json", "w")),
                  lambda d, s=target["score"]: _read(d, "answer.txt") == str(s)))
    # 9 make and run a script (held out)
    k = rng.randint(2, 9)
    tasks.append((f"Create an executable shell script run.sh that prints the numbers 1 to {k}, one per line, then run it and save its output to out.txt.",
                  lambda d: None,
                  lambda d, k=k: os.access(f"{d}/run.sh", os.X_OK) and _read(d, "out.txt").split("\n") == [str(i) for i in range(1, k + 1)]))
    # 10 directory structure (held out)
    proj = rng.choice(WORDS)
    tasks.append((f"Create the folders {proj}/src, {proj}/tests and {proj}/docs, and an empty file {proj}/README.md.",
                  lambda d: None,
                  lambda d, p=proj: all(os.path.isdir(f"{d}/{p}/{s}") for s in ("src", "tests", "docs")) and os.path.isfile(f"{d}/{p}/README.md")))
    # 11 archive (held out)
    m = rng.randint(2, 5)
    tasks.append(("Pack the folder report/ into a gzip-compressed tar archive named report.tar.gz in the current folder.",
                  lambda d, m=m: (os.makedirs(f"{d}/report"), [open(f"{d}/report/part{i}.txt", "w").write(str(i)) for i in range(m)]),
                  lambda d, m=m: _tar_ok(d, m)))
    return tasks


def _read(d, name):
    try:
        return open(os.path.join(d, name)).read().strip()
    except OSError:
        return None


def _make_tree(d, n_logs, n_other):
    paths = [f"logs/{a}/{b}" for a in ("app", "db", "web") for b in ("", "old/")]
    os.makedirs(f"{d}/logs", exist_ok=True)
    for i in range(n_logs + n_other):
        p = f"{d}/{random.choice(paths)}"
        os.makedirs(p, exist_ok=True)
        open(f"{p}/f{i}.{'log' if i < n_logs else 'txt'}", "w").write("x")


def _tar_ok(d, m):
    import tarfile
    try:
        with tarfile.open(f"{d}/report.tar.gz", "r:gz") as t:
            names = {os.path.basename(x) for x in t.getnames()}
        return all(f"part{i}.txt" in names for i in range(m))
    except Exception:
        return False


def run_cmd(cmd, cwd):
    try:
        r = subprocess.run(["bash", "-c", cmd], cwd=cwd, capture_output=True, text=True, timeout=15)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "command timed out after 15 seconds"


def clip(s, n, headtail):
    if len(s) <= n:
        return s
    return s[:n // 2] + "\n...[cut]...\n" + s[-n // 2:] if headtail else s[-n:]


def listing(d):
    out = []
    for root, _, files in os.walk(d):
        for f in sorted(files):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, d)
            try:
                head = open(p, errors="replace").read(200).split("\n")[:3]
            except OSError:
                head = []
            out.append(f"{rel} ({os.path.getsize(p)} bytes): " + " | ".join(head))
    return "\n".join(out) or "(empty folder)"


SYSTEM = ("You are operating a Linux bash shell in a working folder. Reply with exactly one bash command inside a ```bash code block, "
          "or with the single word DONE when the task is complete. Use relative paths.")


def episode(llm, desc, d, harness, max_steps):
    level = int(harness[1])
    msgs = [{"role": "system", "content": SYSTEM}]
    first = f"Task: {desc}"
    if level >= 2:
        first += f"\n\nFiles in the working folder:\n{listing(d)}"
    msgs.append({"role": "user", "content": first})
    refused = False
    for step in range(max_steps):
        out = llm.create_chat_completion(messages=msgs, temperature=0.0, max_tokens=160)
        text = out["choices"][0]["message"]["content"].strip()
        msgs.append({"role": "assistant", "content": text})
        m = re.search(r"```(?:bash|sh)?\s*\n?(.*?)```", text, re.S)
        if not m and "DONE" in text.upper():
            if level >= 3 and not refused:
                refused = True
                msgs.append({"role": "user", "content": "Before finishing, check that the result is exactly what the task asks for. Current files:\n"
                             + listing(d) + "\nIf it is correct, reply DONE; otherwise fix it."})
                continue
            return step + 1
        cmd = m.group(1).strip() if m else text.split("\n")[0]
        code, so, se = run_cmd(cmd, d)
        if level >= 1:
            fb = f"exit code {code}\nstdout:\n{clip(so, 1200, True)}\nstderr:\n{clip(se, 600, True)}"
        else:
            fb = clip(so + se, 400, False) or "(no output)"
        msgs.append({"role": "user", "content": fb})
    return max_steps


def main():
    from llama_cpp import Llama
    gguf, harness, outp = sys.argv[1:4]
    max_steps = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    llm = Llama(model_path=gguf, n_ctx=6144, n_threads=4, verbose=False)
    res = []
    with open(outp, "w") as fo:
        for inst in range(4):
            rng = random.Random(1000 + inst)
            random.seed(2000 + inst)
            for k, (desc, setup, check) in enumerate(T(rng)):
                d = tempfile.mkdtemp()
                try:
                    setup(d)
                    t0 = time.time()
                    steps = episode(llm, desc, d, harness, max_steps)
                    ok = bool(check(d))
                finally:
                    shutil.rmtree(d, ignore_errors=True)
                rec = {"template": k, "instance": inst, "split": "dev" if k < 6 else "test", "pass": ok, "steps": steps, "secs": round(time.time() - t0)}
                res.append(rec)
                fo.write(json.dumps(rec) + "\n")
                fo.flush()
                print(f"  {harness} template {k} instance {inst}: {'pass' if ok else 'fail'} in {steps} steps ({rec['secs']}s)", flush=True)
    for split in ("dev", "test"):
        r = [x for x in res if x["split"] == split]
        print(f"RESULT {harness} {split}: {sum(x['pass'] for x in r)} of {len(r)} ({sum(x['pass'] for x in r) / len(r):.1%})", flush=True)
    print(f"RESULT {harness} by template: " + " ".join(f"{k}:{sum(x['pass'] for x in res if x['template'] == k)}/4" for k in range(12)))


if __name__ == "__main__":
    main()
