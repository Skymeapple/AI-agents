"""SynLogic: classical solvers for 13 of the 27 easy task types, checked by the benchmark's verifiers.

The project's proposed route is to generate verified training answers before any
reinforcement learning. This script tests whether that is possible without a model:
each solver reads only the puzzle's inputs from the structured record (never the
stored solution), searches for an answer, formats it as the task asks, and the
benchmark's own verifier decides whether it is correct.

It also probes the arithmetic-target verifier ("mathador") with a deliberately wrong
expression, because its source compares the result with itself.

Usage:
  python synlogic_solvers.py solve <easy-validation.jsonl> <answers.jsonl>      (no dependencies)
  python synlogic_solvers.py verify <easy-validation.jsonl> <answers.jsonl> <SynLogic repo>
"""
from __future__ import annotations

import itertools
import json
import re
import sys
import time
from collections import defaultdict


def block(x):
    return "```python\n" + str(x) + "\n```"


def latin(n, fixed, ok):
    """Fill an n x n grid with 1..n, no repeats in rows or columns; ok(grid, r, c) prunes after each placement."""
    g = [[fixed.get((r, c), 0) for c in range(n)] for r in range(n)]

    def go(k):
        if k == n * n:
            return True
        r, c = divmod(k, n)
        if (r, c) in fixed:
            return ok(g, r, c) and go(k + 1)
        for v in range(1, n + 1):
            if all(g[r][j] != v for j in range(n)) and all(g[i][c] != v for i in range(n)):
                g[r][c] = v
                if ok(g, r, c) and go(k + 1):
                    return True
        g[r][c] = 0
        return False

    for (r, c), v in fixed.items():   # fixed values must not clash with each other
        pass
    return g if go(0) else None


def sudoku(m, q):
    g = [[0 if v == "X" else int(v) for v in row] for row in m["original_sudoku"]]

    def go():
        for r in range(9):
            for c in range(9):
                if g[r][c] == 0:
                    for v in range(1, 10):
                        if all(g[r][j] != v for j in range(9)) and all(g[i][c] != v for i in range(9)) and all(g[3 * (r // 3) + i][3 * (c // 3) + j] != v for i in range(3) for j in range(3)):
                            g[r][c] = v
                            if go():
                                return True
                    g[r][c] = 0
                    return False
        return True

    return block(tuple(tuple(r) for r in g)) if go() else None


def campsite(m, q):
    g, rc, cc = [row[:] for row in m["grid"]], m["row_constraints"], m["col_constraints"]
    R, C = len(g), len(g[0])
    trees = [(r, c) for r in range(R) for c in range(C) if g[r][c] == "T"]
    cells = [(r, c) for r in range(R) for c in range(C) if g[r][c] == "X" and any((r + a, c + b) in trees for a, b in ((0, 1), (1, 0), (0, -1), (-1, 0)))]
    tents = []

    def matched():   # every tree gets its own adjacent tent
        def aug(t, seen, pair):
            for k, (r, c) in enumerate(tents):
                if abs(r - t[0]) + abs(c - t[1]) == 1 and k not in seen:
                    seen.add(k)
                    if k not in pair or aug(pair[k], seen, pair):
                        pair[k] = t
                        return True
            return False
        pair = {}
        return all(aug(t, set(), pair) for t in trees)

    def go(k):
        if len(tents) == len(trees):
            return all(sum(1 for r, _ in tents if r == i) == rc[i] for i in range(R)) and all(sum(1 for _, c in tents if c == j) == cc[j] for j in range(C)) and matched()
        for i in range(k, len(cells)):
            r, c = cells[i]
            if all(max(abs(r - a), abs(c - b)) > 1 for a, b in tents) and sum(1 for a, _ in tents if a == r) < rc[r] and sum(1 for _, b in tents if b == c) < cc[c]:
                tents.append((r, c))
                if go(i + 1):
                    return True
                tents.pop()
        return False

    if not go(0):
        return None
    for r, c in tents:
        g[r][c] = "C"
    return str(g)


def skyscraper(m, q):
    n = m["n"]
    seen = lambda xs: sum(1 for i, x in enumerate(xs) if all(x > y for y in xs[:i]))

    def ok(g, r, c):
        if c == n - 1 and (seen(g[r]) != m["left"][r] or seen(g[r][::-1]) != m["right"][r]):
            return False
        if r == n - 1:
            col = [g[i][c] for i in range(n)]
            if seen(col) != m["top"][c] or seen(col[::-1]) != m["bottom"][c]:
                return False
        return True

    g = latin(n, {}, ok)
    return block(g) if g else None


def star(m, q):
    n, k, reg = m["n"], m["k"], m["region_grid"]
    stars = []

    def go(r):
        if r == n:
            cnt = defaultdict(int)
            for a, b in stars:
                cnt[reg[a][b]] += 1
            return all(cnt[x] == k for row in reg for x in row) and all(sum(1 for _, b in stars if b == c) == k for c in range(n))
        for cols in itertools.combinations(range(n), k):
            if all(b - a > 1 for a, b in zip(cols, cols[1:])) and all(not (a == r - 1 and abs(b - c) <= 1) for c in cols for a, b in stars) \
               and all(sum(1 for _, b in stars if b == c) < k for c in cols):
                cnt = defaultdict(int)
                for a, b in stars + [(r, c) for c in cols]:
                    cnt[reg[a][b]] += 1
                if max(cnt.values()) <= k:
                    stars.extend((r, c) for c in cols)
                    if go(r + 1):
                        return True
                    del stars[-k:]
        return False

    if not go(0):
        return None
    out = defaultdict(list)
    for a, b in sorted(stars):
        out[reg[a][b]].append((a + 1, b + 1))
    return block("{" + ", ".join(f"'{x}': {out[x]}" for x in sorted(out)) + "}")


def survo(m, q):
    g = [row[:] for row in m["original_matrix"]]
    n = len(g)
    holes = [(r, c) for r in range(n - 1) for c in range(n - 1) if g[r][c] == 0]
    left = list(m["candidate_numbers"])

    def go(k):
        if k == len(holes):
            return all(sum(g[r][:-1]) == g[r][-1] for r in range(n - 1)) and all(sum(g[r][c] for r in range(n - 1)) == g[-1][c] for c in range(n - 1))
        r, c = holes[k]
        for v in sorted(set(left)):
            left.remove(v); g[r][c] = v
            if sum(g[r][:-1]) <= g[r][-1] and sum(g[i][c] for i in range(n - 1)) <= g[-1][c] and go(k + 1):
                return True
            left.append(v); g[r][c] = 0
        return False

    return block(g) if go(0) else None


def rows_answer(g):
    return "[[" + ",".join(" ".join(map(str, r)) for r in g) + "]]"


def futoshiki(m, q):
    n = m["grid_size"]
    rows = re.search(r"Current puzzle:\s*\n((?:[X\d ]+\n){%d})" % n, q)
    if not rows:
        rows = re.search(r"((?:(?:[X\d] ){%d}[X\d]\s*\n){%d})" % (n - 1, n), q)
    grid = [ln.split() for ln in rows.group(1).strip().split("\n")]
    fixed = {(r, c): int(v) for r, row in enumerate(grid) for c, v in enumerate(row) if v != "X"}
    cons = [((a[0], a[1]), (b[0], b[1]), s) for a, b, s in m["constraints"]]

    def ok(g, r, c):
        for a, b, s in cons:
            x, y = g[a[0]][a[1]], g[b[0]][b[1]]
            if x and y and max(a, b) <= (r, c) and not (x < y if s == "<" else x > y):
                return False
        return True

    g = latin(n, fixed, ok)
    return rows_answer(g) if g else None


def calcudoko(m, q):
    n = m["grid_size"]
    regs = [([tuple(int(x) - 1 for x in re.findall(r"\d+", cell)) for cell in r["cells"]], r["target"], r["operator"]) for r in m["regions"]]

    def holds(vals, target, op):
        if op == "+":
            return sum(vals) == target
        if op == "*":
            p = 1
            for v in vals:
                p *= v
            return p == target
        big, rest = max(vals), sorted(vals)[:-1]
        if op == "-":
            return big - sum(rest) == target
        p = 1
        for v in rest:
            p *= v
        return big == target * p

    def ok(g, r, c):
        for cells, target, op in regs:
            if max(cells) == (r, c):
                vals = [g[a][b] for a, b in cells]
                if not holds(vals, target, op):
                    return False
        return True

    for strict in (True, False):   # the prompt also forbids repeats inside a region; fall back if that leaves no solution
        g = latin(n, {}, (lambda g, r, c: ok(g, r, c) and all(len({g[a][b] for a, b in cells}) == len(cells) for cells, _, _ in regs if max(cells) == (r, c))) if strict else ok)
        if g:
            return rows_answer(g)
    return None


def numbrix(m, q):
    g = [[0 if v == "X" else int(v) for v in row] for row in m["grid"]]
    n = len(g)
    pos = {g[r][c]: (r, c) for r in range(n) for c in range(n) if g[r][c]}
    total = n * n
    t0 = time.time()

    def go(v, r, c):
        if time.time() - t0 > 60:
            return False
        if v == total:
            return True
        nxt = v + 1
        for a, b in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
            if 0 <= a < n and 0 <= b < n:
                if nxt in pos:
                    if pos[nxt] == (a, b) and go(nxt, a, b):
                        return True
                elif g[a][b] == 0:
                    later = min((k for k in pos if k > nxt), default=None)   # must still be able to reach the next given number
                    if later is not None and abs(pos[later][0] - a) + abs(pos[later][1] - b) > later - nxt:
                        continue
                    g[a][b] = nxt
                    if go(nxt, a, b):
                        return True
                    g[a][b] = 0
        return False

    starts = [pos[1]] if 1 in pos else [(r, c) for r in range(n) for c in range(n) if g[r][c] == 0]
    for r, c in starts:
        if 1 not in pos:
            g[r][c] = 1
        if go(1, r, c):
            return str(g)
        if 1 not in pos:
            g[r][c] = 0
    return None


def cryptarithm(m, q):
    words, ops = m["letter_words"], m["operators"]
    letters = sorted(set("".join(words)))
    lead = {w[0] for w in words if len(w) > 1}
    for digits in itertools.permutations(range(10), len(letters)):
        d = dict(zip(letters, digits))
        if any(d[x] == 0 for x in lead):
            continue
        nums = [int("".join(str(d[ch]) for ch in w)) for w in words]
        left = str(nums[0]) + "".join(f" {o} {v}" for o, v in zip(ops, nums[1:-1]))
        if eval(left) == nums[-1]:
            return f"{left} = {nums[-1]}"
    return None


def math_path(m, q):
    expr, target = m["query_expr"].rsplit("=", 1)
    target = float(target)
    k = expr.count("?")
    for digits in itertools.product("0123456789", repeat=k):
        it = iter(digits)
        e = "".join(next(it) if ch == "?" else ch for ch in expr)
        try:
            if abs(eval(e) - target) < 1e-9:
                return f"[[{e.strip()} = {m['query_expr'].rsplit('=', 1)[1].strip()}]]"
        except (ZeroDivisionError, SyntaxError):
            pass
    return None


PAIRS = {")": "(", "]": "[", "}": "{", ">": "<"}
CLOSE = {v: k for k, v in PAIRS.items()}


def dyck(m, q):
    s, stack = m["question_sequence"], []
    for ch in s:
        if ch in CLOSE:
            stack.append(ch)
        elif stack:
            stack.pop()
    return s + "".join(CLOSE[ch] for ch in reversed(stack))


def dyck_errors(m, q):
    s, stack = m["bracket_string"], []
    for i, ch in enumerate(s, 1):
        if ch in CLOSE:
            stack.append(ch)
        elif not stack or stack.pop() != PAIRS[ch]:
            return str(i)
    return str(len(s) + 1) if stack else "-1"


def mathador(m, q):
    target = m["result"]

    def go(items):   # items: list of (value, expression)
        if len(items) == 1:
            return items[0][1] if abs(items[0][0] - target) < 1e-9 else None
        for i in range(len(items)):
            for j in range(len(items)):
                if i != j:
                    (a, ea), (b, eb) = items[i], items[j]
                    rest = [items[k] for k in range(len(items)) if k not in (i, j)]
                    for op in m["operators"]:
                        if op in "+*" and i > j:
                            continue
                        if op == "/" and abs(b) < 1e-12:
                            continue
                        v = a + b if op == "+" else a - b if op == "-" else a * b if op == "*" else a / b
                        r = go(rest + [(v, f"({ea} {op} {eb})")])
                        if r:
                            return r
        return None

    r = go([(float(x), str(x)) for x in m["numbers"]])
    return block(r) if r else None


SOLVERS = {"sudoku": sudoku, "campsite": campsite, "skyscraper_puzzle": skyscraper, "star_placement_puzzle": star, "survo": survo,
           "futoshiki": futoshiki, "calcudoko": calcudoko, "numbrix": numbrix, "cryptarithm": cryptarithm, "math_path": math_path,
           "dyck_language": dyck, "dyck_language_errors": dyck_errors, "mathador": mathador}


def solve(records, out_path):
    out = open(out_path, "w")
    for line in open(records):
        r = json.loads(line)
        if r["task"] not in SOLVERS:
            continue
        t = time.time()
        try:
            a = SOLVERS[r["task"]](r["game"]["metadata"], r["game"]["question"])
        except Exception as e:
            a = None
            print("solver error", r["task"], r["i"], repr(e)[:120])
        rec = {"i": r["i"], "task": r["task"], "answer": a, "seconds": round(time.time() - t, 2)}
        if r["task"] == "mathador":   # a wrong expression that uses each number once
            rec["wrong"] = block(" + ".join(map(str, r["game"]["metadata"]["numbers"]))) if sum(r["game"]["metadata"]["numbers"]) != r["game"]["metadata"]["result"] else None
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    out.close()
    rows = [json.loads(l) for l in open(out_path)]
    for t in SOLVERS:
        x = [r for r in rows if r["task"] == t]
        print(f"{t:<26} answered {sum(r['answer'] is not None for r in x)} of {len(x)}, slowest {max((r['seconds'] for r in x), default=0)}s")


def verify(records, answers, repo):
    import contextlib, io
    sys.path.insert(0, repo)
    with contextlib.redirect_stdout(io.StringIO()):
        from task2verifier import verifier_classes
        from base.data import Data
    games = {json.loads(l)["i"]: json.loads(l)["game"] for l in open(records)}

    def check(task, i, text):
        if text is None:
            return False
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                return bool(verifier_classes[task]().verify(Data.from_json_dict(dict(games[i])), text))
        except Exception:
            return False

    rows = [json.loads(l) for l in open(answers)]
    total = good = 0
    print(f"{'task':<26}{'puzzles':>8}{'solver answered':>17}{'verifier accepts':>18}{'also inside <answer> tags':>27}")
    for t in SOLVERS:
        x = [r for r in rows if r["task"] == t]
        a = sum(check(t, r["i"], r["answer"]) for r in x)
        b = sum(check(t, r["i"], None if r["answer"] is None else f"<think>search</think> <answer>{r['answer']}</answer>") for r in x)
        total += len(x); good += a
        print(f"{t:<26}{len(x):>8}{sum(r['answer'] is not None for r in x):>17}{a:>18}{b:>27}")
    print(f"\nverified: {good} of {total} puzzles in {len(SOLVERS)} task types; that is {100 * good / 270:.1f}% of the 270-puzzle easy validation set")
    w = [r for r in rows if r.get("wrong")]
    print(f"\nmathador check: a wrong expression (the plain sum of the numbers, which does not equal the target) was accepted by the verifier in {sum(check('mathador', r['i'], r['wrong']) for r in w)} of {len(w)} puzzles")


if __name__ == "__main__":
    if sys.argv[1] == "solve":
        solve(sys.argv[2], sys.argv[3])
    else:
        verify(sys.argv[2], sys.argv[3], sys.argv[4])
