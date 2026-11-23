"""Reconstruct Epoch AI's 'Large-Scale AI Models' CSV from a newline-flattened text dump.

Source: https://epoch.ai/data/large_scale_ai_models.csv (Epoch AI, CC-BY), retrieved 2026-11-22
via the Tavily extract API (format=markdown, depth=advanced), because direct download was blocked
by this sandbox's egress proxy. The dump is a faithful copy of the CSV except that record
separators (newlines) were replaced by spaces and some markdown escapes (e.g. '\\_') were added.
We re-split records using the known column count (34): after the 33rd unquoted comma of a record,
the last field ('Training compute cost (upfront)') is empty or one numeric token, then a space.
"""
import csv, re, sys
src = sys.argv[1] if len(sys.argv) > 1 else 'data/large_scale_md.txt'
out = sys.argv[2] if len(sys.argv) > 2 else 'data/epoch_large_scale.csv'
s = open(src, encoding='utf-8').read()
key = 'Training compute cost (upfront)'
hdr_end = s.index(key) + len(key)
header = next(csv.reader([s[:hdr_end]]))
ncol = len(header); assert ncol == 34, ncol
body = s[hdr_end:].lstrip(' ')
num = re.compile(r'-?[0-9][0-9.eE+\-]*')
rows, i, n = [], 0, len(body)
while i < n:
    commas, inq, j = 0, False, i
    while j < n and commas < ncol - 1:
        c = body[j]
        if c == '"': inq = not inq
        elif c == ',' and not inq: commas += 1
        j += 1
    if commas < ncol - 1: break
    m = num.match(body, j)
    k = m.end() if m else j
    rows.append(next(csv.reader([body[i:k]])))
    i = k + 1
good = [r for r in rows if len(r) == ncol]
print(f'records={len(rows)} wellformed={len(good)}')
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(header); w.writerows(good)
