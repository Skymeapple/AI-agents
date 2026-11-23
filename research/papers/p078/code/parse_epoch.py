"""Recover Epoch AI 'Notable AI Models' (and 'Frontier AI Models') tables from the markdown text
returned by a web extractor (epoch.ai CSV endpoints were not directly reachable from this sandbox).
The extractor replaced line breaks by spaces, so the last field of row k is fused with the first
field of row k+1 ("No GPT-4"). We re-split rows using the known column count and resynchronise on
(name, org, ISO-date) or (name, ISO-date, org) anchors. Validation: compute values in the two
independently-extracted files are compared for overlapping models (results/parse_validation.json)."""
import csv, io, re, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, '..', 'data'); R = os.path.join(HERE, '..', 'results')
DATE = re.compile(r'^\d{4}-\d{2}-\d{2}$'); MERGED = re.compile(r'^(Yes|No)?\s+(\S.*)$')

def resplit(txt, date_pos):
    f = next(csv.reader(io.StringIO(txt.replace('\n', ' '))))
    h = [i for i, x in enumerate(f) if x.startswith('Open model weights?')][0]
    header = f[:h] + ['Open model weights?']; n = len(header)
    first_name = f[h][len('Open model weights?'):].strip()
    def ok(j): return j + date_pos < len(f) and DATE.match(f[j + date_pos]) and (j == h or MERGED.match(f[j]))
    p, rows, resync = h, [], 0
    while p < len(f) - date_pos:
        end = p + n - 1
        row = f[p:end + 1]
        if not ok(p): break
        if end >= len(f) - 1 or ok(end):
            rows.append(row); p = end; continue
        # next row's name was a quoted string containing commas -> rejoin
        resync += 1; rows.append(row)
        j = end + 1
        while j < len(f) - date_pos and not ok(j): j += 1
        name = ','.join(f[end:j + 1])
        f[j] = name; p = j
    out = []
    for r in rows:
        d = dict(zip(header, r))
        m = MERGED.match(r[0]); d['Model'] = (m.group(2) if m else r[0]).strip().strip('"') if r[0] != r[0].lstrip() or m else r[0]
        if r is rows[0]: d['Model'] = first_name
        last = r[-1] if len(r) == n else ''
        mm = re.match(r'^(Yes|No)?\s', last + ' ')
        d['Open model weights?'] = mm.group(1) if mm else ''
        out.append(d)
    return header, out, resync

def fnum(x):
    try: return float(x)
    except: return None

if __name__ == '__main__':
    hn, notable, rs1 = resplit(open(os.path.join(D, 'notable_raw.md')).read(), 2)
    hf, frontier, rs2 = resplit(open(os.path.join(D, 'frontier_raw.md')).read(), 1)
    keep = ['Model', 'Organization', 'Publication date', 'Training compute (FLOP)', 'Model accessibility',
            'Country (of organization)', 'Organization categorization', 'Frontier model', 'Open model weights?',
            'Training compute cost (2023 USD)', 'Domain', 'Parameters', 'Confidence']
    with open(os.path.join(D, 'notable_parsed.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keep, extrasaction='ignore'); w.writeheader(); w.writerows(notable)
    idx = {(r['Model'], r['Publication date']): r for r in notable}
    n = agree = 0; mism = []
    for fr in frontier:
        r = idx.get((fr['Model'], fr['Publication date']))
        c1 = fnum(fr['Training compute (FLOP)'])
        if r is None or c1 is None: continue
        n += 1; c2 = fnum(r['Training compute (FLOP)'])
        if c2 and abs(c2 / c1 - 1) < 1e-9: agree += 1
        else: mism.append([fr['Model'], c1, c2])
    bad_dates = sum(not DATE.match(r['Publication date']) for r in notable)
    stats = dict(notable_rows=len(notable), notable_resyncs=rs1, frontier_rows=len(frontier), frontier_resyncs=rs2,
                 notable_with_compute=sum(fnum(r['Training compute (FLOP)']) is not None for r in notable),
                 notable_bad_dates=bad_dates,
                 accessibility_values=sorted(set(r['Model accessibility'] for r in notable)),
                 frontier_flag_values=sorted(set(r['Frontier model'] for r in notable)),
                 cross_file_matched=n, cross_file_compute_agree=agree, mismatches=mism,
                 frontier_flagged_in_notable=sum(r['Frontier model'] == 'True' for r in notable),
                 latest_date=max(r['Publication date'] for r in notable))
    os.makedirs(R, exist_ok=True)
    json.dump(stats, open(os.path.join(R, 'parse_validation.json'), 'w'), indent=1)
    print(json.dumps(stats, indent=1))
