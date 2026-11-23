"""Load MMLU-Redux annotation CSVs (github.com/aryopg/mmlu-redux, folder mmlu_redux/) with normalisation."""
import csv, glob, io, os, ast
D = os.path.join(os.path.dirname(__file__), '..', 'data', 'mmlu_redux')
def norm_et(s):
    s = s.strip().lower().replace(' ', '_')
    return {'ok': 'ok'}.get(s, s)
def load():
    rows = []
    for f in sorted(glob.glob(os.path.join(D, '*.csv'))):
        subj = os.path.basename(f)[5:-4]
        raw = open(f, 'rb').read()
        try: txt = raw.decode('utf-8')
        except UnicodeDecodeError: txt = raw.decode('cp1252', errors='replace')
        h = txt.split('\n')[0]
        delim = ';' if h.count(';') > h.count(',') else ','
        for i, r in enumerate(csv.DictReader(io.StringIO(txt), delimiter=delim)):
            try: ch = ast.literal_eval(r['choices'])
            except Exception: ch = None
            rows.append(dict(subject=subj, idx=i, question=r['question'], choices=ch,
                             answer=int(r['answer']), error_type=norm_et(r['error_type']),
                             correct_answer=(r.get('correct_answer') or '').strip()))
    return rows
