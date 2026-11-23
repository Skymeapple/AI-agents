"""Join HELM per-instance MMLU predictions with MMLU-Redux annotations.
Outputs:
  data/full_correct.npz : models x all-test-items exact-match matrix (full MMLU, 57 subjects)
  data/redux_resp.npz   : models x 5700 Redux items, chosen option index (in Redux choice order; -1 unmatched/invalid)
"""
import json, os, glob, re, numpy as np, collections
import redux_io
D = os.path.join(os.path.dirname(__file__), '..', 'data'); RAW = os.path.join(D, 'helm_raw')
vers = json.load(open(os.path.join(D, 'run_versions.json')))
def nt(s): return re.sub(r'\s+', ' ', s or '').strip().lower()
R = redux_io.load()
models = sorted({k.split('|')[0] for k in vers})
# need full 57-subject coverage
cover = collections.Counter(k.split('|')[0] for k in vers)
models = [m for m in models if cover[m] == 57]
subjects = sorted({k.split('|')[1] for k in vers})
# instances per (version, subject)
inst_cache = {}
def inst(v, s):
    if (v, s) not in inst_cache:
        inst_cache[(v, s)] = {d['id']: d for d in json.load(open(os.path.join(RAW, f'inst__{v}__{s}.json')))}
    return inst_cache[(v, s)]
# canonical item keys for full MMLU: (subject, normalized question, normalized choices tuple)
item_index = {}; item_list = []
full = {}
redux_by_key = collections.defaultdict(list)
for j, r in enumerate(R):
    redux_by_key[(r['subject'], nt(r['question']))].append(j)
resp = -np.ones((len(models), len(R)), dtype=np.int8)
for mi, m in enumerate(models):
    for s in subjects:
        v = vers[f'{m}|{s}']
        fn = os.path.join(RAW, f'pred__{m}__{s}.json')
        if not os.path.exists(fn): continue
        I = inst(v, s)
        for p in json.load(open(fn)):
            d = I.get(p['instance_id'])
            if d is None: continue
            q = nt(d['input']['text']); refs = [nt(x['output']['text']) for x in d['references']]
            key = (s, q, tuple(refs))
            if key not in item_index:
                item_index[key] = len(item_list); item_list.append(key)
            full[(mi, item_index[key])] = p['stats'].get('exact_match', 0.0)
            for j in redux_by_key.get((s, q), []):
                ch = [nt(str(c)) for c in R[j]['choices']]
                mo = nt(p.get('mapped_output'))
                if mo in ch: resp[mi, j] = ch.index(mo)
                else:
                    # fallback: letter -> HELM ref -> redux choice
                    lt = (p.get('predicted_text') or '').strip()[:1].upper()
                    k = ord(lt) - 65 if lt else -1
                    if 0 <= k < len(refs) and refs[k] in ch: resp[mi, j] = ch.index(refs[k])
M = np.full((len(models), len(item_list)), np.nan, dtype=np.float32)
for (mi, ii), e in full.items(): M[mi, ii] = e
np.savez_compressed(os.path.join(D, 'full_correct.npz'), M=M, models=np.array(models),
                    subjects=np.array([k[0] for k in item_list]))
np.savez_compressed(os.path.join(D, 'redux_resp.npz'), resp=resp, models=np.array(models))
print('models', len(models), 'full items', len(item_list), 'missing frac', np.isnan(M).mean())
matched = (resp >= 0).mean(0)
print('redux items matched by >=90% of models', (matched >= .9).sum(), 'of', len(R))
et = np.array([r['error_type'] for r in R])
for e in sorted(set(et)): print(e, (et == e).sum(), (matched[et == e] >= .9).sum())
