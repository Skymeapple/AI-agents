import numpy as np, os, redux_io
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RES = os.path.join(ROOT, 'results'); os.makedirs(RES, exist_ok=True)
def ci(s):
    s = s.strip()
    if s and s in 'ABCD': return 'ABCD'.index(s)
    try: return int(s)
    except ValueError: return -1
def load_all():
    R = redux_io.load()
    z = np.load(os.path.join(ROOT, 'data', 'redux_resp.npz'))
    resp, models = z['resp'], [str(m) for m in z['models']]
    et = np.array([r['error_type'] for r in R]); gold = np.array([r['answer'] for r in R])
    corr = np.array([ci(r['correct_answer']) for r in R])
    return R, resp, models, et, gold, corr
def ability(resp, et, gold):
    ok = et == 'ok'
    v = resp[:, ok] >= 0
    acc = ((resp[:, ok] == gold[ok]) & v).sum(1) / v.sum(1)
    th = np.log(acc / (1 - acc))
    return acc, (th - th.mean()) / th.std()
sig = lambda x: 1 / (1 + np.exp(-x))
