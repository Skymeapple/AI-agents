"""Download per-instance predictions (display_predictions.json) for every HELM MMLU run,
and instances.json once per (version, subject). Saves compact per-model response tables."""
import json, os, urllib.request, concurrent.futures as cf, time
D = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW = os.path.join(D, 'helm_raw'); os.makedirs(RAW, exist_ok=True)
BASE = "https://storage.googleapis.com/crfm-helm-public/"
runs = json.load(open(os.path.join(D, 'helm_runs.json')))
# keep one run per (model, subject): latest version
vkey = lambda v: tuple(int(x) for x in v[1:].split('.'))
best = {}
for r in runs:
    k = (r['model'], r['subject'])
    if k not in best or vkey(r['version']) > vkey(best[k]['version']): best[k] = r
jobs = []
seen_inst = set()
for r in best.values():
    fn = os.path.join(RAW, f"pred__{r['model']}__{r['subject']}.json")
    jobs.append((r['prefix'] + 'display_predictions.json', fn))
    ik = (r['version'], r['subject'])
    if ik not in seen_inst:
        seen_inst.add(ik)
        jobs.append((r['prefix'] + 'instances.json', os.path.join(RAW, f"inst__{r['version']}__{r['subject']}.json")))
def fetch(job):
    url, fn = job
    if os.path.exists(fn) and os.path.getsize(fn) > 0: return 0
    for a in range(4):
        try:
            data = urllib.request.urlopen(BASE + urllib.parse.quote(url, safe='/:=,@'), timeout=60).read()
            open(fn, 'wb').write(data); return 1
        except Exception as e:
            time.sleep(2 * (a + 1)); err = e
    print('FAIL', url, err); return -1
import urllib.parse
with cf.ThreadPoolExecutor(8) as ex:
    res = list(ex.map(fetch, jobs))
print(len(jobs), 'jobs', sum(1 for x in res if x == 1), 'downloaded', sum(1 for x in res if x == -1), 'failed')
json.dump({f"{k[0]}|{k[1]}": v['version'] for k, v in best.items()}, open(os.path.join(D, 'run_versions.json'), 'w'))
