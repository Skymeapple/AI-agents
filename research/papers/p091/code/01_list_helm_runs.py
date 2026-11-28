"""List all HELM MMLU run directories across release versions (public GCS bucket crfm-helm-public)."""
import json, urllib.request, urllib.parse, re, os
OUT = os.path.join(os.path.dirname(__file__), '..', 'data', 'helm_runs.json')
API = "https://storage.googleapis.com/storage/v1/b/crfm-helm-public/o"
def list_prefixes(prefix):
    out, tok = [], None
    while True:
        q = {"prefix": prefix, "delimiter": "/", "maxResults": "1000"}
        if tok: q["pageToken"] = tok
        d = json.load(urllib.request.urlopen(API + "?" + urllib.parse.urlencode(q)))
        out += d.get("prefixes", [])
        tok = d.get("nextPageToken")
        if not tok: return out
vers = list_prefixes("mmlu/benchmark_output/runs/")
runs = []
for v in vers:
    for p in list_prefixes(v):
        m = re.search(r"mmlu:subject=([^,]+),method=([^,]+),model=([^,]+),", p)
        if m: runs.append(dict(version=v.split('/')[-2], subject=m.group(1), method=m.group(2), model=m.group(3), prefix=p))
json.dump(runs, open(OUT, 'w'), indent=0)
from collections import Counter
print(len(runs), "runs"); print(Counter(r['version'] for r in runs)); print(Counter(r['method'] for r in runs))
models = Counter(r['model'] for r in runs); print(len(models), "models"); print(models.most_common())
