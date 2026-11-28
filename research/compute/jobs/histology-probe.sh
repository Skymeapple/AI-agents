# Find public lung histology patch datasets on Hugging Face that free compute can read.
python3 - <<'PY'
import json, urllib.request
def get(u):
    with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "research-script"}), timeout=60) as r:
        return json.load(r)
for q in ("lc25000", "lung histopathology", "lung cancer histology", "LUAD LUSC", "histopathology lung", "tcga lung patches"):
    try:
        res = get(f"https://huggingface.co/api/datasets?search={urllib.request.quote(q)}&limit=20&full=true")
    except Exception as e:
        print(q, "ERR", e); continue
    print(f"\n== {q}: {len(res)}")
    for d in res:
        files = [s["rfilename"] for s in d.get("siblings", [])][:6]
        print(f"  {d['id']} gated={d.get('gated')} downloads={d.get('downloads')} license={[t for t in d.get('tags', []) if t.startswith('license')]} files={files}")
PY
