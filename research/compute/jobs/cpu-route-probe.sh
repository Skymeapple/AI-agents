# Check public resources that would give CPU-only routes for GPU-blocked projects:
# pre-extracted pathology slide features, crystal-generation baselines and reference
# data, and the full-atlas lung embedding.
python3 - <<'PY'
import json, urllib.request, urllib.error
def get(url, n=300000):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-script"}), timeout=60) as r:
            return r.status, r.read(n)
    except urllib.error.HTTPError as e:
        return e.code, e.read(300)
    except Exception as e:
        return "ERR", str(e)[:150].encode()
for kind, repo in [("datasets", "MahmoodLab/UNI2-h-features"), ("datasets", "MahmoodLab/Patho-Bench"), ("datasets", "owkin/camelyon16-features"),
                   ("datasets", "bioptimus/H-optimus-0-TCGA-features"), ("datasets", "prov-gigapath/prov-gigapath-tcga-embeddings"),
                   ("models", "MahmoodLab/UNI"), ("models", "MahmoodLab/TITAN"), ("datasets", "microsoft/mattergen"), ("datasets", "fairchem/OMAT24"),
                   ("datasets", "nimashoghi/mptrj"), ("datasets", "chaitjo/MP20_ADiT"), ("datasets", "roneneldan/TinyStories")]:
    s, body = get(f"https://huggingface.co/api/{kind}/{repo}?blobs=true")
    try:
        d = json.loads(body)
        files = d.get("siblings", [])
        total = sum(f.get("size") or 0 for f in files) / 1e9
        print(f"{s} {kind} {repo}: gated={d.get('gated')} private={d.get('private')} files={len(files)} total={total:.1f} GB")
        for f in files[:12]:
            print(f"      {f['rfilename']}  {((f.get('size') or 0) / 1e6):.1f} MB")
    except Exception:
        print(f"{s} {kind} {repo}: {body[:120]}")
for name, url in [("matgen_baselines repo", "https://api.github.com/repos/Bartel-Group/matgen_baselines"),
                  ("matgen_baselines tree", "https://api.github.com/repos/Bartel-Group/matgen_baselines/git/trees/main?recursive=1"),
                  ("mattersim repo", "https://api.github.com/repos/microsoft/mattersim"),
                  ("HLCA full embedding (Zenodo)", "https://zenodo.org/api/records/7599104")]:
    s, body = get(url)
    try:
        d = json.loads(body)
        if "tree" in d:
            print(s, name, [t["path"] for t in d["tree"]][:60])
        elif "files" in d:
            print(s, name, [(f["key"], round(f["size"] / 1e6)) for f in d["files"]])
        else:
            print(s, name, d.get("description"), d.get("size"), (d.get("license") or {}).get("spdx_id"))
    except Exception:
        print(s, name, body[:150])
PY
