# Inspect file sizes and internal layout of two large public datasets without
# downloading them in full: The Well's Rayleigh-Benard files and the Human Lung
# Cell Atlas. Remote HDF5 files are opened over HTTP and only metadata is read.
pip -q install h5py fsspec aiohttp requests numpy
python - <<'PY'
import json, requests, fsspec, h5py

print("== The Well: rayleigh_benard file sizes")
d = requests.get("https://huggingface.co/api/datasets/polymathic-ai/rayleigh_benard?blobs=true", timeout=60).json()
files = [(s["rfilename"], s.get("size") or 0) for s in d.get("siblings", []) if s["rfilename"].endswith(".hdf5")]
by_split = {}
for f, sz in files:
    by_split.setdefault(f.split("/")[1], []).append(sz)
for k, v in by_split.items():
    print(f"  {k}: {len(v)} files, {sum(v)/1e9:.1f} GB total, {min(v)/1e9:.2f}-{max(v)/1e9:.2f} GB each")
small = min((f for f in files if "/test/" in f[0]), key=lambda x: x[1])
print("  smallest test file:", small)
url = f"https://huggingface.co/datasets/polymathic-ai/rayleigh_benard/resolve/main/{small[0]}"
with fsspec.open(url, "rb", block_size=2**20) as fh, h5py.File(fh, "r") as h:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"    {name}: shape {obj.shape}, {obj.dtype}, chunks {obj.chunks}")
    h.visititems(show)
    print("    root attrs:", {k: (v if not hasattr(v, 'shape') or v.size < 6 else v.shape) for k, v in h.attrs.items()})

print("== Human Lung Cell Atlas on CELLxGENE")
c = requests.get("https://api.cellxgene.cziscience.com/curation/v1/collections/6f6d381a-7701-4781-935c-db10d30de293", timeout=60).json()
core = None
for ds in c.get("datasets", []):
    assets = {a["filetype"]: (a["filesize"], a["url"]) for a in ds.get("assets", [])}
    print(f"  {ds.get('title')}: {ds.get('cell_count')} cells; " + ", ".join(f"{k} {v[0]/1e9:.1f} GB" for k, v in assets.items()))
    if "core" in (ds.get("title") or "").lower() and "H5AD" in assets:
        core = assets["H5AD"][1]
if core:
    print("  opening core file remotely:", core)
    with fsspec.open(core, "rb", block_size=4 * 2**20) as fh, h5py.File(fh, "r") as h:
        print("    top level:", list(h.keys()))
        print("    obsm:", {k: h["obsm"][k].shape for k in h["obsm"].keys()})
        print("    obs columns:", list(h["obs"].keys())[:80])
        x = h["X"]
        print("    X:", dict(x.attrs) if hasattr(x, "attrs") else "", list(x.keys()) if isinstance(x, h5py.Group) else x.shape)
        for col in ("study", "dataset", "ann_finest_level", "cell_type", "donor_id"):
            if col in h["obs"]:
                g = h["obs"][col]
                n = len(g["categories"]) if isinstance(g, h5py.Group) and "categories" in g else "?"
                print(f"    obs[{col}]: {n} categories")
PY
