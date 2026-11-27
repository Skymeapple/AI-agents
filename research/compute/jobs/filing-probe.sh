# Look for public financial-statement (XBRL) and daily-price datasets on Hugging Face that need no SEC contact address.
python3 - <<'PY'
import json, urllib.request
def get(u):
    return json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "research-script"}), timeout=60))
for q in ("xbrl", "sec financial statements", "edgar financial", "financial statement data sets", "10-K fundamentals",
          "stock prices daily", "us stock daily ohlcv", "sp500 daily prices", "earnings announcements", "compustat"):
    try:
        res = get(f"https://huggingface.co/api/datasets?search={urllib.request.quote(q)}&limit=15&full=true&sort=downloads")
    except Exception as e:
        print(q, "ERR", e); continue
    print(f"\n== {q}: {len(res)}")
    for d in res:
        files = [s["rfilename"] for s in d.get("siblings", [])]
        print(f"  {d['id']} dl={d.get('downloads')} gated={d.get('gated')} lic={[t for t in d.get('tags', []) if t.startswith('license')]} n_files={len(files)} e.g. {files[:5]}")
PY
