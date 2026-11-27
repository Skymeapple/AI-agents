# Find the real download links for LOBSTER's free sample files (round 2: the site is a single-page app).
curl -sS -L -m 120 https://lobsterdata.com/assets/index-9e0decf7.js -o app.js; ls -la app.js
grep -o '"[^"]*\(sample\|Sample\)[^"]*"' app.js | sort -u | head -40
grep -o '[A-Za-z0-9_/.:-]*\.\(zip\|7z\)' app.js | sort -u | head -40
python3 - <<'PY'
import json, urllib.request
for q in ("lobster", "limit order book nasdaq", "LOBSTER_SampleFile"):
    u = f"https://huggingface.co/api/datasets?search={urllib.request.quote(q)}&limit=20&full=true"
    try:
        res = json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "research-script"}), timeout=60))
    except Exception as e:
        print(q, "ERR", e); continue
    print(f"== HF {q}: {len(res)}")
    for d in res:
        print("  ", d["id"], [t for t in d.get("tags", []) if t.startswith("license")], [s["rfilename"] for s in d.get("siblings", [])][:8])
PY
