# Round 2: inspect the structure of candidate filing and price datasets.
python3 - <<'PY'
import json, urllib.request
def get(u, n=None):
    r = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "research-script"}), timeout=120)
    return r.read(n) if n else json.load(r)
for repo in ("CyberMax-tools/figurewell-sec-xbrl-financials", "DenyTranDFW/edgar_xbrl_companyfacts", "no-ry/world-stock-prices-daily-updating", "FinzData/sec-financial-statements"):
    print(f"\n===== {repo}")
    try:
        info = get(f"https://huggingface.co/api/datasets/{repo}")
        print("card:", (info.get("cardData") or {}).get("license"), [s["rfilename"] for s in info.get("siblings", [])][:30])
        pq = get(f"https://huggingface.co/api/datasets/{repo}/parquet")
        print("parquet:", json.dumps(pq)[:600])
        readme = get(f"https://huggingface.co/datasets/{repo}/resolve/main/README.md", 2500).decode("utf-8", "replace")
        print("README:", readme[:2500])
    except Exception as e:
        print("ERR", e)
PY
pip -q install pyarrow fsspec aiohttp
python3 - <<'PY'
import json, urllib.request, pyarrow.parquet as pq, fsspec
def get(u):
    return json.load(urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "research-script"}), timeout=120))
for repo in ("CyberMax-tools/figurewell-sec-xbrl-financials", "no-ry/world-stock-prices-daily-updating"):
    try:
        urls = [u for c in get(f"https://huggingface.co/api/datasets/{repo}/parquet").values() for s in c.values() for u in s]
        print(f"\n== {repo}: {len(urls)} parquet files; first {urls[0]}")
        with fsspec.open(urls[0]).open() as f:
            p = pq.ParquetFile(f)
            print(p.schema_arrow)
            print("rows in first file:", p.metadata.num_rows)
            print(p.read_row_group(0).slice(0, 5).to_pandas().to_string()[:2500])
    except Exception as e:
        print("ERR", repo, e)
PY
