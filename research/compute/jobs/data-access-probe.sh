# Check that the public datasets and models the remaining projects depend on can be
# reached from free compute, and report sizes where the server gives them.
python3 - <<'PY'
import json, urllib.request, urllib.error

def get(url, n=2000, method="GET", headers=None):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "research-script", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            body = r.read(n) if method == "GET" else b""
            return r.status, r.headers.get("Content-Length"), body
    except urllib.error.HTTPError as e:
        return e.code, None, e.read(300)
    except Exception as e:
        return "ERR", None, str(e)[:200].encode()

def line(project, what, url, **kw):
    s, size, body = get(url, **kw)
    size = f"{int(size)/1e6:.1f} MB" if size and size.isdigit() else "-"
    print(f"{project:<28} | {str(s):<4} | {size:>10} | {what} | {url}", flush=True)
    return s, body

def hf(project, kind, repo):
    s, body = line(project, f"Hugging Face {kind} {repo}", f"https://huggingface.co/api/{kind}/{repo}", n=400000)
    try:
        d = json.loads(body)
        files = [x["rfilename"] for x in d.get("siblings", [])]
        print(f"{'':<28}   gated={d.get('gated')} files={len(files)} e.g. {files[:6]}")
    except Exception:
        pass

print("project | http | size | resource | url")
# earth science
s, body = line("weather-24h", "WeatherBench 2 bucket listing", "https://storage.googleapis.com/storage/v1/b/weatherbench2/o?prefix=datasets/&delimiter=/&maxResults=200", n=200000)
try:
    print("   WB2 dataset folders:", [p.split("/")[1] for p in json.loads(body).get("prefixes", [])])
except Exception as e:
    print("   could not parse listing", e)
for model in ("era5", "graphcast", "pangu", "hres", "neuralgcm_deterministic", "fuxi", "keisler", "sphericalcnn"):
    s, body = get(f"https://storage.googleapis.com/storage/v1/b/weatherbench2/o?prefix=datasets/{model}/&delimiter=/&maxResults=100", n=200000)
    try:
        print(f"   {model}:", [p.split('/')[-2] for p in json.loads(body).get("prefixes", [])][:25])
    except Exception:
        print(f"   {model}: -")
hf("weather-24h", "models", "gcouairon/ArchesWeather")
# physics
hf("rayleigh-benard", "datasets", "polymathic-ai/rayleigh_benard")
hf("rayleigh-benard", "models", "polymathic-ai/walrus")
line("pm25-calibration", "SensEURCity record", "https://zenodo.org/api/records/17858205", n=300)
# biology / medicine / materials / astronomy
line("lung-cell-annotation", "HLCA collection (CELLxGENE)", "https://api.cellxgene.cziscience.com/curation/v1/collections/6f6d381a-7701-4781-935c-db10d30de293", n=3000)
hf("lung-cell-annotation", "models", "minwoosun/uce-100m")
hf("regulatory-dna-mpra", "models", "arcinstitute/evo2_7b")
hf("lung-histology", "models", "owkin/phikon-v2")
line("lung-histology", "GDC API: TCGA-LUAD/LUSC diagnostic slide count", "https://api.gdc.cancer.gov/files?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.project.project_id%22%2C%22value%22%3A%5B%22TCGA-LUAD%22%2C%22TCGA-LUSC%22%5D%7D%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22experimental_strategy%22%2C%22value%22%3A%5B%22Diagnostic%20Slide%22%5D%7D%7D%5D%7D&size=0", n=600)
hf("crystal-generation", "models", "microsoft/mattergen")
hf("crystal-generation", "models", "microsoft/mattersim")
hf("photometric-redshift", "models", "polymathic-ai/aion-base")
line("photometric-redshift", "Legacy Survey cutout service", "https://www.legacysurvey.org/viewer/cutout.jpg?ra=190.1086&dec=1.2005&layer=ls-dr9&pixscale=0.262&size=64", n=100)
# agent harnesses and post-training
for proj, kind, repo in [("table-qa-agent", "datasets", "cardiffnlp/databench"), ("dafny-agent", "datasets", "wendy-sun/DafnyBench"),
                         ("paper-memory-agent", "datasets", "allenai/qasper"), ("synlogic-training", "datasets", "MiniMaxAI/SynLogic"),
                         ("embspatial-training", "datasets", "FlagEval/EmbSpatial-Bench"), ("fabner-extraction", "datasets", "DFKI-SLT/fabner"),
                         ("post-training", "models", "Qwen/Qwen2.5-7B-Instruct"), ("post-training", "models", "Qwen/Qwen2-VL-2B-Instruct"),
                         ("pretraining", "datasets", "HuggingFaceFW/fineweb-edu"), ("pretraining", "datasets", "mlfoundations/dclm-baseline-1.0")]:
    hf(proj, kind, repo)
for proj, repo in [("arc-grid-agent", "fchollet/ARC-AGI"), ("lean-prover-agent", "openai/miniF2F"), ("terminal-agent-harness", "laude-institute/terminal-bench"),
                   ("scienceworld-training", "allenai/ScienceWorld"), ("mapqa-training", "OSU-slatelab/MapQA"), ("topv2-parsing", "facebookresearch/low-resource-semantic-parsing")]:
    line(proj, f"GitHub repo {repo}", f"https://api.github.com/repos/{repo}", n=300)
# operations
line("bus-holding", "HSL vehicle positions (HFP) info", "https://hfp.digitransit.fi/", n=300)
line("bus-holding", "HSL HFP daily archive container", "https://hslstoragekarttatuotanto.blob.core.windows.net/hfp-v2?restype=container&comp=list&maxresults=3", n=1500)
line("bike-rebalancing", "Helsinki bike stations with capacity", "https://opendata.arcgis.com/datasets/726277c507ef4914b0aec3cbcfcbfafc_0.csv", n=400)
line("ride-hail-dispatch", "NYC TLC for-hire trips 2024-01", "https://d37ci6vzurychx.cloudfront.net/trip-data/fhvhv_tripdata_2024-01.parquet", method="HEAD")
line("dam-spill", "Columbia Basin DART", "https://www.cbr.washington.edu/dart/query/river_daily", n=300)
line("arterial-signals", "Toronto open data: traffic volumes", "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action/package_show?id=traffic-volumes-at-intersections-for-all-modes", n=1500)
# finance
line("portfolio-with-frictions", "Qlib community data release", "https://api.github.com/repos/chenditc/investment_data/releases/latest", n=1500)
line("cross-venue-routing", "Bybit public trade archive", "https://public.bybit.com/trading/BTCUSDT/", n=400)
line("cross-venue-routing", "OKX public market data API", "https://www.okx.com/api/v5/market/history-trades?instId=BTC-USDT-SWAP&limit=1", n=400)
line("order-book projects", "LOBSTER free sample", "https://lobsterdata.com/info/sample/LOBSTER_SampleFile_AMZN_2012-06-21_10.zip", method="HEAD")
line("filing-event-strategy", "SEC financial statement data sets", "https://www.sec.gov/files/dera/data/financial-statement-data-sets/2024q1.zip", method="HEAD")
PY
