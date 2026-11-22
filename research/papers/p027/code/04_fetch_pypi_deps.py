"""For every PyPI package referenced by an MCP Registry server, fetch PyPI JSON metadata:
requires_dist of the latest release (MCP-related requirements only), and upload dates
of all releases.

Output: data/pypi_deps.jsonl
"""
import json, os, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

D = os.path.join(os.path.dirname(__file__), "..", "data")
KEYS = ("mcp", "fastmcp", "mcp-types")


def norm(n):
    return n.lower().replace("_", "-").replace(".", "-")


def fetch(name):
    url = "https://pypi.org/pypi/%s/json" % name
    for a in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-measurement/0.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {"_error": 404}
            time.sleep(2 + 3 * a)
        except Exception:
            time.sleep(2 + 3 * a)
    return {"_error": "fail"}


def extract(name):
    pk = fetch(name)
    if "_error" in pk:
        return {"name": name, "error": pk["_error"]}
    info = pk.get("info", {})
    reqs = []
    for r in info.get("requires_dist") or []:
        base = r.split(";")[0].strip()
        nm = norm(base.split("[")[0].split("<")[0].split(">")[0].split("=")[0].split("!")[0].split("~")[0].split(" ")[0].split("(")[0])
        if nm in KEYS and "extra ==" not in r:
            reqs.append(r)
        elif nm in KEYS:
            reqs.append("EXTRA:" + r)
    rel = {v: min((f["upload_time_iso_8601"] for f in fs), default=None)
           for v, fs in (pk.get("releases") or {}).items() if fs}
    return {"name": name, "version": info.get("version"), "requires": reqs, "releases": rel}


def main():
    reg = json.load(open(os.path.join(D, "registry_latest.json")))
    names = sorted({p["identifier"] for x in reg for p in (x["server"].get("packages") or [])
                    if p.get("registryType") == "pypi"})
    out = os.path.join(D, "pypi_deps.jsonl")
    with open(out, "w") as f, ThreadPoolExecutor(12) as ex:
        for i, r in enumerate(ex.map(extract, names)):
            f.write(json.dumps(r) + "\n")
            if i % 1000 == 0:
                print(i, file=sys.stderr)
    print("done", len(names))


if __name__ == "__main__":
    main()
