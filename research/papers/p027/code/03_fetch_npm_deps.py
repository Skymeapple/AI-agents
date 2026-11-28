"""For every npm package referenced by an MCP Registry server, fetch its npm packument
and keep, for every published version: publish time and the declared dependency
ranges on the official MCP SDK packages.

Output: data_cache/npm_deps.jsonl  (one JSON object per package)
"""
import json, os, sys, time, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor

D = os.path.join(os.path.dirname(__file__), "..", "data")
SDK_PKGS = ["@modelcontextprotocol/sdk", "@modelcontextprotocol/server",
            "@modelcontextprotocol/client", "@modelcontextprotocol/core",
            "fastmcp", "mcp-framework", "@mastra/mcp", "mcp-use", "xmcp", "mcp-handler",
            "@vercel/mcp-adapter", "agents", "@cloudflare/workers-oauth-provider"]


def fetch(name):
    url = "https://registry.npmjs.org/" + urllib.parse.quote(name, safe="@")
    url = url.replace("@", "%40", 1) if name.startswith("@") else url
    url = url.replace("%40", "@", 1)  # npm accepts @scope%2Fname
    url = "https://registry.npmjs.org/" + name.replace("/", "%2F")
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
    t = pk.get("time", {})
    vers = {}
    for v, m in (pk.get("versions") or {}).items():
        deps = {}
        for field in ("dependencies", "peerDependencies", "devDependencies"):
            for k, r in (m.get(field) or {}).items():
                if k in SDK_PKGS:
                    deps[field[:3] + ":" + k] = r
        vers[v] = {"t": t.get(v), "deps": deps,
                   "bundled": bool(m.get("bundleDependencies") or m.get("bundledDependencies")),
                   "ndeps": len(m.get("dependencies") or {})}
    return {"name": name, "latest": (pk.get("dist-tags") or {}).get("latest"),
            "created": t.get("created"), "modified": t.get("modified"), "versions": vers}


def main():
    reg = json.load(open(os.path.join(D, "..", "data_cache", "registry_latest.json")))
    names = sorted({p["identifier"] for x in reg for p in (x["server"].get("packages") or [])
                    if p.get("registryType") == "npm"})
    out = os.path.join(D, "..", "data_cache", "npm_deps.jsonl")
    done = set()
    if os.path.exists(out):
        for line in open(out):
            done.add(json.loads(line)["name"])
    todo = [n for n in names if n not in done]
    print("total", len(names), "todo", len(todo), file=sys.stderr)
    with open(out, "a") as f, ThreadPoolExecutor(16) as ex:
        for i, r in enumerate(ex.map(extract, todo)):
            f.write(json.dumps(r) + "\n")
            if i % 1000 == 0:
                print(i, file=sys.stderr); f.flush()


if __name__ == "__main__":
    main()
