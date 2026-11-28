"""Supplementary fetches (read-only, artifacts are read, never executed).

(a) v2 SDK lines: which releases of PyPI `mcp-types` (the types package pinned by
    `mcp` 2.x) and npm `@modelcontextprotocol/server` implement the stateless
    2026-07-28 revision.
(b) FastMCP (PyPI `fastmcp`): per-release requirement on `mcp`, so that servers that
    reach the SDK through FastMCP can be resolved.
(c) `mcp` per-release requirement on `mcp-types` (to tie mcp 2.x to its types release).

Output: data/sdk_v2_map.json, data/fastmcp_pypi_reqs.json
"""
import io, json, os, re, sys, tarfile, time, zipfile, urllib.request
from concurrent.futures import ThreadPoolExecutor

D = os.path.join(os.path.dirname(__file__), "..", "data")
REV = "2026-07-28"


def fetch(url):
    for a in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-measurement/0.1"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except Exception as e:
            print("retry", url, e, file=sys.stderr)
            time.sleep(2 + 3 * a)
    return None


def py_types_one(v, files):
    whl = [f for f in files if f["filename"].endswith(".whl")]
    if not whl:
        return None
    z = zipfile.ZipFile(io.BytesIO(fetch(whl[0]["url"])))
    modern = None
    for n in z.namelist():
        if n.endswith("mcp_types/version.py"):
            t = z.read(n).decode("utf8", "ignore")
            m = re.search(r"MODERN_PROTOCOL_VERSIONS[^=\n]*=\s*\(([^)]*)\)", t)
            modern = re.findall(r"\d{4}-\d{2}-\d{2}", m.group(1)) if m else []
    return {"pkg": "mcp-types", "v": v, "date": whl[0]["upload_time_iso_8601"], "modern": modern}


def ts_server_one(v, meta, t):
    tf = tarfile.open(fileobj=io.BytesIO(fetch(meta["dist"]["tarball"])), mode="r:gz")
    found = False
    for mem in tf.getmembers():
        if mem.isfile() and mem.name.endswith((".js", ".mjs", ".cjs")) and mem.size < 8_000_000:
            s = tf.extractfile(mem).read().decode("utf8", "ignore")
            if re.search(r"(FIRST_MODERN_PROTOCOL_VERSION|MODERN_WIRE_REVISION)\s*=\s*\"" + REV, s):
                found = True
                break
    return {"pkg": "@modelcontextprotocol/server", "v": v, "date": t, "modern": [REV] if found else []}


def pypi_reqs(name, key, extras=False):
    pk = json.loads(fetch("https://pypi.org/pypi/%s/json" % name))
    rel = {v: fs for v, fs in pk["releases"].items() if fs}

    def one(v):
        b = fetch("https://pypi.org/pypi/%s/%s/json" % (name, v))
        if b is None:
            return None
        info = json.loads(b)["info"]
        reqs = [r for r in (info.get("requires_dist") or [])
                if re.match(r"^%s(\[[^\]]*\])?\s*([<>=!~(,]|$)" % re.escape(key), r.split(";")[0].strip().lower().replace("_", "-"))
                and (extras or "extra ==" not in r)]
        return {"v": v, "date": min(f["upload_time_iso_8601"] for f in rel[v]), "req": reqs}

    with ThreadPoolExecutor(8) as ex:
        return [r for r in ex.map(one, rel) if r]


def main():
    out = []
    pk = json.load(open(os.path.join(D, "mcp_types_pypi.json")))
    for v, fs in pk["releases"].items():
        if fs:
            r = py_types_one(v, fs)
            if r:
                out.append(r)
    sv = json.load(open(os.path.join(D, "server_npm.json")))
    for v, meta in sv["versions"].items():
        out.append(ts_server_one(v, meta, sv["time"][v]))
    json.dump(out, open(os.path.join(D, "sdk_v2_map.json"), "w"), indent=1)
    # fastmcp >= 3.3 is a thin wrapper around fastmcp-slim[client,server]==<same version>;
    # the mcp requirement then sits behind fastmcp-slim's "server" extra.
    fm = {"fastmcp->mcp": pypi_reqs("fastmcp", "mcp"), "mcp->mcp-types": pypi_reqs("mcp", "mcp-types"),
          "fastmcp-slim->mcp": pypi_reqs("fastmcp-slim", "mcp", extras=True)}
    json.dump(fm, open(os.path.join(D, "fastmcp_pypi_reqs.json"), "w"), indent=1)
    # (d) module paths present in the last mcp 1.x wheel but absent from each stable mcp 2.x wheel
    pk = json.loads(fetch("https://pypi.org/pypi/mcp/json"))

    def modules(v):
        whl = [f for f in pk["releases"][v] if f["filename"].endswith(".whl")][0]
        z = zipfile.ZipFile(io.BytesIO(fetch(whl["url"])))
        mods = set()
        for n in z.namelist():
            if n.endswith(".py") and n.startswith("mcp/"):
                # a "tombstone" module whose only statement raises ModuleNotFoundError
                # (mcp >= 2.2 ships one for mcp.server.fastmcp) counts as absent
                if z.getinfo(n).file_size < 2000 and re.search(r"^raise ModuleNotFoundError\(", z.read(n).decode("utf8", "ignore"), re.M):
                    tomb.setdefault(v, []).append(n)
                    continue
                mods.add(n[:-3].replace("/", ".").replace(".__init__", ""))
        return mods

    tomb = {}
    base = modules("1.30.0")
    rem = {"baseline": "1.30.0", "removed": {}, "tombstones": tomb}
    for v in ["2.0.0", "2.1.0", "2.2.0", "2.3.0"]:
        rem["removed"][v] = sorted(base - modules(v))
    json.dump(rem, open(os.path.join(D, "mcp_v2_removed_modules.json"), "w"), indent=1)
    print("v2 map", len(out), "fastmcp releases", len(fm["fastmcp->mcp"]), "mcp releases", len(fm["mcp->mcp-types"]))


if __name__ == "__main__":
    main()
