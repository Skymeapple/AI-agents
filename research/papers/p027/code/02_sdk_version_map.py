"""Build SDK-release -> supported MCP protocol versions map.

For every release of the official TypeScript SDK (npm @modelcontextprotocol/sdk) and
the official Python SDK (PyPI `mcp`), download the published artifact, read (never
execute) the source file that declares LATEST_PROTOCOL_VERSION and
SUPPORTED_PROTOCOL_VERSIONS, and record them with the release date.

Output: data/sdk_versions_ts.json, data/sdk_versions_py.json
"""
import io, json, os, re, sys, tarfile, zipfile, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

D = os.path.join(os.path.dirname(__file__), "..", "data")
DATE = re.compile(r"[\"'](\d{4}-\d{2}-\d{2})[\"']")


def fetch(url, accept=None):
    for a in range(5):
        try:
            h = {"User-Agent": "research-measurement/0.1"}
            if accept:
                h["Accept"] = accept
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120) as r:
                return r.read()
        except Exception as e:
            print("retry", url, e, file=sys.stderr)
            time.sleep(2 + 3 * a)
    return None


def parse_versions(text):
    """Return (latest, supported list) from TS/JS/Python source text."""
    latest = None
    m = re.search(r"LATEST_PROTOCOL_VERSION\s*[:=][^\n]*?[\"'](\d{4}-\d{2}-\d{2})[\"']", text)
    if m:
        latest = m.group(1)
    sup = None
    m = re.search(r"SUPPORTED_PROTOCOL_VERSIONS[^=\n]*=\s*\[([^\]]*)\]", text, re.S)
    if m:
        body = m.group(1)
        sup = DATE.findall(body)
        if "LATEST_PROTOCOL_VERSION" in body and latest:
            sup = [latest] + sup
        if "DEFAULT_NEGOTIATED_PROTOCOL_VERSION" in body:
            m2 = re.search(r"DEFAULT_NEGOTIATED_PROTOCOL_VERSION\s*=\s*[\"'](\d{4}-\d{2}-\d{2})", text)
            if m2:
                sup.append(m2.group(1))
    return latest, sup


def ts_one(v, meta):
    url = meta["dist"]["tarball"]
    blob = fetch(url)
    if blob is None:
        return None
    tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")
    texts = []
    for mem in tf.getmembers():
        n = mem.name
        if mem.isfile() and (n.endswith("types.js") or n.endswith("types.ts") or "constants" in n) and "esm" in n or n.endswith("dist/types.js"):
            f = tf.extractfile(mem)
            if f:
                texts.append(f.read().decode("utf8", "ignore"))
    # fallback: any .js file containing the constant
    if not any("SUPPORTED_PROTOCOL_VERSIONS" in t for t in texts):
        for mem in tf.getmembers():
            if mem.isfile() and mem.name.endswith(".js") and mem.size < 3_000_000:
                t = tf.extractfile(mem).read().decode("utf8", "ignore")
                if "SUPPORTED_PROTOCOL_VERSIONS" in t:
                    texts.append(t)
                    break
    text = "\n".join(t for t in texts if "PROTOCOL_VERSION" in t)
    latest, sup = parse_versions(text)
    return {"sdk": v, "lang": "ts", "latest": latest, "supported": sup}


def ts():
    pk = json.loads(open(os.path.join(D, "sdk_npm.json")).read())
    times = pk["time"]
    vs = [v for v in pk["versions"]]
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda v: ts_one(v, pk["versions"][v]), vs))
    out = []
    for v, r in zip(vs, res):
        if r:
            r["date"] = times[v]
            out.append(r)
    json.dump(out, open(os.path.join(D, "sdk_versions_ts.json"), "w"), indent=1)
    print("ts releases", len(out))


def py_one(v, files):
    whl = [f for f in files if f["filename"].endswith(".whl")]
    src = whl or [f for f in files if f["filename"].endswith(".tar.gz")]
    if not src:
        return None
    f = src[0]
    blob = fetch(f["url"])
    if blob is None:
        return None
    text = ""
    if f["filename"].endswith(".whl"):
        z = zipfile.ZipFile(io.BytesIO(blob))
        for n in z.namelist():
            if n.endswith("shared/version.py") or n.endswith("mcp/types.py"):
                text += z.read(n).decode("utf8", "ignore") + "\n"
    else:
        tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz")
        for mem in tf.getmembers():
            if mem.name.endswith("shared/version.py") or mem.name.endswith("mcp/types.py"):
                text += tf.extractfile(mem).read().decode("utf8", "ignore") + "\n"
    latest, sup = parse_versions(text)
    return {"sdk": v, "lang": "py", "latest": latest, "supported": sup, "date": f["upload_time_iso_8601"]}


def py():
    pk = json.loads(fetch("https://pypi.org/pypi/mcp/json"))
    json.dump(pk, open(os.path.join(D, "mcp_pypi.json"), "w"))
    rel = {v: fs for v, fs in pk["releases"].items() if fs}
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda kv: py_one(*kv), rel.items()))
    out = [r for r in res if r]
    out.sort(key=lambda r: r["date"])
    json.dump(out, open(os.path.join(D, "sdk_versions_py.json"), "w"), indent=1)
    print("py releases", len(out))


if __name__ == "__main__":
    ts()
    py()
