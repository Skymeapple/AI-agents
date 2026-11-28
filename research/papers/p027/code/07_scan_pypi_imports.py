"""Static import scan of registry-listed PyPI servers (artifacts are read, never executed).

For every PyPI package in data/pypi_deps.jsonl that declares a direct, non-optional
requirement on the official `mcp` SDK, download the wheel (or sdist) of the recorded
release (<= 2 MB), and record which `mcp.*` modules its Python sources import.
Purpose: `mcp` 2.0 removed several v1 module paths (e.g. `mcp.server.fastmcp`); a package
whose requirement range admits mcp 2.x but which imports a removed path cannot import
under a fresh dependency resolution.

Output: data/pypi_imports.jsonl
"""
import io, json, os, re, sys, tarfile, time, zipfile, urllib.request
from concurrent.futures import ThreadPoolExecutor

D = os.path.join(os.path.dirname(__file__), "..", "data")
CAP = 2_000_000
IMP = re.compile(r"^\s*(?:from\s+(mcp(?:\.[A-Za-z_][\w]*)*)\s+import|import\s+(mcp(?:\.[A-Za-z_][\w]*)*))", re.M)


def fetch(url):
    for a in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-measurement/0.1"})
            with urllib.request.urlopen(req, timeout=90) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 + 3 * a)
        except Exception:
            time.sleep(2 + 3 * a)
    return None


def sources(blob, fname):
    if fname.endswith(".whl") or fname.endswith(".zip"):
        z = zipfile.ZipFile(io.BytesIO(blob))
        for n in z.namelist():
            if n.endswith(".py") and z.getinfo(n).file_size < 2_000_000:
                yield n, z.read(n).decode("utf8", "ignore")
    else:
        tf = tarfile.open(fileobj=io.BytesIO(blob), mode="r:*")
        for m in tf.getmembers():
            if m.isfile() and m.name.endswith(".py") and m.size < 2_000_000:
                yield m.name, tf.extractfile(m).read().decode("utf8", "ignore")


def one(rec):
    name, ver = rec["name"], rec["version"]
    out = {"name": name, "version": ver}
    b = fetch("https://pypi.org/pypi/%s/%s/json" % (name, ver))
    if b is None:
        return dict(out, error="meta")
    urls = json.loads(b).get("urls") or []
    whl = [u for u in urls if u["filename"].endswith(".whl")]
    cand = whl or [u for u in urls if u["filename"].endswith((".tar.gz", ".zip"))]
    if not cand:
        return dict(out, error="nofile")
    u = min(cand, key=lambda u: u["size"])
    if u["size"] > CAP:
        return dict(out, error="toolarge", size=u["size"])
    blob = fetch(u["url"])
    if blob is None:
        return dict(out, error="download")
    mods, nfiles = set(), 0
    try:
        for n, t in sources(blob, u["filename"]):
            if "/tests/" in n or n.startswith("tests/"):
                continue
            nfiles += 1
            for m in IMP.finditer(t):
                mods.add(m.group(1) or m.group(2))
    except Exception as e:
        return dict(out, error="parse:" + type(e).__name__)
    return dict(out, size=u["size"], uploaded=u["upload_time_iso_8601"], nfiles=nfiles, mods=sorted(mods))


def main():
    recs = []
    for line in open(os.path.join(D, "pypi_deps.jsonl")):
        r = json.loads(line)
        if "error" in r:
            continue
        if any(re.match(r"^mcp(\[[^\]]*\])?\s*([<>=!~(,]|$)", q.split(";")[0].strip().lower()) for q in r["requires"]
               if not q.startswith("EXTRA:")):
            recs.append(r)
    out = os.path.join(D, "pypi_imports.jsonl")
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["name"] for l in open(out)}
    todo = [r for r in recs if r["name"] not in done]
    print("direct-mcp packages", len(recs), "todo", len(todo), file=sys.stderr)
    with open(out, "a") as f, ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(one, todo)):
            f.write(json.dumps(r) + "\n")
            if i % 500 == 0:
                f.flush(); print(i, file=sys.stderr)


if __name__ == "__main__":
    main()
