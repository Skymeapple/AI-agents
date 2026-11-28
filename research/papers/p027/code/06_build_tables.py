"""Derive the compact, committed analysis tables from the raw downloads in data_cache/.

Output: data/registry_slim.jsonl.gz  (one line per registry server, fields used here)
        data/npm_deps.jsonl.gz       (gzip of data_cache/npm_deps.jsonl)
"""
import gzip, json, os, shutil

B = os.path.join(os.path.dirname(__file__), "..")
reg = json.load(open(os.path.join(B, "data_cache", "registry_latest.json")))
with gzip.open(os.path.join(B, "data", "registry_slim.jsonl.gz"), "wt") as f:
    for x in reg:
        s, m = x["server"], x["_meta"]["io.modelcontextprotocol.registry/official"]
        sch = s.get("$schema") or ""
        f.write(json.dumps({
            "name": s["name"], "pub": m["publishedAt"], "upd": m.get("updatedAt"), "status": m["status"],
            "schema": sch.split("/")[-2] if sch.count("/") > 2 else None,
            "remotes": sorted({r.get("type") for r in (s.get("remotes") or [])}),
            "pkgs": [[p.get("registryType"), p.get("identifier"), p.get("version"),
                      (p.get("transport") or {}).get("type")] for p in (s.get("packages") or [])],
        }, separators=(",", ":")) + "\n")
with open(os.path.join(B, "data_cache", "npm_deps.jsonl"), "rb") as i, \
        gzip.open(os.path.join(B, "data", "npm_deps.jsonl.gz"), "wb") as o:
    shutil.copyfileobj(i, o)
print("servers", len(reg))
