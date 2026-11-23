"""Fetch the full official MCP Registry (latest version of every server).

Output: data/registry_latest.json  (list of server records, raw API objects)
Source: https://registry.modelcontextprotocol.io/v0/servers (public, read-only API)
"""
import json, time, urllib.request, urllib.parse, os, sys

BASE = "https://registry.modelcontextprotocol.io/v0/servers"
OUT = os.path.join(os.path.dirname(__file__), "..", "data", "registry_latest.json")


def get(url):
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "research-measurement/0.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # noqa
            print("retry", attempt, e, file=sys.stderr)
            time.sleep(2 + 2 * attempt)
    raise RuntimeError(url)


def main():
    out, cursor, pages = [], None, 0
    while True:
        q = {"limit": 100, "version": "latest"}
        if cursor:
            q["cursor"] = cursor
        d = get(BASE + "?" + urllib.parse.urlencode(q))
        out.extend(d.get("servers", []))
        pages += 1
        cursor = d.get("metadata", {}).get("nextCursor")
        if pages % 10 == 0:
            print(pages, len(out), file=sys.stderr)
        if not cursor:
            break
        time.sleep(0.2)
    json.dump(out, open(OUT, "w"))
    print("servers:", len(out), "pages:", pages)


if __name__ == "__main__":
    main()
