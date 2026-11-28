# Code for "Stranded or Broken"

Python 3 with `numpy`, `matplotlib`, `packaging` and `node-semver` (`pip install node-semver`).

## Reproduce the paper's numbers (offline, about 10 seconds)

```
python3 code/08_analyze.py   # -> results/summary.json, reach_weekly.csv, cohort_weekly.csv, fig1-3
python3 code/09_model.py     # -> results/model.json, fig4
```

Both read only the committed tables in `data/`.

## Re-collect the data (network; changes the snapshot date)

```
python3 code/01_fetch_registry.py        # official MCP Registry  -> data_cache/registry_latest.json (67 MB, git-ignored)
python3 code/02_sdk_version_map.py       # v1 SDK release -> protocol revisions (reads published artifacts)
python3 code/03_fetch_npm_deps.py        # npm packuments of registry servers -> data_cache/npm_deps.jsonl
python3 code/04_fetch_pypi_deps.py       # PyPI metadata of registry servers  -> data/pypi_deps.jsonl
python3 code/05_fetch_frameworks_v2.py   # v2 SDK revision map, FastMCP requirements, mcp 1.x -> 2.x removed modules
python3 code/06_build_tables.py          # data_cache/* -> data/registry_slim.jsonl.gz, data/npm_deps.jsonl.gz
python3 code/07_scan_pypi_imports.py     # static import scan of PyPI wheels (about 4 minutes)
```

`02` also needs the packuments `data/sdk_npm.json`, `data/server_npm.json` and `data/mcp_types_pypi.json`
(fetch `https://registry.npmjs.org/@modelcontextprotocol%2Fsdk`, `.../@modelcontextprotocol%2Fserver`
and `https://pypi.org/pypi/mcp-types/json`). Published artifacts are downloaded and read as text;
nothing downloaded is executed. The snapshot used in the paper was collected up to 2026-11-22
(steps 01-04) and on 2026-11-28 (steps 05 and 07).
