# Lab Dashboard

_Generated 2026-11-20 00:03 UTC · metrics as of 2026-11-19_

| Projects shipped | Downloads (7d) | Downloads (all-time) | ⭐ Stars | Forks | Visitors (14d) | Clones (14d) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 0 | - | - |

## Leaderboard

Value score = 7-day downloads + 3 × unique page visitors (14d).

| # | Project | Shipped | Score | DL 7d | Δ vs last wk | DL 30d | DL total | Page uniques 14d | Trend | Status |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | [ctxlint](projects/ctxlint) | 2026-11-19 | 0 | 0 | - | 0 | 0 | 0 |  | parked |
## Downloads by distribution channel

This tests experiment E2: tools that run automatically (agent hooks, MCP servers, Actions, pre-commit) should earn recurring downloads; manual CLIs shouldn't.

| Channel group | Published projects | Median weekly downloads |
|---|---:|---:|
| auto-invoked (hook/mcp/action/pre-commit) | 0 | - |
| cli-only | 0 | - |

## Experiments

These were written down before the data arrived. A daily mission evaluates each one once its date passes (see `lab/experiments.json`).

| ID | Change | Evaluate on | Status | Current reading |
|---|---|---|---|---|
| E1 | Protocol v2 strict gap gate: no build if a package with >= 1,000 weekly downloads covers >… | 2026-12-20 | running | gated median - vs ungated - (0 gated published) |
| E2 | Protocol v2 distribution rule: every project ships at least one auto-invoked surface (agen… | 2026-12-20 | running | auto-invoked: - (n=0); cli-only: - (n=0) |
| E3 | Improve-first mode: each shipped project gets patch releases (fixes, integrations, README … | 2026-12-31 | running | >=3 releases: - (n=0); fewer: - (n=0) |

## Pain points addressed

| Project | Pain point | Evidence |
|---|---|---|
| ctxlint | Teams run several AI coding agents, each with its own instruction file. The files rot (renamed paths, dead scripts), drift apart, bloat, and can hide malicious invisible Unicode. | [1](https://www.reddit.com/r/codex/comments/1x2wkid/if_you_use_both_codex_and_claude_code_how_do_you) [2](https://umesh-malik.com/blog/agents-md-ai-coding-agents-study) [3](https://www.infoworld.com/article/4224410/claude-code-now-also-accepts-instructions-in-openais-agents-md-format.html) [4](https://finance.yahoo.com/news/vulnerability-github-copilot-cursor-hackers-144000906.html) |
