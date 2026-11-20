# Lab Dashboard

_Generated 2026-11-20 07:50 UTC · metrics as of 2026-11-20_

| Projects shipped | Downloads (7d) | Downloads (all-time) | ⭐ Stars | Forks | Visitors (14d) | Clones (14d) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 0 | 0 | 0 | - | - |

## Leaderboard

Value score = 7-day downloads + 3 × unique page visitors (14d).

| # | Project | Shipped | Score | DL 7d | Δ vs last wk | DL 30d | DL total | Page uniques 14d | Trend | Status |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | [ctxlint](projects/ctxlint) | 2026-11-19 | 0 | 0 | 0 | 0 | 0 | 0 |  | parked |
| 2 | [agentscrub](projects/agentscrub) | 2026-11-20 | 0 | 0 | - | 0 | 0 | 0 |  | active |

## Market benchmarks

These are competitors in each project's category. *Day-30 rate* is a package's downloads in days 24–30 after its first publish, which lets the lab compare a new project against established ones at the same age.

### agent-instruction-linters

Median day-30 rate of small entrants (< 5,000/wk now): **38**

| Package | Weekly downloads now | First 7 days | First 30 days | Day-30 rate |
|---|---:|---:|---:|---:|
| agnix | 42,530 | 1,537 | 3,611 | 492 |
| @yawlabs/ctxlint | 39,400 | 1,524 | 5,557 | 538 |
| claude-code-lint | 14,202 | 294 | 1,013 | 324 |
| agentlinter | 3,963 | 2,067 | 2,344 | 64 |
| @promptci/cli | 2,348 | 309 | 562 | 43 |
| @felixgeelhaar/cclint | 452 | 291 | 400 | 14 |
| tidyagents | 56 | 732 | 779 | 0 |
| unrot | 42 | 2,078 | 2,310 | 34 |
| rulecheck | 26 | 547 | 585 | 0 |
| **ctxlint (lab)** _parked_ | 0 | - | - | - |

### agent-guardrail-hooks

Median day-30 rate of small entrants (< 5,000/wk now): **23**

| Package | Weekly downloads now | First 7 days | First 30 days | Day-30 rate |
|---|---:|---:|---:|---:|
| block-no-verify | 55,877 | 369 | 1,229 | 417 |
| @hasna/hooks | 8,518 | 1,325 | 1,413 | 13 |
| ghostimport | 1,053 | 1,051 | 1,449 | 35 |
| slopcheck | 180 | 522 | 695 | 12 |
| @mharoon07/slopguard | 25 | 443 | 461 | 0 |

### unicode-security

Median day-30 rate of small entrants (< 5,000/wk now): **-**

| Package | Weekly downloads now | First 7 days | First 30 days | Day-30 rate |
|---|---:|---:|---:|---:|
| anti-trojan-source | 40,758 | 365 | 7,504 | 3,655 |
| out-of-character | 8,783 | 193 | 279 | 8 |

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
| agentscrub | Claude Code's "don't ask again" saves full approved commands, passwords included, to .claude/settings.local.json; Apiiro found 1,122 public copies holding 1,379 secrets (Nov 2026). MCP configs and session transcripts leak the same way, and generic scanners miss passwords inside commands. | [1](https://thehackernews.com/2026/11/claude-codes-dont-ask-again-file.html?m=1) [2](https://medium.com/@chinmayshringi4/claude-code-dont-ask-again-saved-secrets-in-settings-local-json-d1323c85bb2d) [3](https://www.darkreading.com/cyber-risk/ai-coding-agents-leak-secrets-github) [4](https://www.backslash.security/blog/secrets-in-ai-agent-session-history) |
| ctxlint | Teams run several AI coding agents, each with its own instruction file. The files rot (renamed paths, dead scripts), drift apart, bloat, and can hide malicious invisible Unicode. | [1](https://www.reddit.com/r/codex/comments/1x2wkid/if_you_use_both_codex_and_claude_code_how_do_you) [2](https://umesh-malik.com/blog/agents-md-ai-coding-agents-study) [3](https://www.infoworld.com/article/4224410/claude-code-now-also-accepts-instructions-in-openais-agents-md-format.html) [4](https://finance.yahoo.com/news/vulnerability-github-copilot-cursor-hackers-144000906.html) |
