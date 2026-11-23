# Lab Dashboard

_Generated 2026-11-23 07:43 UTC · metrics as of 2026-11-23_

| Projects shipped | Downloads (7d) | Downloads (all-time) | ⭐ Stars | Forks | Visitors (14d) | Clones (14d) |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0 | 0 | 0 | 0 | - | - |

## Leaderboard

Value score = 7-day downloads + 3 × unique page visitors (14d).

| # | Project | Shipped | Score | DL 7d | Δ vs last wk | DL 30d | DL total | Page uniques 14d | Trend | Status |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|
| 1 | [ctxlint](projects/ctxlint) | 2026-11-19 | 0 | 0 | 0 | 0 | 0 | 0 |  | parked |
| 2 | [agentscrub](projects/agentscrub) | 2026-11-20 | 0 | 0 | 0 | 0 | 0 | 0 |  | active |
| 3 | [pthscan](projects/pthscan) | 2026-11-21 | 0 | 0 | 0 | 0 | 0 | 0 |  | active |

## Market benchmarks

These are competitors in each project's category. *Day-30 rate* is a package's downloads in days 24–30 after its first publish, which lets the lab compare a new project against established ones at the same age.

### agent-instruction-linters

Median day-30 rate of small entrants (< 5,000/wk now): **38**

| Package | Weekly downloads now | First 7 days | First 30 days | Day-30 rate |
|---|---:|---:|---:|---:|
| agnix | 47,823 | 1,537 | 3,611 | 492 |
| @yawlabs/ctxlint | 41,642 | 1,524 | 5,557 | 538 |
| claude-code-lint | 12,744 | 294 | 1,013 | 324 |
| agentlinter | 4,214 | 2,067 | 2,344 | 64 |
| @promptci/cli | 1,059 | 309 | 562 | 43 |
| @felixgeelhaar/cclint | 327 | 291 | 400 | 14 |
| unrot | 42 | 2,078 | 2,310 | 34 |
| tidyagents | 33 | 732 | 786 | 0 |
| rulecheck | 14 | 547 | 589 | 0 |
| **ctxlint (lab)** _parked_ | 0 | - | - | - |

### agent-guardrail-hooks

Median day-30 rate of small entrants (< 5,000/wk now): **23**

| Package | Weekly downloads now | First 7 days | First 30 days | Day-30 rate |
|---|---:|---:|---:|---:|
| block-no-verify | 49,988 | 369 | 1,229 | 417 |
| @hasna/hooks | 13,496 | 1,325 | 1,413 | 13 |
| ghostimport | 407 | 1,051 | 1,449 | 35 |
| slopcheck | 136 | 522 | 695 | 12 |
| @mharoon07/slopguard | 11 | 443 | 470 | 0 |

### unicode-security

Median day-30 rate of small entrants (< 5,000/wk now): **-**

| Package | Weekly downloads now | First 7 days | First 30 days | Day-30 rate |
|---|---:|---:|---:|---:|
| anti-trojan-source | 46,716 | 365 | 7,504 | 3,655 |
| out-of-character | 8,377 | 193 | 279 | 8 |

### agent-secret-leaks

Median day-30 rate of small entrants (< 5,000/wk now): **-**

| Package | Weekly downloads now | First 7 days | First 30 days | Day-30 rate |
|---|---:|---:|---:|---:|
| secretlint | 2,019,995 | 85 | 2,497 | 879 |
| eslint-plugin-no-secrets | 441,442 | 94 | 123 | 8 |
| gitleaks-secret-scanner | 27,179 | 0 | 352 | 352 |
| @visulima/secret-scanner | 13,087 | 239 | 649 | 306 |
| claude-settings-leak-check | 43 | 358 | 411 | 0 |
| **agentscrub (lab)** _active_ | 0 | - | - | - |

### python-supply-chain

Median day-30 rate of small entrants (< 5,000/wk now): **-**

| Package | Weekly downloads now | First 7 days | First 30 days | Day-30 rate |
|---|---:|---:|---:|---:|
| **pthscan (lab)** _active_ | 0 | - | - | - |

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
| pthscan | Malicious PyPI releases (Microsoft agent-framework 1.26.1/1.26.2 on 17 Nov 2026, LiteLLM 1.82.8, the Hades campaign) plant .pth hooks that run a credential stealer on every Python start. Advice is manual find/grep, which drowns in legitimate hooks and misses payloads in imported modules; no packaged tool checks this. | [1](https://www.stepsecurity.io/blog/microsoft-agent-framework-pypi-package-compromised) [2](https://www.elastic.co/security-labs/threat-command/we-caught-the-agent-framework-pypi-compromise-six-minutes-after-it-published) [3](https://snyk.io/blog/poisoned-security-scanner-backdooring-litellm) [4](https://github.com/jlevy/supply-chain-hardening/blob/main/guidelines/hardening-pypi.md) |
| agentscrub | Claude Code's "don't ask again" saves full approved commands, passwords included, to .claude/settings.local.json; Apiiro found 1,122 public copies holding 1,379 secrets (Nov 2026). MCP configs and session transcripts leak the same way, and generic scanners miss passwords inside commands. | [1](https://thehackernews.com/2026/11/claude-codes-dont-ask-again-file.html?m=1) [2](https://medium.com/@chinmayshringi4/claude-code-dont-ask-again-saved-secrets-in-settings-local-json-d1323c85bb2d) [3](https://www.darkreading.com/cyber-risk/ai-coding-agents-leak-secrets-github) [4](https://www.backslash.security/blog/secrets-in-ai-agent-session-history) |
| ctxlint | Teams run several AI coding agents, each with its own instruction file. The files rot (renamed paths, dead scripts), drift apart, bloat, and can hide malicious invisible Unicode. | [1](https://www.reddit.com/r/codex/comments/1x2wkid/if_you_use_both_codex_and_claude_code_how_do_you) [2](https://umesh-malik.com/blog/agents-md-ai-coding-agents-study) [3](https://www.infoworld.com/article/4224410/claude-code-now-also-accepts-instructions-in-openais-agents-md-format.html) [4](https://finance.yahoo.com/news/vulnerability-github-copilot-cursor-hackers-144000906.html) |
