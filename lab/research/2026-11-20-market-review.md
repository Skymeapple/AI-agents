# Market review: 2026-11-20

**Question:** how does Day 1's `ctxlint` compare with the projects that have high download numbers, and what would raise the value and downloads of what the lab ships?

**Method:** npm registry search API (`registry.npmjs.org/-/v1/search`, which reports weekly and monthly downloads), full package metadata (version history, dependencies, READMEs) for the top competitors, and README feature comparisons. Download figures are npm's weekly counts as reported on 2026-11-19/20. They include CI and mirror traffic, so they measure relative reach, not exact user counts.

## 1. How `ctxlint` compares

### Direct competitors (AI agent instruction/config linters)

| Package | Weekly downloads | First published | Versions | Notes |
|---|---:|---|---:|---|
| `agnix` | 42,530 | 2026-02-05 | 102 | Lints SKILL.md, CLAUDE.md, hooks and MCP configs; auto-fix, SARIF, LSP / VS Code |
| `@yawlabs/ctxlint` | 39,400 | 2026-04-05 | 82 | **Same name.** Broken paths, wrong commands (npm/Make/Gradle/Maven), contradictions, frontmatter, token budget, CI coverage, inline secrets, MCP-config and session linting, `--fix`, SARIF, pre-commit, Action. All its docs use `npx …@latest`. |
| `claude-code-lint` | 14,038 | 2026-01-31 | 20 | Claude Code plugin plus CLI |
| `agentlinter` | 3,963 | 2026-02-05 | 20 | Score plus shareable score |
| `@promptci/cli` | 2,176 | 2026-08-17 | 12 | |
| `tidyagents` | 443 | 2026-11-06 | 4 | Lint, score and tidy; badge; "big-name repos" proof |
| `@felixgeelhaar/cclint` | 418 | | | Drift between CLAUDE.md and AGENTS.md |
| `unrot` | 42 | 2026-07-30 | | |
| `rulecheck` | 31 | 2026-11-03 | 3 | Almost the same pitch as ours: stale paths, broken scripts, precision over recall |

Five more scoped packages are *also* named `ctxlint`: `@ctxlint/ctxlint` (234/wk), `@veriova/ctxlint` (399/wk), and three others.

### Feature overlap

| ctxlint rule | Already covered by |
|---|---|
| missing-path, missing-script, missing-make-target | `@yawlabs/ctxlint` (broken paths, wrong commands), `rulecheck`, `tidyagents` |
| broken-import, too-long | `claude-code-lint`, `tidyagents`, `@yawlabs/ctxlint` (token budget) |
| drift | `@yawlabs/ctxlint` (contradictions, redundancy), `cclint` |
| secret | `@yawlabs/ctxlint`, `agentlinter`, `rulecheck` |
| hidden-unicode | `anti-trojan-source` (40,758/wk) works on any file, Markdown included |
| wrong-package-manager | `only-allow` (2,280,119/wk) blocks the wrong package manager at install time |

**Verdict:** `ctxlint` has no feature that someone isn't already offering, and its name collides with a 39k-a-week tool and five scoped clones. Published as is, it would at best join the long tail (31–443 a week) and would confuse users of `@yawlabs/ctxlint`. **Decision: park it and don't publish.** It's marked `private` in package.json, and its README points to the better tools.

**Root cause:** Day 1's gap check looked up two exact package names. It never searched the category or looked at download counts.

## 2. What drives high downloads in this ecosystem

| Pattern | Evidence |
|---|---|
| **Automatic invocation beats a manual CLI** | `block-no-verify` gets **55,877/wk**. It's a single-purpose guard that teams commit to their agent hook config, so it runs on every agent tool call on every contributor's machine. Its "Used by" list includes Dify (133k★), promptfoo and RxDB. `@yawlabs/ctxlint` is documented as `npx @yawlabs/ctxlint@latest`, which goes to the registry on every run. Manual-CLI lookalikes sit at 31–443/wk. |
| **Release cadence** | The leaders have 82–102 versions in 7–10 months. Small entrants have 3–20. |
| **Breadth across the agent-config surface** | The leaders lint MCP configs, skills, hooks and sessions, not just one file type. |
| **Many channels** | Leaders ship a CLI plus an Action, pre-commit hook, SARIF, LSP or VS Code extension, a Claude Code plugin and an MCP mode. |
| **First mover wins** | `agnix` and `@yawlabs/ctxlint` launched Feb–Apr 2026. Entrants from Oct–Nov 2026 (`rulecheck`, `tidyagents`) haven't caught up, even with similar features. |
| **Saturation** | Building is cheap now, so every AI-dev-tool niche we checked had 5–10 entrants: instruction linters, slopsquatting guards (`ghostimport` 942/wk, `rxray` 1,295, `slopguard` 436, `slopcheck` 168) and Claude Code hook kits. |

## 3. Options considered for raising value

| Option | Assessment |
|---|---|
| Publish `ctxlint` anyway, under a new name | ✗ It would be a weaker copy of a 39k-a-week tool. |
| Contribute `ctxlint`'s two lightly covered checks (hidden Unicode in context files, package manager vs lockfile) upstream to `@yawlabs/ctxlint` | Real value for users, but not lab downloads. Still a reasonable good-citizen move. |
| Pivot to an agent guardrail hook, for example against slopsquatting | ✗ The gate fails: `ghostimport` already has Claude and Cursor hooks, typo distance, age and download signals. |
| Pivot to a package-manager guard hook | ✗ `only-allow` has 2.28M/wk. |
| **Fix the lab's process** | ✓ This is where the real gain is. The lab's ceiling is set by what it chooses to build and how its tools reach users, not by build quality. |

## 4. Changes made (protocol v2)

1. **Market gate** in Phase 1, using the new `scripts/npm_market.py`, which shows weekly downloads by query and checks scoped name collisions. Hard stops: a package with ≥ 1,000/wk covering ≥ 70% of features; ≥ 5 entrants with no verified differentiator; a name used by a package with ≥ 100/wk.
2. **Auto-invoked surface is mandatory** (hook, MCP, Action or pre-commit), plus a new *Auto-invocation* scoring criterion.
3. **No-ship days are allowed.** Killing a bad idea beats shipping long-tail noise.
4. **Improve-first.** Projects get patch releases, which are recorded in the registry.
5. **Benchmark tracking.** `lab/benchmarks.json` lists the competitors in each category. The daily metrics job records their weekly downloads and their **launch curves** (downloads in their first 7 and 30 days), so each lab project can be compared with them at the same age.
6. **Preregistered experiments** (`lab/experiments.json`), evaluated automatically on their dates:
   * **E1:** gated projects reach the category's small-entrant median at day 30.
   * **E2:** auto-invoked channels get at least 3× the median downloads of cli-only.
   * **E3:** projects with ≥ 3 releases in their first 30 days outperform those with one.
7. **Integrity guardrails:** no download inflation and no name squatting.

## 5. How this will be validated

* `DASHBOARD.md` now has *Market benchmarks*, *Downloads by distribution channel* and *Experiments* sections, refreshed on every lab push.
* E1 and E2 are evaluated on 2026-12-20 and E3 on 2026-12-31 by the daily mission (Phase 0, step 3). Verdicts go to `lab/LEARNINGS.md`, and refuted hypotheses change the protocol.
* Caveat: with one new project a day at most, the samples by mid-December will be small (n ≈ 5–20). The protocol allows an "inconclusive" verdict rather than over-reading noise.
