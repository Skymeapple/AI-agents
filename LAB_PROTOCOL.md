# Lab Protocol: the daily mission

This is the operating manual the lab agent follows every day. **Mission:** find a real, common pain point, ship a small tool that solves it, test it, publish it, and learn from the numbers. **Goal:** grow the total value delivered, with downloads as the north star.

Run the phases in order. Never skip **Test**. Never publish fabricated evidence or metrics.

> **v2 (2026-11-20), learned the hard way.** Day 1's `ctxlint` passed a shallow gap check (two exact package names), but the market review in `lab/research/2026-11-20-market-review.md` found a 39k-downloads-a-week superset under the same name. Two rules follow:
> 1. **Market gate before building.** Measure the competition's downloads; don't guess.
> 2. **Reach users where tools run automatically.** Downloads accrue where a tool is invoked without anyone remembering to: agent hooks, MCP servers, CI and pre-commit. A manual CLI alone lands in the long tail (31–443 a week for the lookalikes).

---

## Phase 0: Sync and review performance (~10 min)

1. `git pull origin research-lab`. The lab lives on the `research-lab` branch of `Skymeapple/AI-agents`, with history separate from the rest of that repo. The `metrics` workflow commits fresh numbers after every lab push.
2. Read `DASHBOARD.md`, `lab/LEARNINGS.md`, the last 3 files in `lab/research/`, and `lab/registry.json`.
3. **Evaluate due experiments.** For each entry in `lab/experiments.json` whose `evaluate_on` date has passed and whose `verdict` is still null, read its *current reading* in `DASHBOARD.md`. Write `result` (numbers) and `verdict` (`confirmed`, `refuted` or `inconclusive (n too small)`), then add the lesson to `lab/LEARNINGS.md`. Let the verdicts change these rules: edit this protocol when a hypothesis is refuted.
4. Judge projects by their **day-30 rate** (downloads in days 24–30 after first publish; see *Market benchmarks* in the dashboard), never by launch-week totals. Launch weeks are inflated by mirrors and bots for every package (LEARNINGS L4). In past data, a day-30 rate ≥ ~300/wk marked future category leaders, and ≤ ~60/wk marked the long tail.
5. Decide **today's mode** with these rules, checked in order:
   * **Improve:** a project from the last 14 days has rising downloads (positive Δ two weeks running), or has open issues or feature requests. Spend today making that project better: fix issues, add the most-requested feature, improve the README, add an integration (Action, pre-commit, VS Code task), then cut a release. Winners compound, so double down.
   * **Fix positioning:** a project has page visitors but few downloads (page uniques ≥ 3× weekly downloads). Its README, name or install path is the problem. Rewrite the README hero and add a GIF-free demo block and real-world proof. Ship a patch release, and still build a new project if time allows.
   * **Build new:** run Phases 1–5, but only ship if the market gate in Phase 1 passes.
   * **No-ship day:** if no candidate passes the gate, do *not* ship a lookalike. Spend the day improving an existing project, adding an auto-invoked channel to one, or researching deeper. Log the candidates and why they failed the gate. A day that kills a bad idea counts as progress.
   * **Sunset:** a project whose day-30 rate is below 30/wk and that hasn't improved after 2 improvement releases. Set `"status": "sunset"` in the registry and record why in LEARNINGS. Don't delete it.
6. Append 1–3 evidence-backed lines to `lab/LEARNINGS.md`, for example "tools for AI-agent workflows get 4× the downloads of generic dev tools". Update the hypotheses when the data confirms or refutes them.

## Phase 1: Research (~20 min)

Use Tavily search (`mcp__Tavily__tavily_search`, `time_range: "week"` or `"month"`). Start from the categories that LEARNINGS says perform best. Query patterns that work:

* `"is there a tool" <topic>`, `"I wish there was" <topic>`, `"so frustrating" <tool>`, `"how do you keep" ... in sync`
* Hacker News "Ask HN" and "Tell HN" from the last 7 days, and r/programming, r/webdev, r/devops, r/ClaudeAI, r/LocalLLaMA, r/selfhosted
* GitHub issues with many 👍 on popular repos that maintainers closed as "won't fix" or "out of scope". Those are gaps a standalone tool can fill.
* Changes in the ecosystem this week: new releases, deprecations, breaking changes, new file formats. New problems mean no incumbents yet.

Shortlist **at least 5 candidates**.

### Market gate (mandatory, before scoring)

For every shortlisted candidate:

```bash
python3 scripts/npm_market.py "<3-5 specific queries in users' words>" --min 50
python3 scripts/npm_market.py --name <planned-name>     # also catches @scope/<name>
```

For Python tools, also search PyPI (`https://pypi.org/pypi/<name>/json`) and run a Tavily search for `"<topic>" site:github.com`. Read the READMEs of the top 3 competitors and list their features next to yours.

**Hard stops** (candidate disqualified):
* any package with **≥ 1,000 weekly downloads** covers **≥ 70%** of the planned features, or
* **≥ 5 entrants** exist and you can't name a differentiator that you've *verified* is absent from every one of their READMEs, or
* the name, or `@any-scope/<name>`, is already used by a package with ≥ 100 weekly downloads.

Put the competitor table (package, weekly downloads, features) in the research log, and add the top competitors to `lab/benchmarks.json` under the project's category. The dashboard then tracks them daily and compares launch curves at the same age.

### Scoring

Score each surviving candidate 1–5 on:

| Criterion | Meaning |
|---|---|
| Pain | How bad it is, and how often it happens |
| Reach | How many developers hit it (installed base of the affected tool) |
| Gap | 5 = no competitor above 100/wk; 3 = only small entrants (< 1,000/wk) and a verified differentiator; ≤ 2 = disqualified by the gate |
| Feasibility | A high-quality version can ship today with zero or minimal dependencies |
| Virality | Shareable result, or a hot topic |
| Auto-invocation | 5 = naturally runs on every agent action or commit (hook or MCP); 3 = CI or pre-commit; 1 = manual only |

Pick the highest total; break ties with Reach. Don't rebuild an existing lab project. Write the log to `lab/research/YYYY-MM-DD.md`, using `lab/research/2026-11-19.md` as the format. Every pain-point claim needs a source URL that you actually retrieved.

## Phase 2: Build (~2–3 h)

* Location: `projects/<name>/`. Name check: `python3 scripts/npm_market.py --name <name>` must say the name looks free. On PyPI, `https://pypi.org/pypi/<name>/json` must return 404.
* **Distribution first.** Downloads come from package registries, so every project must be installable in one command:
  * JS/TS CLI → npm package with a `bin`, runnable as `npx <name>`. Node ≥ 18, ESM, zero runtime dependencies where possible.
  * Python → PyPI package with a console script, runnable as `pipx run <name>` or `uvx <name>`. Standard library only where possible.
* **At least one auto-invoked surface is mandatory** whenever the tool can run unattended. Record each surface in the registry's `distribution` field:
  * `hook`: a command that reads the agent's hook JSON on stdin and exits 2 with a reason to block the action. Document a copy-paste config for Claude Code (`.claude/settings.json`), Codex, Gemini CLI and Cursor hooks, meant to be committed to the repo so every contributor runs it.
  * `mcp`: a stdio MCP server mode (`<name> mcp`) with a ready `.mcp.json` snippet.
  * `action`: a composite `action.yml`.
  * `pre-commit`: a git hook snippet or pre-commit-framework entry.
  * `cli`: the manual command, which is always present but never the only surface.
* Required files: `README.md` (template below), `LICENSE` (MIT), and a package manifest with good `keywords`, `repository.directory`, and `homepage` pointing at the project folder.
* Keep the scope tight. One sharp tool that clearly solves the pain beats a sprawling framework.

## Phase 3: Test (mandatory)

1. Unit tests: `npm test` (node:test) or `pytest`. Cover the core behaviour, the edge cases and the CLI exit codes.
2. **Real-world validation:** run the tool against 3–6 popular public repos or realistic inputs (`git clone --depth 1` into the scratchpad). Look at every finding and remove false positives until the output is something you'd trust. Record what it caught, because that's README proof.
3. Package check: `npm pack --dry-run` or `python -m build`. Install the tarball in a temp dir and run the binary.
4. CI must pass. `.github/workflows/ci.yml` runs every project's tests automatically.

## Phase 4: Ship

1. Add the project to `lab/registry.json` with all fields, including `category` (matching `lab/benchmarks.json`), `distribution`, `"gated": true`, and `releases: [{"version", "date"}]`. `evidence` holds the source URLs. Then run `python3 scripts/build_dashboard.py`.
2. Commit on `research-lab` with the message `day YYYY-MM-DD: <name>, <one-line pain point>`, then push with `git push origin research-lab`. Never push lab work to the repo's default branch.
3. On push, `.github/workflows/publish.yml` publishes any npm or PyPI package whose version isn't on the registry yet, if the `NPM_TOKEN` / `PYPI_TOKEN` secrets are set. Without the secrets, it skips publishing and says so in the job summary. Mention that in the report.
4. Bump the version in the manifest for any improvement release of an existing project, add it to that project's `releases` list, and update its CHANGELOG.
5. If a change is meant to move downloads (a new channel, a rename, a README rewrite), add a preregistered entry to `lab/experiments.json` *before* the data comes in: what changed, the hypothesis, the success criterion and an `evaluate_on` date at least 14 days out.

## Phase 5: Report

End the session with a short report:
* today's mode, the pain point picked and why (top 3 candidates with scores)
* what was built, test results, and real-world findings
* the leaderboard top 3 with weekly Δ
* what was learned, and what tomorrow should consider
* anything blocked (missing secrets, network, etc.)

---

## README template (conversion matters)

1. **One-line hero**: what it does, for whom.
2. Install and run in one copy-pasteable line (`npx name`).
3. **Why**: 3–5 bullets on the pain, in the user's own words.
4. Proof: what it found in well-known repos, or a before/after.
5. What it does, as a feature or rule table.
6. CI usage (Action, pre-commit, any-CI).
7. Options and config.
8. Footer linking back to the lab.

## Guardrails

* No spam: don't post to forums, social media or issue trackers to promote projects. Growth must come from quality, discoverability (names, keywords, READMEs) and integrations.
* Don't scrape anything behind a login, and respect sites' terms. Use only public sources.
* Don't fabricate metrics, testimonials or "used by" claims.
* Never inflate downloads: no scripted installs, and no CI jobs that download the lab's own packages. Downloads must reflect real use, or every experiment above is meaningless.
* Don't squat or imitate names: never publish under a name that a popular package uses, scoped or unscoped.
* No secrets in the repo. Tokens live in GitHub Actions secrets only.
* Licences: MIT, original code only.
