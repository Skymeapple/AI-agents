# Lab Protocol: the daily mission

This is the operating manual the lab agent follows every day. **Mission:** find a real, common pain point, ship a small tool that solves it, test it, publish it, and learn from the numbers. **Goal:** grow the total value delivered, with downloads as the north star.

Run the phases in order. Never skip **Test**. Never publish fabricated evidence or metrics.

---

## Phase 0: Sync and review performance (~10 min)

1. `git pull`. The `metrics` workflow commits fresh numbers every morning.
2. Read `DASHBOARD.md`, `lab/LEARNINGS.md`, the last 3 files in `lab/research/`, and `lab/registry.json`.
3. Decide **today's mode** with these rules:
   * **Improve:** a project from the last 14 days has rising downloads (positive Δ two weeks running), or has open issues or feature requests. Spend today making that project better: fix issues, add the most-requested feature, improve the README, add an integration (Action, pre-commit, VS Code task), then cut a release. Winners compound, so double down.
   * **Fix positioning:** a project has page visitors but few downloads (page uniques ≥ 3× weekly downloads). Its README, name or install path is the problem. Rewrite the README hero and add a GIF-free demo block and real-world proof. Ship a patch release, and still build a new project if time allows.
   * **Build new** (the default): run Phases 1–5.
   * **Sunset:** a project over 30 days old with fewer than 5 weekly downloads. Set `"status": "sunset"` in the registry and record why in LEARNINGS. Don't delete it.
4. Append 1–3 evidence-backed lines to `lab/LEARNINGS.md`, for example "tools for AI-agent workflows get 4× the downloads of generic dev tools". Update the hypotheses when the data confirms or refutes them.

## Phase 1: Research (~20 min)

Use Tavily search (`mcp__Tavily__tavily_search`, `time_range: "week"` or `"month"`). Start from the categories that LEARNINGS says perform best. Query patterns that work:

* `"is there a tool" <topic>`, `"I wish there was" <topic>`, `"so frustrating" <tool>`, `"how do you keep" ... in sync`
* Hacker News "Ask HN" and "Tell HN" from the last 7 days, and r/programming, r/webdev, r/devops, r/ClaudeAI, r/LocalLLaMA, r/selfhosted
* GitHub issues with many 👍 on popular repos that maintainers closed as "won't fix" or "out of scope". Those are gaps a standalone tool can fill.
* Changes in the ecosystem this week: new releases, deprecations, breaking changes, new file formats. New problems mean no incumbents yet.

Shortlist **at least 5 candidates** and score each 1–5 on:

| Criterion | Meaning |
|---|---|
| Pain | How bad it is, and how often it happens |
| Reach | How many developers hit it (installed base of the affected tool) |
| Gap | No good existing solution. **Check npm, PyPI and GitHub before you score.** |
| Feasibility | A high-quality version can ship today with zero or minimal dependencies |
| Virality | Shareable result, a CI integration, or a hot topic |

Pick the highest total; break ties with Reach. Don't rebuild an existing lab project. Write the log to `lab/research/YYYY-MM-DD.md`, using `lab/research/2026-11-19.md` as the format. Every pain-point claim needs a source URL that you actually retrieved.

## Phase 2: Build (~2–3 h)

* Location: `projects/<name>/`. Before naming, check the name is free: `curl -s -o /dev/null -w '%{http_code}' https://registry.npmjs.org/<name>` should return 404 (PyPI: `https://pypi.org/pypi/<name>/json`).
* **Distribution first.** Downloads come from package registries, so every project must be installable in one command:
  * JS/TS CLI → npm package with a `bin`, runnable as `npx <name>`. Node ≥ 18, ESM, zero runtime dependencies where possible.
  * Python → PyPI package with a console script, runnable as `pipx run <name>` or `uvx <name>`. Standard library only where possible.
  * Add a `action.yml` (composite GitHub Action) whenever the tool fits CI.
* Required files: `README.md` (template below), `LICENSE` (MIT), and a package manifest with good `keywords`, `repository.directory`, and `homepage` pointing at the project folder.
* Keep the scope tight. One sharp tool that clearly solves the pain beats a sprawling framework.

## Phase 3: Test (mandatory)

1. Unit tests: `npm test` (node:test) or `pytest`. Cover the core behaviour, the edge cases and the CLI exit codes.
2. **Real-world validation:** run the tool against 3–6 popular public repos or realistic inputs (`git clone --depth 1` into the scratchpad). Look at every finding and remove false positives until the output is something you'd trust. Record what it caught, because that's README proof.
3. Package check: `npm pack --dry-run` or `python -m build`. Install the tarball in a temp dir and run the binary.
4. CI must pass. `.github/workflows/ci.yml` runs every project's tests automatically.

## Phase 4: Ship

1. Add the project to `lab/registry.json` (all fields; `evidence` holds the source URLs), then run `python3 scripts/build_dashboard.py`.
2. Commit on `main` with the message `day YYYY-MM-DD: <name>, <one-line pain point>`, then push.
3. On push, `.github/workflows/publish.yml` publishes any npm or PyPI package whose version isn't on the registry yet, if the `NPM_TOKEN` / `PYPI_TOKEN` secrets are set. Without the secrets, it skips publishing and says so in the job summary. Mention that in the report.
4. Bump the version in the manifest for any improvement release of an existing project.

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
* No secrets in the repo. Tokens live in GitHub Actions secrets only.
* Licences: MIT, original code only.
