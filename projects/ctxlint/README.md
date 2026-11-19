# ctxlint

**A linter for the instruction files your AI coding agents read**: `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.cursorrules` and `.cursor/rules/`, plus the instruction files for GitHub Copilot, Windsurf, Cline, Kiro, Junie and Amazon Q.

```bash
npx ctxlint
```

Zero dependencies. Node ≥ 18. Scans VS Code's 39 instruction files in about 0.3s.

## Why

Agent instruction files are code that nobody tests. They rot quietly:

* Someone renames `src/server/` and the agent keeps looking for it.
* `npm run lint` became `pnpm lint`, but the agent was never told.
* `CLAUDE.md` and `AGENTS.md` started as copies and slowly drifted apart.
* A zero-width or bidi Unicode character hides an instruction no reviewer can see. This is the "Rules File Backdoor" attack.
* The file grew to 1,000+ lines, and the agent now ignores half of it.

When we ran ctxlint on popular open-source repos, it found stale file references in VS Code's agent instructions and a 1,161-line `AGENTS.md`.

## What it checks

| Rule | Severity | Catches |
|---|---|---|
| `hidden-unicode` | error | Zero-width, bidi-override, and Unicode tag characters that hide text from reviewers |
| `secret` | error | API keys and tokens (AWS, GitHub, Anthropic, OpenAI, Slack, Stripe, private keys) |
| `broken-import` | error | `@path/to/file.md` imports in `CLAUDE.md` / `GEMINI.md` that don't resolve |
| `missing-script` | error | `npm run x` / `pnpm run x` / `yarn run x` / `bun run x` / `npm test` with no matching script in any `package.json` |
| `missing-make-target` | error | `make deploy` where the Makefile has no `deploy` target |
| `missing-path` | warning | `` `src/foo.ts` `` and `[link](docs/x.md)` that no longer exist (checks ancestor dirs, partial paths, and `.js`→`.ts`) |
| `wrong-package-manager` | warning | `npm install` in a repo that has a `pnpm-lock.yaml` (or yarn/bun) |
| `conflicting-package-managers` | warning | One file says `yarn`, another says `npm` |
| `drift` | warning / info | Near-duplicate instruction files that have diverged, and identical copies that should be an import or symlink |
| `too-long` | warning | Files over 200 lines or ~6k tokens, which are loaded into every session |
| `empty-file` | warning | Empty instruction files |

ctxlint tries hard to avoid false positives. It ignores prose like "make sure", globs, URLs, package specifiers (`@types/node`, `lodash/merge`), workspace-filtered commands, and negated mentions ("do **not** create `src/x.ts`").

## Use in CI

**GitHub Action** (annotates the PR diff):

```yaml
- uses: actions/checkout@v4
- uses: Skymeapple/AI-agents/projects/ctxlint@research-lab
  with:
    strict: false   # true = fail on warnings too
```

**Any CI:**

```bash
npx ctxlint --format github   # or --format json
```

Exit codes: `0` clean, `1` errors (or warnings with `--strict`), `2` usage error.

**pre-commit hook:**

```bash
echo 'npx -y ctxlint' >> .git/hooks/pre-commit
```

## Options

```
ctxlint [dir] [--format text|json|github] [--strict] [--ignore rule,rule]
              [--max-lines 200] [--max-tokens 6000] [--list-rules]
```

To configure a project, add a `.ctxlintrc.json`:

```json
{ "ignore": ["too-long"], "maxLines": 300 }
```

To turn rules off for a single file, add `<!-- ctxlint-disable missing-path, too-long -->`. To turn off every rule for that file, use `<!-- ctxlint-disable -->`.

## Programmatic API

```js
import { lint } from 'ctxlint';
const { files, issues } = lint('.', { maxLines: 300 });
```

## Files discovered

`AGENTS.md`, `AGENTS.override.md`, `CLAUDE.md`, `CLAUDE.local.md` and `GEMINI.md` are found at any depth. These locations are also scanned:

* `.claude/CLAUDE.md` and `.claude/rules/`
* `.gemini/GEMINI.md`
* `.cursorrules` and `.cursor/rules/*.mdc`
* `.windsurfrules` and `.windsurf/rules/`
* `.clinerules`
* `.github/copilot-instructions.md` and `.github/instructions/*.instructions.md`
* `.junie/guidelines.md`
* `.amazonq/rules/`
* `.kiro/steering/`
* `.agents/rules/`

`node_modules`, `dist`, `vendor` and similar directories are skipped.

---

Built by the [AI Research Lab](https://github.com/Skymeapple/AI-agents/tree/research-lab), which ships one tool a day for a real developer pain point. Issues and stars help us decide what to build next.
