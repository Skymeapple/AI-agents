# agentscrub

**Find and remove the secrets your AI coding agent saved to disk, and stop it from committing them.**

```bash
npx agentscrub          # scan this project
npx agentscrub --fix    # remove leaked entries, git-ignore the local settings file
npx agentscrub init     # install the guard hook + pre-commit check for the whole team
npx agentscrub --home   # also check ~/.claude.json, Claude Desktop, Cursor, Windsurf, Gemini user configs
```

It has zero dependencies, needs Node ≥ 18, makes no network calls, and never prints a secret value.

## Why

When you answer **"Yes, and don't ask again"**, Claude Code saves the *entire approved command* in `.claude/settings.local.json`. If you typed a password into it (`psql postgres://app:hunter2@db…`, `PGPASSWORD=… psql`, `curl -H "Authorization: Bearer …"`), the password is now in a file that looks like harmless config.

* In November 2026, Apiiro researchers found **104,185 copies** of that file on public GitHub. **1,122** of them held real secrets, **1,379 distinct credentials** in all ([The Hacker News](https://thehackernews.com/2026/11/claude-codes-dont-ask-again-file.html), [write-up](https://medium.com/@chinmayshringi4/claude-code-dont-ask-again-saved-secrets-in-settings-local-json-d1323c85bb2d)).
* In **539** of the 1,105 repos with a real secret that they traced, a commit **co-authored by Claude** had changed that file. Agents commit these files too, not just people.
* MCP configs (`.mcp.json`) leak the same way, with API keys pasted into `env` blocks. So do session transcripts, where most secrets get in through the agent's own work, such as reading a `.env` file ([Backslash](https://www.backslash.security/blog/secrets-in-ai-agent-session-history)).
* Generic secret scanners match known key formats. They miss **passwords inside commands**, which is exactly what agent allowlists are full of.

## What it looks like

```
$ npx agentscrub

COMMITTED  in git history: rotate, then purge history before any push
  .claude/settings.local.json:5 permissions.allow[1]  Password in URL hu…(20 chars)  url-credentials
  .claude/settings.local.json:6 permissions.allow[2]  Authorization header hu…(40 chars)  auth-header
  .mcp.json:10 mcpServers.postgres.env.DATABASE_PASSWORD  Literal value for DATABASE_PASSWORD hu…(20 chars)  named-secret

3 secret(s) in 2 file(s). 2 can be removed with agentscrub --fix.

$ npx agentscrub --fix
fixed .claude/settings.local.json: removed 2 permission entries holding secrets
fixed .gitignore: added .claude/settings.local.json
todo  .claude/settings.local.json: still tracked by git: run `git rm --cached .claude/settings.local.json`, then rotate any secret it ever held
```

Each finding is graded by how far it may have spread:

| Grade | Meaning |
|---|---|
| `COMMITTED+REMOTE` | In git history, and the repo has a remote. Assume it's public and rotate now. |
| `COMMITTED` | In git history. Rotate, then purge history before pushing. |
| `AT-RISK` | Not git-ignored. The next `git add .` publishes it. |
| `LOCAL` | Git-ignored and never committed, but still plain text on disk. |

Tested on real projects that commit agent configs, with **zero false positives** across 71 agent files: getsentry/sentry, PostHog/posthog, modelcontextprotocol/servers, openai/codex, google-gemini/gemini-cli, vercel/ai, anthropics/claude-code-action, astral-sh/uv and microsoft/vscode.

## Stop it happening again (runs automatically)

`npx agentscrub init` sets up two guards. Commit `.claude/settings.json` so they run for everyone on the project.

**1. Claude Code hook.** Blocks the agent's `git commit` / `git push` / `git add` while a file git would publish holds a secret. The agent is told why it was stopped and how to fix it.

```json
{
  "hooks": {
    "PreToolUse": [
      { "matcher": "Bash", "hooks": [{ "type": "command", "command": "npx -y agentscrub hook" }] }
    ]
  }
}
```

To also scrub leaked allowlist entries automatically at the end of every turn, add a `Stop` hook:

```json
"Stop": [{ "hooks": [{ "type": "command", "command": "npx -y agentscrub hook --scrub" }] }]
```

The hook fails open on internal errors. If you want it to fail closed, set `"onFailure": "block"` on it (Claude Code ≥ 2.1.295).

**2. git pre-commit.** Works for every agent (Cursor, Codex, Gemini CLI, aider) and for humans:

```sh
npx -y agentscrub --staged || exit 1
```

`--staged` checks what's actually being committed (the git index), not your working copy.

**CI** (GitHub Action, annotates the PR):

```yaml
- uses: actions/checkout@v4
- uses: Skymeapple/AI-agents/projects/agentscrub@research-lab
```

Or in any CI: `npx -y agentscrub --format github`.

## What it scans

| Agent | Files |
|---|---|
| Claude Code | `.claude/settings.local.json`, `.claude/settings.json` (`permissions.allow` / `ask` / `deny`, `env`) |
| MCP (any client) | `.mcp.json`, `.cursor/mcp.json`, `.vscode/mcp.json`, `.roo/mcp.json`, `.kiro/settings/mcp.json`, `.amazonq/mcp.json` |
| Gemini CLI / Codex | `.gemini/`, `.codex/` |
| SpecStory / aider | `.specstory/`, `.aider.chat.history.md`, `.aider.input.history` |
| User-level configs (`--home`) | `~/.claude.json`, `~/.claude/settings*.json`, Claude Desktop `claude_desktop_config.json`, `~/.cursor/mcp.json`, Windsurf `mcp_config.json`, `~/.gemini/settings.json`, `~/.codex/config.toml` |
| Session transcripts (`--sessions`) | `~/.claude/projects/**/*.jsonl`, `~/.codex/sessions/**` |

Nested projects (monorepos) are found up to `--depth 4`.

**Detected:**
* **Key formats:** AWS, GitHub, GitLab, Anthropic, OpenAI, Slack, Stripe, Google, npm, Hugging Face, SendGrid, JWTs (such as Supabase service-role keys) and private keys.
* **Credentials in commands:** `scheme://user:pass@host`; `PGPASSWORD=…` / `*_TOKEN=…` / `*_API_KEY=…`; `--password=…`; `mysql -p…`; `sshpass -p`; `curl -u user:pass`; `Authorization:` and `X-API-Key:` headers.
* **MCP env blocks:** literal values under secret-named keys.

**Not flagged:** references such as `$TOKEN`, `${VAR}` and `${env:VAR}`; placeholders (`<password>`, `your-key-here`, `REDACTED`); well-known local defaults (`POSTGRES_PASSWORD=postgres`); and variable *names* (`bearer_token_env_var: "MY_TOKEN"`).

## Options

```
agentscrub [dir] [--fix] [--staged] [--sessions [dir]] [--format text|json|github] [--depth 4]
agentscrub init [dir]
agentscrub hook [--scrub]
```

Exit codes: `0` clean, `1` secrets found, `2` usage error. In hook mode, `2` blocks the tool call.

## If it finds something

1. **Rotate the credential first.** Removing it doesn't un-leak a secret that was pushed, cloned or scraped.
2. Run `agentscrub --fix` and `git rm --cached` any agent file that shouldn't be tracked.
3. If it was ever pushed, purge it from history (`git filter-repo`) *after* rotating.
4. From now on, pass secrets as environment variables (`$PGPASSWORD`), so the saved command holds a reference rather than the value.

## How it compares

| | agentscrub | [claude-settings-leak-check](https://www.npmjs.com/package/claude-settings-leak-check) | gitleaks / secretlint |
|---|:-:|:-:|:-:|
| Claude Code allowlists | ✓ | ✓ | partly (key formats only) |
| Passwords inside commands | ✓ | ✓ | ✗ mostly |
| MCP configs, Gemini, Codex, SpecStory, aider | ✓ | ✗ | if configured |
| Session transcripts | ✓ | ✗ | ✗ |
| Auto-fix (`--fix`) | ✓ | ✗ | ✗ |
| Agent hook that blocks the commit | ✓ | ✗ | ✗ |
| pre-commit `--staged` mode | ✓ | ✗ | ✓ |
| Exposure grading using git | ✓ | ✓ | ✗ |

Pattern matching can't catch everything, so a clean scan isn't proof that no secrets exist. Treat findings as leads to check.

---

Built by the [AI Research Lab](https://github.com/Skymeapple/AI-agents/tree/research-lab). Found a leak shape it misses? Open an issue describing the *shape*, never the secret.
