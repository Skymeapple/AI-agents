# Changelog

## 0.2.0 (2026-11-23)

* `--home`: also scan user-level agent configs: `~/.claude.json` (per-project MCP servers), `~/.claude/settings*.json`, Claude Desktop `claude_desktop_config.json` (macOS, Linux, Windows), `~/.cursor/mcp.json`, Windsurf `mcp_config.json`, `~/.gemini/settings.json`, `~/.codex/config.toml` and `~/.aider.conf.yml`. People paste API keys into these, dotfiles repos commit them, and credential stealers harvest them.
* Files outside the project are shown as `~/…`.
* Tests clean up their temporary fixtures.

## 0.1.0 (2026-11-20)

First release.

* Scans Claude Code settings (`permissions.allow/ask/deny`, `env`), MCP configs (`.mcp.json`, `.cursor/`, `.vscode/`, `.roo/`, `.kiro/`, `.amazonq/`), `.gemini/`, `.codex/`, `.specstory/` and aider histories.
* Detects 14 key formats plus credentials inside commands: URL passwords, `PGPASSWORD=`-style assignments, `--password`, `mysql -p`, `sshpass -p`, `curl -u`, and auth and API-key headers.
* Grades exposure using git: local, at-risk, committed, or committed with a remote.
* `--fix` removes leaked permission entries and git-ignores `.claude/settings.local.json`.
* Guards: a Claude Code PreToolUse hook (blocks `git commit`/`push`/`add`), a Stop hook with `--scrub`, a pre-commit `--staged` mode, `init`, and a GitHub Action.
* `--sessions` scans `~/.claude/projects` and `~/.codex/sessions` transcripts.
* Output formats: text, json and github annotations. Secret values are never printed.
