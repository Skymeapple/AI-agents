# Changelog

## 0.1.0 (2026-11-19)

First release.

* Discovers instruction files for AGENTS.md, Claude Code, Gemini CLI, Cursor, GitHub Copilot, Windsurf, Cline, Kiro, Junie and Amazon Q.
* Rules: `hidden-unicode`, `secret`, `broken-import`, `missing-script`, `missing-make-target`, `missing-path`, `wrong-package-manager`, `conflicting-package-managers`, `drift`, `too-long`, `empty-file`.
* Output formats: `text`, `json` and `github` (annotations), plus a composite GitHub Action.
* Tested against openai/codex, google-gemini/gemini-cli, vercel/ai, microsoft/vscode, astral-sh/uv and anthropics/claude-code-action, with false positives fixed.
