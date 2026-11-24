#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { scan, fix, scanSessions, sessionDirs } from '../src/scan.js';

const HELP = `agent-commit-guard - find and remove secrets that AI coding agents save to disk

Coding agents store approved shell commands ("Yes, and don't ask again"),
MCP server env blocks and session transcripts in plain files. Passwords and
API keys typed into those commands end up there, and then in git.

Usage:
  agent-commit-guard [dir]              scan agent files in a project (default: .)
  agent-commit-guard --fix              remove leaked permission entries, git-ignore
                                .claude/settings.local.json
  agent-commit-guard --staged           pre-commit mode: scan staged agent files only
  agent-commit-guard --home             also scan user-level configs (~/.claude.json,
                                Claude Desktop, ~/.cursor, Windsurf, ~/.gemini)
  agent-commit-guard --sessions [dir]   also scan session transcripts
                                (~/.claude/projects, ~/.codex/sessions)
  agent-commit-guard init               add the Claude Code guard hook to
                                .claude/settings.json and a git pre-commit hook
  agent-commit-guard hook [--scrub]     run as a Claude Code hook (reads JSON on stdin)

Options:
  --format text|json|github     output format (default text)
  --depth <n>                   how deep to look for nested projects (default 4)
  -v, --version                 print version
  -h, --help                    show this help

Exit codes: 0 no secrets, 1 secrets found, 2 usage error (hook: 2 = block).
Secret values are never printed, only a masked prefix and the length.
`;

const VERSION = JSON.parse(fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'package.json'), 'utf8')).version;
const useColor = process.stdout.isTTY && !process.env.NO_COLOR;
const c = (code, s) => (useColor ? `\x1b[${code}m${s}\x1b[0m` : s);
const GRADE_TEXT = {
  'committed+remote': c('31;1', 'COMMITTED+REMOTE'),
  committed: c('31', 'COMMITTED'),
  'at-risk': c('33', 'AT-RISK'),
  local: c('36', 'LOCAL'),
  'no-git': c('36', 'NO-GIT'),
  session: c('35', 'SESSION'),
};
const GRADE_HELP = {
  'committed+remote': 'in git history and the repo has a remote: assume it is public, rotate now',
  committed: 'in git history: rotate, then purge history before any push',
  'at-risk': 'not git-ignored: the next `git add .` publishes it',
  local: 'git-ignored and never committed, but still plain text on disk',
  'no-git': 'not in a git repo',
  session: 'agent session transcript on this machine',
};

function die(msg) { process.stderr.write(`agent-commit-guard: ${msg}\n`); process.exit(2); }

function parse(argv) {
  const o = { cmd: 'scan', dir: '.', format: 'text', depth: 4 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => { const v = argv[++i]; if (v === undefined) die(`missing value for ${a}`); return v; };
    if (a === '-h' || a === '--help') o.help = true;
    else if (a === '-v' || a === '--version') o.version = true;
    else if (a === 'hook' && i === 0) o.cmd = 'hook';
    else if (a === 'init' && i === 0) o.cmd = 'init';
    else if (a === '--fix') o.fix = true;
    else if (a === '--scrub') o.scrub = true;
    else if (a === '--staged') o.staged = true;
    else if (a === '--home') o.home = true;
    else if (a === '--sessions') { o.sessions = []; if (argv[i + 1] && !argv[i + 1].startsWith('-')) o.sessions.push(argv[++i]); }
    else if (a === '--format') o.format = next();
    else if (a === '--depth') o.depth = Number(next());
    else if (a.startsWith('-')) die(`unknown option ${a}`);
    else o.dir = a;
  }
  if (!['text', 'json', 'github'].includes(o.format)) die(`unknown format ${o.format}`);
  return o;
}

function describe(f) {
  const where = f.location ? ` ${f.location}` : '';
  return `${f.file}:${f.line}${where}  ${f.label} ${f.masked}`;
}

function report(findings, extra, o) {
  if (o.format === 'json') {
    process.stdout.write(JSON.stringify({ version: VERSION, ...extra, findings: findings.map(({ abs, ...rest }) => rest) }, null, 2) + '\n');
    return;
  }
  if (o.format === 'github') {
    for (const f of findings) {
      const lvl = ['committed+remote', 'committed', 'at-risk'].includes(f.grade) ? 'error' : 'warning';
      process.stdout.write(`::${lvl} file=${f.file},line=${f.line},title=agent-commit-guard ${f.rule}::${f.label} ${f.masked} (${f.grade})${f.location ? ` at ${f.location}` : ''}\n`);
    }
  }
  if (!findings.length) {
    process.stdout.write(c('32', `✓ No secrets found in ${extra.scanned} agent file(s).`) + '\n');
    return;
  }
  if (o.format === 'text') {
    let grade = null;
    for (const f of findings) {
      if (f.grade !== grade) {
        grade = f.grade;
        process.stdout.write(`\n${GRADE_TEXT[grade]}  ${c('2', GRADE_HELP[grade])}\n`);
      }
      process.stdout.write(`  ${describe(f)}  ${c('2', f.rule)}\n`);
    }
  }
  const fixable = findings.filter((f) => f.fixable).length;
  process.stdout.write(`\n${c('1', `${findings.length} secret(s)`)} in ${new Set(findings.map((f) => f.file)).size} file(s).`);
  if (fixable && !o.fix) process.stdout.write(` ${fixable} can be removed with ${c('1', 'agent-commit-guard --fix')}.`);
  process.stdout.write('\nRotate any credential that was ever committed: removing it does not un-leak it.\n');
}

async function readStdin() {
  if (process.stdin.isTTY) return '';
  const chunks = [];
  for await (const ch of process.stdin) chunks.push(ch);
  return Buffer.concat(chunks).toString('utf8');
}

// ---------------------------------------------------------------------------
// hook mode (Claude Code: JSON on stdin, exit 2 blocks the tool call)
// ---------------------------------------------------------------------------
async function hook(o) {
  let input = {};
  try { input = JSON.parse((await readStdin()) || '{}'); } catch { process.exit(0); }
  const cwd = input.cwd || process.cwd();
  const event = input.hook_event_name || '';
  try {
    if (event === 'PreToolUse' && input.tool_name === 'Bash') {
      const cmd = String(input.tool_input?.command || '');
      if (!/\bgit\b[^;&|\n]*\b(commit|push|add)\b/.test(cmd)) process.exit(0);
      const res = scan(cwd, { depth: o.depth });
      const risky = res.findings.filter((f) => ['committed+remote', 'committed', 'at-risk'].includes(f.grade));
      if (!risky.length) process.exit(0);
      const lines = risky.slice(0, 8).map((f) => `  - ${describe(f)} [${f.grade}]`).join('\n');
      process.stderr.write(
        `agent-commit-guard blocked this git command: ${risky.length} secret(s) in agent files that git would publish.\n${lines}\n` +
        `Fix: run \`npx -y agent-commit-guard --fix\` (removes leaked permission entries and git-ignores .claude/settings.local.json), ` +
        `move other secrets to environment variables, ask the user to rotate any credential that was committed, then retry.\n`);
      process.exit(2);
    }
    if (o.scrub && ['Stop', 'SubagentStop', 'SessionEnd'].includes(event)) {
      const res = scan(cwd, { depth: o.depth });
      const actions = fix(res);
      if (actions.length) process.stderr.write(`agent-commit-guard: ${actions.map((a) => `${a.file}: ${a.action}`).join('; ')}\n`);
    }
  } catch (e) {
    process.stderr.write(`agent-commit-guard: hook error (${e.message}); not blocking\n`);
  }
  process.exit(0);
}

// ---------------------------------------------------------------------------
// init: commit the guard so every contributor runs it
// ---------------------------------------------------------------------------
function init(o) {
  const root = path.resolve(o.dir);
  const done = [];
  const settingsPath = path.join(root, '.claude', 'settings.json');
  let settings = {};
  if (fs.existsSync(settingsPath)) {
    try { settings = JSON.parse(fs.readFileSync(settingsPath, 'utf8')); } catch { die(`${settingsPath} is not valid JSON; not touching it`); }
  }
  settings.hooks ??= {};
  settings.hooks.PreToolUse ??= [];
  const has = JSON.stringify(settings.hooks).includes('agent-commit-guard hook');
  if (!has) {
    settings.hooks.PreToolUse.push({ matcher: 'Bash', hooks: [{ type: 'command', command: 'npx -y agent-commit-guard hook' }] });
    fs.mkdirSync(path.dirname(settingsPath), { recursive: true });
    fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2) + '\n');
    done.push(`.claude/settings.json: PreToolUse hook blocks git commit/push/add while agent files hold secrets`);
  } else done.push('.claude/settings.json: hook already present');

  let gitDir = null;
  try { gitDir = execFileSync('git', ['rev-parse', '--git-path', 'hooks'], { cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim(); } catch { /* not a repo */ }
  if (gitDir) {
    const hp = path.resolve(root, gitDir, 'pre-commit');
    const line = 'npx -y agent-commit-guard --staged || exit 1';
    const cur = fs.existsSync(hp) ? fs.readFileSync(hp, 'utf8') : '#!/bin/sh\n';
    if (!cur.includes('agent-commit-guard')) {
      fs.mkdirSync(path.dirname(hp), { recursive: true });
      fs.writeFileSync(hp, cur + (cur.endsWith('\n') ? '' : '\n') + line + '\n');
      fs.chmodSync(hp, 0o755);
      done.push(`${path.relative(root, hp)}: pre-commit check (works for every agent and for humans)`);
    } else done.push('git pre-commit hook already present');
  }
  process.stdout.write(done.map((d) => `✓ ${d}`).join('\n') + '\nCommit .claude/settings.json so the guard runs for everyone on the project.\n');
}

// ---------------------------------------------------------------------------
const o = parse(process.argv.slice(2));
if (o.help) { process.stdout.write(HELP); process.exit(0); }
if (o.version) { process.stdout.write(VERSION + '\n'); process.exit(0); }
if (o.cmd === 'hook') await hook(o);
else if (o.cmd === 'init') init(o);
else {
  if (!fs.existsSync(o.dir)) die(`no such directory: ${o.dir}`);
  let res = scan(o.dir, { depth: o.depth, staged: o.staged, home: o.home });
  let actions = [];
  if (o.fix) {
    actions = fix(res);
    for (const a of actions) process.stdout.write(`${a.todo ? c('33', 'todo ') : c('32', 'fixed')} ${a.file}: ${a.action}\n`);
    res = scan(o.dir, { depth: o.depth, home: o.home });
  }
  let findings = res.findings;
  let sessionFiles = 0;
  if (o.sessions) {
    const s = await scanSessions(o.sessions.length ? o.sessions : sessionDirs());
    findings = findings.concat(s.findings);
    sessionFiles = s.files;
  }
  report(findings, { scanned: res.files.length + sessionFiles, files: res.files, actions }, o);
  process.exit(findings.length ? 1 : 0);
}
