import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import readline from 'node:readline';
import { execFileSync } from 'node:child_process';
import { detect, detectNamedValue } from './detect.js';

// ---------------------------------------------------------------------------
// Which files coding agents write secrets into
// ---------------------------------------------------------------------------

// JSON files, relative to a project directory. `fixable` marks Claude Code
// permission lists, where removing an entry is always safe.
export const JSON_FILES = [
  ['.claude/settings.local.json', 'claude'],
  ['.claude/settings.json', 'claude'],
  ['.mcp.json', 'mcp'],
  ['.cursor/mcp.json', 'cursor'],
  ['.vscode/mcp.json', 'vscode'],
  ['.gemini/settings.json', 'gemini'],
  ['.roo/mcp.json', 'roo'],
  ['.kiro/settings/mcp.json', 'kiro'],
  ['.amazonq/mcp.json', 'amazonq'],
];
// Directories / files scanned as text (histories, transcripts, agent state).
const TEXT_DIRS = [['.specstory', 'specstory'], ['.codex', 'codex'], ['.gemini', 'gemini']];
const TEXT_FILES = [['.aider.chat.history.md', 'aider'], ['.aider.input.history', 'aider']];
const SKIP = new Set(['node_modules', '.git', 'dist', 'build', 'vendor', '.venv', 'venv', '__pycache__', 'target', '.next', 'coverage']);
const MAX_TEXT_BYTES = 5 * 1024 * 1024;

const isFile = (p) => { try { return fs.statSync(p).isFile(); } catch { return false; } };
const isDir = (p) => { try { return fs.statSync(p).isDirectory(); } catch { return false; } };

function walkFiles(dir, out, depth = 0) {
  if (depth > 6) return;
  let entries;
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
  for (const e of entries) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walkFiles(p, out, depth + 1);
    else if (e.isFile()) out.push(p);
  }
}

/** Project directories to inspect: root plus any nested dir holding an agent folder (monorepos). */
function projectDirs(root, maxDepth) {
  const dirs = new Set([root]);
  const walk = (dir, depth) => {
    if (depth >= maxDepth) return;
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
    for (const e of entries) {
      if (!e.isDirectory() || SKIP.has(e.name)) continue;
      const p = path.join(dir, e.name);
      if (['.claude', '.cursor', '.vscode', '.gemini', '.codex', '.specstory', '.roo', '.kiro', '.amazonq'].includes(e.name)) dirs.add(dir);
      else if (!e.name.startsWith('.')) walk(p, depth + 1);
    }
    if (TEXT_FILES.some(([f]) => isFile(path.join(dir, f))) || isFile(path.join(dir, '.mcp.json'))) dirs.add(dir);
  };
  walk(root, 0);
  return [...dirs];
}

export function discover(root, { depth = 4 } = {}) {
  const files = [];
  for (const dir of projectDirs(root, depth)) {
    for (const [rel, agent] of JSON_FILES) {
      const p = path.join(dir, rel);
      if (isFile(p)) files.push({ abs: p, agent, kind: 'json' });
    }
    for (const [rel, agent] of TEXT_FILES) {
      const p = path.join(dir, rel);
      if (isFile(p)) files.push({ abs: p, agent, kind: 'text' });
    }
    for (const [rel, agent] of TEXT_DIRS) {
      const d = path.join(dir, rel);
      if (!isDir(d)) continue;
      const found = [];
      walkFiles(d, found);
      for (const p of found) {
        if (files.some((f) => f.abs === p)) continue;
        files.push({ abs: p, agent, kind: p.endsWith('.json') ? 'json' : 'text' });
      }
    }
  }
  return files;
}

// ---------------------------------------------------------------------------
// Scanning
// ---------------------------------------------------------------------------

function parseJSONLoose(text) {
  try { return JSON.parse(text); } catch { /* fall through */ }
  // .vscode/*.json allows comments and trailing commas
  const stripped = text.replace(/("(?:\\.|[^"\\])*")|\/\/[^\n]*|\/\*[\s\S]*?\*\//g, (m, str) => str || '').replace(/,(\s*[}\]])/g, '$1');
  try { return JSON.parse(stripped); } catch { return undefined; }
}

function lineOf(raw, value) {
  const needle = JSON.stringify(value).slice(1, -1);
  const i = raw.indexOf(needle);
  return i < 0 ? 1 : raw.slice(0, i).split('\n').length;
}

function scanJSON(file, raw) {
  const data = parseJSONLoose(raw);
  if (data === undefined) return scanText(file, raw);
  const out = [];
  const visit = (node, trail, key) => {
    if (typeof node === 'string') {
      const permList = trail.length >= 3 && trail[trail.length - 3] === 'permissions' && ['allow', 'ask', 'deny'].includes(trail[trail.length - 2]);
      const hits = permList || key === undefined || typeof key === 'number' ? detect(node) : detectNamedValue(String(key), node);
      for (const h of hits) {
        out.push({ ...h, line: lineOf(raw, node), location: fmtTrail(trail),
          fixable: permList && file.agent === 'claude' });
      }
    } else if (Array.isArray(node)) {
      node.forEach((v, i) => visit(v, [...trail, i], i));
    } else if (node && typeof node === 'object') {
      for (const [k, v] of Object.entries(node)) visit(v, [...trail, k], k);
    }
  };
  visit(data, [], undefined);
  return out;
}

function fmtTrail(trail) {
  return trail.map((t) => (typeof t === 'number' ? `[${t}]` : `.${t}`)).join('').replace(/^\./, '') || '(root)';
}

function scanText(file, raw) {
  const out = [];
  raw.split(/\r?\n/).forEach((ln, i) => {
    for (const h of detect(ln)) out.push({ ...h, line: i + 1, location: null, fixable: false });
  });
  return out;
}

// ---------------------------------------------------------------------------
// Git exposure grading
// ---------------------------------------------------------------------------

function gitOk(cwd, args) {
  try {
    execFileSync('git', args, { cwd, stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

function git(cwd, args) {
  try {
    return execFileSync('git', args, { cwd, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
  } catch (e) {
    return e.status === 1 ? '' : null;
  }
}

export function exposure(abs) {
  const dir = path.dirname(abs);
  const top = git(dir, ['rev-parse', '--show-toplevel']);
  if (!top) return { grade: 'no-git', repo: null };
  const rel = path.relative(top, abs);
  const committed = Boolean(git(top, ['log', '--all', '--format=%h', '-n', '1', '--', rel]));
  const tracked = git(top, ['ls-files', '--', rel]) !== '';
  const ignored = gitOk(top, ['check-ignore', '-q', '--no-index', '--', rel]);
  const remote = Boolean(git(top, ['remote']));
  let grade;
  if (committed || tracked) grade = remote ? 'committed+remote' : 'committed';
  else if (ignored) grade = 'local';
  else grade = 'at-risk';
  return { grade, repo: top, rel, tracked, ignored, committed, remote };
}

export const GRADE_ORDER = { 'committed+remote': 0, committed: 1, 'at-risk': 2, local: 3, 'no-git': 4 };

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/** Scan agent files in a project. Returns { files, findings }. */
export function scan(root, { depth = 4, staged = false } = {}) {
  root = path.resolve(root);
  let files = discover(root, { depth });
  const readers = new Map();
  if (staged) {
    // pre-commit: scan what is about to be committed, from the index
    const top = git(root, ['rev-parse', '--show-toplevel']);
    const names = top ? (git(top, ['diff', '--cached', '--name-only', '--diff-filter=ACMR']) || '').split('\n').filter(Boolean) : [];
    const stagedAbs = new Set(names.map((n) => path.join(top, n)));
    const known = files.filter((f) => stagedAbs.has(f.abs));
    files = known;
    for (const f of files) readers.set(f.abs, () => git(top, ['show', `:${path.relative(top, f.abs)}`]) || '');
  }
  const findings = [];
  for (const f of files) {
    let raw;
    try {
      if (readers.has(f.abs)) raw = readers.get(f.abs)();
      else {
        if (fs.statSync(f.abs).size > MAX_TEXT_BYTES) continue;
        raw = fs.readFileSync(f.abs, 'utf8');
      }
    } catch { continue; }
    const hits = f.kind === 'json' ? scanJSON(f, raw) : scanText(f, raw);
    if (!hits.length) continue;
    const exp = exposure(f.abs);
    for (const h of hits) findings.push({ file: path.relative(root, f.abs) || f.abs, abs: f.abs, agent: f.agent, grade: exp.grade, ...h });
  }
  findings.sort((a, b) => GRADE_ORDER[a.grade] - GRADE_ORDER[b.grade] || a.file.localeCompare(b.file) || a.line - b.line);
  return { root, files: files.map((f) => ({ file: path.relative(root, f.abs), agent: f.agent })), findings };
}

/**
 * Remove leaked entries from Claude Code permission lists, and make sure
 * .claude/settings.local.json is git-ignored. Never touches anything else.
 */
export function fix(result) {
  const actions = [];
  const byFile = new Map();
  for (const f of result.findings.filter((x) => x.fixable)) {
    if (!byFile.has(f.abs)) byFile.set(f.abs, []);
    byFile.get(f.abs).push(f);
  }
  for (const [abs, list] of byFile) {
    const raw = fs.readFileSync(abs, 'utf8');
    const data = JSON.parse(raw);
    let removed = 0;
    for (const listName of ['allow', 'ask', 'deny']) {
      const arr = data?.permissions?.[listName];
      if (!Array.isArray(arr)) continue;
      const keep = arr.filter((entry) => typeof entry !== 'string' || detect(entry).length === 0);
      removed += arr.length - keep.length;
      data.permissions[listName] = keep;
    }
    if (removed) {
      const indent = (raw.match(/\n([ \t]+)"/) || [, '  '])[1];
      fs.writeFileSync(abs, JSON.stringify(data, null, indent) + (raw.endsWith('\n') ? '\n' : ''));
      actions.push({ file: path.relative(result.root, abs), action: `removed ${removed} permission entr${removed === 1 ? 'y' : 'ies'} holding secrets` });
    }
    void list;
  }
  // .gitignore the local settings file wherever one exists un-ignored
  const locals = new Set(discover(result.root).filter((f) => f.abs.endsWith(path.join('.claude', 'settings.local.json'))).map((f) => f.abs));
  for (const abs of locals) {
    const exp = exposure(abs);
    if (!exp.repo) continue;
    if (!exp.ignored) {
      const gi = path.join(exp.repo, '.gitignore');
      const line = exp.rel.split(path.sep).join('/');
      const cur = isFile(gi) ? fs.readFileSync(gi, 'utf8') : '';
      fs.writeFileSync(gi, cur + (cur && !cur.endsWith('\n') ? '\n' : '') + line + '\n');
      actions.push({ file: '.gitignore', action: `added ${line}` });
    }
    if (exp.tracked) {
      actions.push({ todo: true, file: exp.rel, action: `still tracked by git: run \`git rm --cached ${exp.rel}\`, then rotate any secret it ever held (it remains in history)` });
    }
  }
  return actions;
}

// ---------------------------------------------------------------------------
// Session transcripts (~/.claude/projects, ~/.codex/sessions)
// ---------------------------------------------------------------------------

export function sessionDirs() {
  const home = os.homedir();
  return [path.join(home, '.claude', 'projects'), path.join(home, '.codex', 'sessions')].filter(isDir);
}

function strings(node, out) {
  if (typeof node === 'string') out.push(node);
  else if (Array.isArray(node)) node.forEach((v) => strings(v, out));
  else if (node && typeof node === 'object') Object.values(node).forEach((v) => strings(v, out));
  return out;
}

export async function scanSessions(dirs) {
  const files = [];
  for (const d of dirs) walkFiles(d, files);
  const findings = [];
  for (const abs of files.filter((f) => /\.(jsonl|json)$/.test(f))) {
    const rl = readline.createInterface({ input: fs.createReadStream(abs, 'utf8'), crlfDelay: Infinity });
    const seen = new Set();
    let n = 0;
    for await (const line of rl) {
      n++;
      if (!line) continue;
      let texts;
      try { texts = strings(JSON.parse(line), []); } catch { texts = [line]; }
      for (const t of texts) {
        for (const h of detect(t)) {
          const k = h.rule + h.masked;
          if (seen.has(k)) continue;
          seen.add(k);
          findings.push({ file: abs, line: n, grade: 'session', agent: abs.includes(`${path.sep}.codex${path.sep}`) ? 'codex' : 'claude', ...h, location: null, fixable: false });
        }
      }
    }
  }
  return { files: files.length, findings };
}
