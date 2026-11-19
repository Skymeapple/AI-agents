import fs from 'node:fs';
import path from 'node:path';

const SKIP_DIRS = new Set([
  'node_modules', '.git', 'dist', 'build', 'out', 'vendor', '.venv', 'venv',
  '__pycache__', '.next', '.turbo', 'target', 'coverage', '.cache',
]);

// Root-level (or any-level for AGENTS.md / CLAUDE.md) single files.
const ANY_DEPTH_NAMES = new Set(['AGENTS.md', 'AGENTS.override.md', 'CLAUDE.md', 'CLAUDE.local.md', 'GEMINI.md']);
const ROOT_FILES = [
  '.cursorrules',
  '.windsurfrules',
  '.clinerules',
  '.github/copilot-instructions.md',
  '.junie/guidelines.md',
  '.claude/CLAUDE.md',
  '.gemini/GEMINI.md',
];
// Directories whose markdown-ish files are rule files.
const RULE_DIRS = [
  ['.cursor/rules', /\.(md|mdc)$/],
  ['.windsurf/rules', /\.md$/],
  ['.clinerules', /\.md$/],
  ['.claude/rules', /\.md$/],
  ['.github/instructions', /\.instructions\.md$/],
  ['.amazonq/rules', /\.md$/],
  ['.kiro/steering', /\.md$/],
  ['.agents/rules', /\.md$/],
];

export function toolFor(rel) {
  const r = rel.split(path.sep).join('/');
  const base = path.posix.basename(r);
  if (base.startsWith('AGENTS')) return 'agents.md';
  if (base.startsWith('CLAUDE') || r.startsWith('.claude/')) return 'claude';
  if (base === 'GEMINI.md') return 'gemini';
  if (r.startsWith('.cursor') ) return 'cursor';
  if (r.startsWith('.windsurf')) return 'windsurf';
  if (r.startsWith('.clinerules')) return 'cline';
  if (r.startsWith('.github/')) return 'copilot';
  if (r.startsWith('.junie')) return 'junie';
  if (r.startsWith('.amazonq')) return 'amazonq';
  if (r.startsWith('.kiro')) return 'kiro';
  return 'other';
}

function isFile(p) {
  try { return fs.statSync(p).isFile(); } catch { return false; }
}

function walkRuleDir(dir, re, out) {
  let entries;
  try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
  for (const e of entries) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) walkRuleDir(p, re, out);
    else if (e.isFile() && re.test(e.name)) out.add(p);
  }
}

/** Find all AI-agent instruction files under `root`. Returns absolute paths, sorted. */
export function discover(root, { maxDepth = 8 } = {}) {
  const found = new Set();
  for (const f of ROOT_FILES) {
    const p = path.join(root, f);
    if (isFile(p)) found.add(p);
  }
  for (const [d, re] of RULE_DIRS) walkRuleDir(path.join(root, d), re, found);

  const walk = (dir, depth) => {
    if (depth > maxDepth) return;
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { return; }
    for (const e of entries) {
      if (e.isDirectory()) {
        if (SKIP_DIRS.has(e.name) || (e.name.startsWith('.') && depth > 0)) continue;
        if (e.name.startsWith('.') && depth === 0) continue; // dot-dirs handled above
        walk(path.join(dir, e.name), depth + 1);
      } else if (e.isFile() && ANY_DEPTH_NAMES.has(e.name)) {
        found.add(path.join(dir, e.name));
      }
    }
  };
  walk(root, 0);
  return [...found].sort();
}
