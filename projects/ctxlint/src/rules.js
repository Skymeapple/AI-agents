import fs from 'node:fs';
import path from 'node:path';

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

const exists = (p) => { try { fs.accessSync(p); return true; } catch { return false; } };
const readJSON = (p) => { try { return JSON.parse(fs.readFileSync(p, 'utf8')); } catch { return null; } };

/** Split markdown into code segments (fenced blocks + inline backticks) and prose, with line numbers. */
export function segments(text) {
  const lines = text.split(/\r?\n/);
  const code = []; // {line, text, inline}
  const prose = []; // {line, text}
  let inFence = false;
  let fenceMarker = '';
  lines.forEach((ln, i) => {
    const lineNo = i + 1;
    const fm = ln.match(/^\s*(```+|~~~+)/);
    if (fm) {
      if (!inFence) { inFence = true; fenceMarker = fm[1][0]; return; }
      if (fm[1][0] === fenceMarker) { inFence = false; return; }
    }
    if (inFence) { code.push({ line: lineNo, text: ln, inline: false }); return; }
    let rest = ln;
    const re = /`([^`]+)`/g;
    let m;
    while ((m = re.exec(ln))) code.push({ line: lineNo, text: m[1], inline: true });
    rest = ln.replace(/`[^`]*`/g, ' ');
    prose.push({ line: lineNo, text: rest });
  });
  return { lines, code, prose };
}

const HIDDEN = new Map([
  [0x00ad, 'SOFT HYPHEN'],
  [0x180e, 'MONGOLIAN VOWEL SEPARATOR'],
  [0x200b, 'ZERO WIDTH SPACE'],
  [0x200c, 'ZERO WIDTH NON-JOINER'],
  [0x200d, 'ZERO WIDTH JOINER'],
  [0x200e, 'LEFT-TO-RIGHT MARK'],
  [0x200f, 'RIGHT-TO-LEFT MARK'],
  [0x202a, 'LEFT-TO-RIGHT EMBEDDING'],
  [0x202b, 'RIGHT-TO-LEFT EMBEDDING'],
  [0x202c, 'POP DIRECTIONAL FORMATTING'],
  [0x202d, 'LEFT-TO-RIGHT OVERRIDE'],
  [0x202e, 'RIGHT-TO-LEFT OVERRIDE'],
  [0x2060, 'WORD JOINER'],
  [0x2061, 'FUNCTION APPLICATION'],
  [0x2062, 'INVISIBLE TIMES'],
  [0x2063, 'INVISIBLE SEPARATOR'],
  [0x2064, 'INVISIBLE PLUS'],
  [0x2066, 'LEFT-TO-RIGHT ISOLATE'],
  [0x2067, 'RIGHT-TO-LEFT ISOLATE'],
  [0x2068, 'FIRST STRONG ISOLATE'],
  [0x2069, 'POP DIRECTIONAL ISOLATE'],
  [0xfeff, 'ZERO WIDTH NO-BREAK SPACE (BOM)'],
]);

function hiddenName(cp) {
  if (HIDDEN.has(cp)) return HIDDEN.get(cp);
  if (cp >= 0xe0000 && cp <= 0xe007f) return 'UNICODE TAG CHARACTER';
  if (cp >= 0xfe00 && cp <= 0xfe0f) return null; // variation selectors are fine (emoji)
  return null;
}

const SECRET_PATTERNS = [
  ['AWS access key', /\bAKIA[0-9A-Z]{16}\b/],
  ['GitHub token', /\bgh[pousr]_[A-Za-z0-9]{36,}\b/],
  ['GitHub fine-grained token', /\bgithub_pat_[A-Za-z0-9_]{22,}\b/],
  ['Anthropic API key', /\bsk-ant-[A-Za-z0-9_-]{20,}/],
  ['OpenAI-style API key', /\bsk-(?:proj-)?[A-Za-z0-9]{20,}/],
  ['Slack token', /\bxox[baprs]-[A-Za-z0-9-]{10,}/],
  ['Google API key', /\bAIza[0-9A-Za-z_-]{35}\b/],
  ['Stripe secret key', /\b[sr]k_live_[0-9A-Za-z]{20,}\b/],
  ['Private key block', /-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----/],
];

const LOCKFILES = [
  ['pnpm-lock.yaml', 'pnpm'],
  ['yarn.lock', 'yarn'],
  ['bun.lockb', 'bun'],
  ['bun.lock', 'bun'],
  ['package-lock.json', 'npm'],
];

const PM_CMD = /(?:^|[\s;&|(])(npm|pnpm|yarn|bun)\s+(install|i|ci|add|run|test|remove|exec|dlx|x)\b/g;

function nearestUp(startDir, root, name) {
  let dir = startDir;
  const stop = path.resolve(root);
  for (;;) {
    const p = path.join(dir, name);
    if (exists(p)) return p;
    if (path.resolve(dir) === stop) return null;
    const parent = path.dirname(dir);
    if (parent === dir) return null;
    dir = parent;
  }
}

function makeTargets(makefile) {
  let src;
  try { src = fs.readFileSync(makefile, 'utf8'); } catch { return null; }
  if (/^\s*-?include\s/m.test(src) || /^%[^:]*:/m.test(src)) return null; // too dynamic to judge
  const targets = new Set();
  for (const ln of src.split(/\r?\n/)) {
    const m = ln.match(/^([A-Za-z0-9_.\-/ ]+?)\s*:(?!=)/);
    if (m && !ln.startsWith('\t')) m[1].split(/\s+/).forEach((t) => t && targets.add(t));
    const phony = ln.match(/^\.PHONY\s*:\s*(.*)$/);
    if (phony) phony[1].split(/\s+/).forEach((t) => t && targets.add(t));
  }
  return targets;
}

const PATH_EXT = /\.(md|mdc|js|mjs|cjs|ts|tsx|jsx|json|ya?ml|toml|py|rb|go|rs|java|kt|swift|c|h|cc|cpp|hpp|cs|php|sh|bash|sql|css|scss|html|vue|svelte|txt|env|lock|cfg|ini|xml|gradle|tf|proto)$/i;
const DYNAMIC = /[*?<>{}$%|\\]|\.\.\.|^[-@~/]|:\/\/|^\w+:/;

function looksLikeRepoPath(tok, root) {
  if (!tok.includes('/')) return false;
  if (DYNAMIC.test(tok) || /\s/.test(tok)) return false;
  const first = tok.replace(/^\.\//, '').split('/')[0];
  if (['node_modules', 'dist', 'build', '.git', 'out', 'target', 'coverage'].includes(first)) return false;
  return PATH_EXT.test(tok.replace(/\/$/, '')) || exists(path.join(root, first));
}

function cleanPathToken(t) {
  return t.trim().replace(/^['"(]+/, '').replace(/[),.;:'"]+$/, '').replace(/:\d+(?::\d+)?$/, '').replace(/#.*$/, '');
}

const INDEX_SKIP = new Set(['node_modules', '.git', 'dist', 'build-output', '.venv', 'venv', '__pycache__', '.next', '.turbo', 'target', 'coverage', '.cache']);

/** Walk the repo once: all relative paths (files + dirs, posix) and every package.json's scripts. */
export function buildIndex(root, limit = 300000) {
  const paths = new Set();
  const scripts = new Set();
  const stack = [''];
  while (stack.length && paths.size < limit) {
    const rel = stack.pop();
    let entries;
    try { entries = fs.readdirSync(path.join(root, rel), { withFileTypes: true }); } catch { continue; }
    for (const e of entries) {
      const r = rel ? `${rel}/${e.name}` : e.name;
      if (e.isDirectory()) {
        if (INDEX_SKIP.has(e.name)) continue;
        paths.add(r);
        stack.push(r);
      } else {
        paths.add(r);
        if (e.name === 'package.json') {
          const pkg = readJSON(path.join(root, r));
          Object.keys(pkg?.scripts || {}).forEach((k) => scripts.add(k));
        }
      }
    }
  }
  return { paths, scripts, truncated: paths.size >= limit };
}

function pathKnown(tok, file, ctx) {
  // Compiled-output references (TS ESM imports use .js) count if the source exists.
  const alts = [tok];
  const js = tok.match(/^(.*)\.(m?js|cjs|jsx)$/);
  if (js) alts.push(`${js[1]}.ts`, `${js[1]}.tsx`, `${js[1]}.mts`, `${js[1]}.cts`);
  return alts.some((t) => pathKnownExact(t, file, ctx));
}

function pathKnownExact(tok, file, ctx) {
  const clean = tok.replace(/^\.\//, '').replace(/\/$/, '');
  // Points outside the repo (sibling checkout etc.): can't verify, don't guess.
  const fromFile = path.resolve(file.dir, tok);
  if (path.relative(ctx.root, fromFile).startsWith('..')) return true;
  // Generated dot-directories (.build/, .vscode-test/) that aren't checked in.
  const first = clean.split('/')[0];
  if (first.startsWith('.') && first !== '..' && first !== '.' && !exists(path.join(ctx.root, first))) return true;
  // relative to the file's dir and every ancestor up to the repo root
  let dir = file.dir;
  for (;;) {
    if (exists(path.resolve(dir, tok))) return true;
    if (path.resolve(dir) === path.resolve(ctx.root)) break;
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  if (clean.startsWith('..')) return false;
  // partial path: some repo path ends with it
  const idx = ctx.index;
  if (idx.truncated) return true; // can't be sure in huge repos; avoid false alarms
  const suffix = '/' + clean;
  for (const p of idx.paths) if (p === clean || p.endsWith(suffix)) return true;
  return false;
}

const NEGATED_OR_FUTURE = /\b(don'?t|do not|never|avoid|instead of|rather than|not\s+(?:in|under|at)|create[sd]?|creating|add(?:ed|ing)? a new|new file|generate[sd]?|will be|e\.g\.|for example|such as|deprecated|removed|renamed|moved|legacy|old|sibling|outside (?:the|this) repo|separate repo|external)\b/i;

// ---------------------------------------------------------------------------
// per-file rules: (file, ctx) => issues[]
// ---------------------------------------------------------------------------

export const fileRules = {
  'hidden-unicode'(file) {
    const out = [];
    file.lines.forEach((ln, i) => {
      let col = 0;
      for (const ch of ln) {
        col++;
        const cp = ch.codePointAt(0);
        if (cp === 0xfeff && i === 0 && col === 1) continue; // leading BOM is harmless
        const name = hiddenName(cp);
        if (name) {
          out.push({ severity: 'error', line: i + 1, col,
            message: `Invisible character U+${cp.toString(16).toUpperCase().padStart(4, '0')} (${name}) can hide instructions from human reviewers` });
        }
      }
    });
    return out;
  },

  'secret'(file) {
    const out = [];
    file.lines.forEach((ln, i) => {
      for (const [name, re] of SECRET_PATTERNS) {
        if (re.test(ln)) { out.push({ severity: 'error', line: i + 1, message: `Looks like a ${name}; agent instruction files are sent to model providers` }); break; }
      }
    });
    return out;
  },

  'empty-file'(file) {
    return file.text.trim() === '' ? [{ severity: 'warning', line: 1, message: 'Instruction file is empty' }] : [];
  },

  'too-long'(file, ctx) {
    const out = [];
    const n = file.lines.length;
    const tokens = Math.round(file.text.length / 4);
    if (n > ctx.options.maxLines) {
      out.push({ severity: 'warning', line: n,
        message: `${n} lines (limit ${ctx.options.maxLines}). Long instruction files lower adherence; move task-specific parts into scoped rules or skills` });
    } else if (tokens > ctx.options.maxTokens) {
      out.push({ severity: 'warning', line: 1, message: `~${tokens} tokens (limit ${ctx.options.maxTokens}); this is loaded into every agent session` });
    }
    return out;
  },

  'missing-path'(file, ctx) {
    const out = [];
    const seen = new Set();
    const check = (raw, line, kind) => {
      const tok = cleanPathToken(raw);
      if (!tok || seen.has(tok)) return;
      if (kind === 'code' && !looksLikeRepoPath(tok, ctx.root)) return;
      if (kind === 'link' && (/^[a-z]+:/i.test(tok) || tok.startsWith('#') || tok.startsWith('/') || tok === '')) return;
      seen.add(tok);
      if (NEGATED_OR_FUTURE.test(file.lines[line - 1] || '')) return;
      if (!pathKnown(tok, file, ctx)) {
        out.push({ severity: 'warning', line, message: `References \`${tok}\`, which does not exist (stale instruction?)` });
      }
    };
    for (const c of file.seg.code) if (c.inline) check(c.text, c.line, 'code');
    for (const p of file.seg.prose) {
      const re = /\[[^\]]*\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g;
      let m;
      while ((m = re.exec(p.text))) check(decodeURI(m[1]), p.line, 'link');
    }
    return out;
  },

  'broken-import'(file) {
    if (!/^(CLAUDE|GEMINI)/.test(path.basename(file.abs)) && !file.rel.startsWith('.claude')) return [];
    const out = [];
    for (const p of file.seg.prose) {
      const re = /(?:^|\s)@((?:\.{1,2}\/)?[\w.\-/]+\.[A-Za-z0-9]+)/g;
      let m;
      while ((m = re.exec(p.text))) {
        const target = m[1].replace(/[.,;:]+$/, '');
        if (!exists(path.resolve(file.dir, target))) {
          out.push({ severity: 'error', line: p.line, message: `Import @${target} points to a file that does not exist` });
        }
      }
    }
    return out;
  },

  'missing-script'(file, ctx) {
    const out = [];
    const pkgPath = nearestUp(file.dir, ctx.root, 'package.json');
    const pkg = pkgPath && readJSON(pkgPath);
    const rootPkg = readJSON(path.join(ctx.root, 'package.json'));
    if (!pkg && !rootPkg) return out;
    const scripts = ctx.index.scripts;
    const seen = new Set();
    for (const c of file.seg.code) {
      if (/--filter|--workspace|\s-w\s|--prefix|\s-C\s|--cwd|\bcd\s/.test(c.text)) continue;
      const re = /(?:^|[\s;&|(])(npm|pnpm|yarn|bun)\s+(?:run(?:-script)?\s+)([A-Za-z0-9:_.\-]+)|(?:^|[\s;&|(])npm\s+(test|t)\b/g;
      let m;
      while ((m = re.exec(c.text))) {
        const name = m[2] || 'test';
        if (seen.has(name)) continue;
        seen.add(name);
        if (!scripts.has(name)) {
          const rel = path.relative(ctx.root, pkgPath || path.join(ctx.root, 'package.json')) || 'package.json';
          out.push({ severity: 'error', line: c.line, message: `Script "${name}" is not defined in ${rel}` });
        }
      }
    }
    return out;
  },

  'missing-make-target'(file, ctx) {
    const out = [];
    const mf = nearestUp(file.dir, ctx.root, 'Makefile') || nearestUp(file.dir, ctx.root, 'makefile') || nearestUp(file.dir, ctx.root, 'GNUmakefile');
    const seen = new Set();
    for (const c of file.seg.code) {
      if (/\s-C\s|\s-f\s|--directory|\bcd\s/.test(c.text)) continue;
      const re = /(?:^|[\s;&|(])make\s+((?:[A-Za-z0-9_.\-/]+\s*)+)/g;
      let m;
      while ((m = re.exec(c.text))) {
        for (const t of m[1].trim().split(/\s+/)) {
          if (!t || t.includes('=') || t.startsWith('-') || seen.has(t)) continue;
          seen.add(t);
          if (!mf) { out.push({ severity: 'error', line: c.line, message: `Uses \`make ${t}\` but there is no Makefile` }); continue; }
          const targets = makeTargets(mf);
          if (targets && !targets.has(t)) {
            out.push({ severity: 'error', line: c.line, message: `Make target "${t}" is not defined in ${path.relative(ctx.root, mf)}` });
          }
        }
      }
    }
    return out;
  },

  'wrong-package-manager'(file, ctx) {
    const lock = ctx.lockfilePM;
    if (!lock) return [];
    const out = [];
    const seen = new Set();
    for (const c of file.seg.code) {
      if (/\s-g\b|--global|\bglobal\s+add/.test(c.text)) continue;
      PM_CMD.lastIndex = 0;
      let m;
      while ((m = PM_CMD.exec(c.text))) {
        const pm = m[1];
        if (pm !== lock && !seen.has(pm)) {
          seen.add(pm);
          out.push({ severity: 'warning', line: c.line,
            message: `Uses ${pm} but the repo's lockfile is for ${lock}; agents will create a second lockfile or install different versions` });
        }
      }
    }
    return out;
  },
};

// ---------------------------------------------------------------------------
// cross-file rules: (files, ctx) => issues[] (each with .file)
// ---------------------------------------------------------------------------

function normLines(file) {
  return new Set(file.lines
    .map((l) => l.trim().toLowerCase().replace(/\s+/g, ' '))
    .filter((l) => l.length > 3 && !/^#+\s/.test(l) && !/^[-*_=`~]{3,}$/.test(l) && !/^---$/.test(l)));
}

export const crossRules = {
  'drift'(files) {
    const out = [];
    // Only compare "whole-project" files, not scoped rule fragments.
    const main = files.filter((f) => !/\/(rules|instructions|steering)\//.test(f.rel.split(path.sep).join('/')));
    const sets = main.map(normLines);
    for (let i = 0; i < main.length; i++) {
      for (let j = i + 1; j < main.length; j++) {
        const a = sets[i]; const b = sets[j];
        if (a.size < 5 || b.size < 5) continue;
        let inter = 0;
        for (const l of a) if (b.has(l)) inter++;
        const union = a.size + b.size - inter;
        const jac = inter / union;
        if (jac === 1) {
          out.push({ file: main[j], severity: 'info', line: 1,
            message: `Identical to ${main[i].rel}. Keep one copy and import it (CLAUDE.md: "@AGENTS.md") or symlink, so they can't drift` });
        } else if (jac >= 0.5) {
          out.push({ file: main[j], severity: 'warning', line: 1,
            message: `Drifted copy of ${main[i].rel}: ${Math.round(jac * 100)}% shared, ${a.size - inter} line(s) only there, ${b.size - inter} only here` });
        }
      }
    }
    return out;
  },

  'conflicting-package-managers'(files, ctx) {
    if (ctx.lockfilePM) return []; // wrong-package-manager covers this precisely
    const byPM = new Map();
    for (const f of files) {
      for (const c of f.seg.code) {
        PM_CMD.lastIndex = 0;
        let m;
        while ((m = PM_CMD.exec(c.text))) if (!byPM.has(m[1])) byPM.set(m[1], { file: f, line: c.line });
      }
    }
    if (byPM.size < 2) return [];
    const list = [...byPM.entries()].map(([pm, loc]) => `${pm} (${loc.file.rel}:${loc.line})`).join(', ');
    const first = [...byPM.values()][1];
    return [{ file: first.file, line: first.line, severity: 'warning', message: `Instruction files disagree on the package manager: ${list}` }];
  },
};

export function detectLockfilePM(root) {
  const found = LOCKFILES.filter(([f]) => exists(path.join(root, f))).map(([, pm]) => pm);
  const uniq = [...new Set(found)];
  return uniq.length === 1 ? uniq[0] : null;
}

export const ALL_RULES = [...Object.keys(fileRules), ...Object.keys(crossRules)];
