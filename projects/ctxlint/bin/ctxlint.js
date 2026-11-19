#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { lint, summarize, ALL_RULES } from '../src/index.js';

const HELP = `ctxlint - lint the instruction files your AI coding agents read
(AGENTS.md, CLAUDE.md, GEMINI.md, .cursorrules, .cursor/rules, Copilot,
Windsurf, Cline, Kiro, Junie, Amazon Q).

Usage: ctxlint [dir] [options]

Options:
  --format <f>       text (default) | json | github
  --strict           exit 1 on warnings too, not only errors
  --ignore <rules>   comma-separated rule ids to skip
  --max-lines <n>    line budget per file (default 200)
  --max-tokens <n>   token budget per file (default 6000)
  --list-rules       print rule ids
  -v, --version      print version
  -h, --help         show help

Disable rules for one file with:  <!-- ctxlint-disable rule-a, rule-b -->
Project config: .ctxlintrc.json  {"ignore": [], "maxLines": 200}
`;

function parse(argv) {
  const o = { dir: '.', format: 'text' };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => { const v = argv[++i]; if (v === undefined) die(`Missing value for ${a}`); return v; };
    if (a === '-h' || a === '--help') o.help = true;
    else if (a === '-v' || a === '--version') o.version = true;
    else if (a === '--list-rules') o.listRules = true;
    else if (a === '--strict') o.strict = true;
    else if (a === '--format') o.format = next();
    else if (a === '--ignore') o.ignore = next().split(',').map((s) => s.trim()).filter(Boolean);
    else if (a === '--max-lines') o.maxLines = Number(next());
    else if (a === '--max-tokens') o.maxTokens = Number(next());
    else if (a.startsWith('-')) die(`Unknown option ${a}`);
    else o.dir = a;
  }
  return o;
}

function die(msg) { process.stderr.write(`ctxlint: ${msg}\n`); process.exit(2); }

const useColor = process.stdout.isTTY && !process.env.NO_COLOR;
const c = (code, s) => (useColor ? `\x1b[${code}m${s}\x1b[0m` : s);
const SEV = { error: (s) => c('31', s), warning: (s) => c('33', s), info: (s) => c('36', s) };

const o = parse(process.argv.slice(2));
if (o.help) { process.stdout.write(HELP); process.exit(0); }
if (o.version) {
  const pkg = JSON.parse(fs.readFileSync(path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'package.json'), 'utf8'));
  process.stdout.write(`${pkg.version}\n`); process.exit(0);
}
if (o.listRules) { process.stdout.write(ALL_RULES.join('\n') + '\n'); process.exit(0); }
if (!fs.existsSync(o.dir)) die(`No such directory: ${o.dir}`);

const unknown = (o.ignore || []).filter((r) => !ALL_RULES.includes(r));
if (unknown.length) die(`Unknown rule(s): ${unknown.join(', ')}. See --list-rules`);

const res = lint(o.dir, { ignore: o.ignore, maxLines: o.maxLines, maxTokens: o.maxTokens });
const sum = summarize(res.issues);

if (o.format === 'json') {
  process.stdout.write(JSON.stringify({ ...res, summary: sum }, null, 2) + '\n');
} else if (o.format === 'github') {
  const lvl = { error: 'error', warning: 'warning', info: 'notice' };
  for (const i of res.issues) {
    const file = path.relative(process.cwd(), path.join(res.root, i.file));
    process.stdout.write(`::${lvl[i.severity]} file=${file},line=${i.line}${i.col ? `,col=${i.col}` : ''},title=ctxlint ${i.rule}::${i.message.replace(/\r?\n/g, ' ')}\n`);
  }
  process.stdout.write(`ctxlint: ${res.files.length} file(s), ${sum.error} error(s), ${sum.warning} warning(s), ${sum.info} info\n`);
} else if (o.format === 'text') {
  if (!res.files.length) {
    process.stdout.write('No agent instruction files found (AGENTS.md, CLAUDE.md, .cursorrules, ...).\n');
  } else {
    let current = null;
    for (const i of res.issues) {
      if (i.file !== current) { current = i.file; process.stdout.write(`\n${c('4', i.file)}\n`); }
      const loc = `${i.line}${i.col ? `:${i.col}` : ''}`.padEnd(7);
      process.stdout.write(`  ${c('2', loc)} ${SEV[i.severity](i.severity.padEnd(7))} ${i.message}  ${c('2', i.rule)}\n`);
    }
    const scanned = res.files.map((f) => `${f.rel} [${f.tool}]`).join(', ');
    process.stdout.write(`\n${c('2', `Scanned: ${scanned}`)}\n`);
    const line = `${sum.error} error(s), ${sum.warning} warning(s), ${sum.info} info`;
    process.stdout.write((sum.error ? SEV.error(line) : sum.warning ? SEV.warning(line) : c('32', `✓ ${line}`)) + '\n');
  }
} else die(`Unknown format ${o.format}`);

process.exit(sum.error || (o.strict && sum.warning) ? 1 : 0);
