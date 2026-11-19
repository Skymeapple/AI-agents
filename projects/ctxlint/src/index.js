import fs from 'node:fs';
import path from 'node:path';
import { discover, toolFor } from './discover.js';
import { fileRules, crossRules, segments, detectLockfilePM, buildIndex, ALL_RULES } from './rules.js';

export { ALL_RULES, discover };

export const DEFAULTS = { maxLines: 200, maxTokens: 6000, ignore: [], files: null };

function loadConfig(root) {
  const p = path.join(root, '.ctxlintrc.json');
  try { return JSON.parse(fs.readFileSync(p, 'utf8')); } catch { return {}; }
}

function disabledRules(text) {
  const set = new Set();
  const re = /<!--\s*ctxlint-disable(?:\s+([\w\s,-]+?))?\s*-->/g;
  let m;
  while ((m = re.exec(text))) {
    if (!m[1]) set.add('*');
    else m[1].split(/[\s,]+/).filter(Boolean).forEach((r) => set.add(r));
  }
  return set;
}

/**
 * Lint all agent instruction files under `root`.
 * @returns {{root:string, files:{rel:string,tool:string,lines:number}[], issues:{file:string,rule:string,severity:string,line:number,col?:number,message:string}[]}}
 */
export function lint(root, opts = {}) {
  root = path.resolve(root);
  const options = { ...DEFAULTS, ...loadConfig(root), ...Object.fromEntries(Object.entries(opts).filter(([, v]) => v !== undefined)) };
  const ignore = new Set(options.ignore || []);
  const paths = options.files ? options.files.map((f) => path.resolve(root, f)) : discover(root);

  const files = paths.map((abs) => {
    const text = fs.readFileSync(abs, 'utf8');
    const seg = segments(text);
    return { abs, rel: path.relative(root, abs), dir: path.dirname(abs), text, lines: seg.lines, seg, disabled: disabledRules(text) };
  });
  const ctx = { root, options, lockfilePM: detectLockfilePM(root), index: buildIndex(root) };

  const issues = [];
  const push = (file, rule, iss) => {
    if (ignore.has(rule) || file.disabled.has('*') || file.disabled.has(rule)) return;
    issues.push({ file: file.rel, rule, severity: iss.severity, line: iss.line, ...(iss.col ? { col: iss.col } : {}), message: iss.message });
  };
  for (const file of files) {
    for (const [rule, fn] of Object.entries(fileRules)) {
      if (ignore.has(rule)) continue;
      for (const iss of fn(file, ctx)) push(file, rule, iss);
    }
  }
  for (const [rule, fn] of Object.entries(crossRules)) {
    if (ignore.has(rule)) continue;
    for (const iss of fn(files, ctx)) push(iss.file, rule, iss);
  }
  const order = { error: 0, warning: 1, info: 2 };
  issues.sort((a, b) => a.file.localeCompare(b.file) || a.line - b.line || order[a.severity] - order[b.severity]);
  return {
    root,
    files: files.map((f) => ({ rel: f.rel, tool: toolFor(f.rel), lines: f.lines.length })),
    issues,
  };
}

export function summarize(issues) {
  const s = { error: 0, warning: 0, info: 0 };
  for (const i of issues) s[i.severity]++;
  return s;
}
