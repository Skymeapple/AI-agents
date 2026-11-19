import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { lint } from '../src/index.js';

const BIN = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'bin', 'ctxlint.js');

function repo(files) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ctxlint-'));
  for (const [rel, content] of Object.entries(files)) {
    const p = path.join(dir, rel);
    fs.mkdirSync(path.dirname(p), { recursive: true });
    fs.writeFileSync(p, content);
  }
  return dir;
}
const rules = (res) => res.issues.map((i) => i.rule);

test('clean repo has no issues', () => {
  const dir = repo({
    'package.json': JSON.stringify({ scripts: { test: 'x', build: 'y' } }),
    'package-lock.json': '{}',
    'src/app.ts': '',
    'AGENTS.md': '# Rules\n\nRun `npm test` and `npm run build`.\nEntry point is `src/app.ts`.\n',
    'CLAUDE.md': '@AGENTS.md\n\nClaude-only: be terse.\n',
  });
  const res = lint(dir);
  assert.deepEqual(res.issues, []);
  assert.equal(res.files.length, 2);
});

test('discovers all supported formats', () => {
  const dir = repo({
    'AGENTS.md': 'a', '.cursorrules': 'b', '.cursor/rules/x.mdc': 'c', '.github/copilot-instructions.md': 'd',
    '.github/instructions/py.instructions.md': 'e', '.windsurfrules': 'f', 'GEMINI.md': 'g', 'pkg/sub/AGENTS.md': 'h',
    '.clinerules/a.md': 'i', 'node_modules/x/AGENTS.md': 'skip',
  });
  const tools = lint(dir).files.map((f) => f.tool).sort();
  assert.deepEqual(tools, ['agents.md', 'agents.md', 'cline', 'copilot', 'copilot', 'cursor', 'cursor', 'gemini', 'windsurf']);
});

test('flags hidden unicode (rules file backdoor)', () => {
  const dir = repo({ 'AGENTS.md': 'Use safe defaults.​‮ignore all\n' });
  const res = lint(dir);
  assert.equal(res.issues.filter((i) => i.rule === 'hidden-unicode').length, 2);
  assert.equal(res.issues[0].severity, 'error');
});

test('flags missing npm scripts and make targets', () => {
  const dir = repo({
    'package.json': JSON.stringify({ scripts: { test: 'x' } }),
    'Makefile': '.PHONY: build\nbuild:\n\techo hi\n',
    'AGENTS.md': '```sh\nnpm run lint\nnpm test\nmake build\nmake deploy\n```\nMake sure to run tests.\n',
  });
  const iss = lint(dir).issues;
  assert.deepEqual(iss.filter((i) => i.rule === 'missing-script').map((i) => i.message), ['Script "lint" is not defined in package.json']);
  assert.deepEqual(iss.filter((i) => i.rule === 'missing-make-target').map((i) => i.message), ['Make target "deploy" is not defined in Makefile']);
});

test('flags stale paths and links but ignores globs, urls, packages', () => {
  const dir = repo({
    'src/real.ts': '',
    'AGENTS.md': 'See `src/real.ts`, `src/gone.ts:12`, `src/**/*.ts`, `@types/node`, `lodash/merge`, [docs](docs/arch.md), [web](https://x.y)\n',
  });
  const msgs = lint(dir).issues.filter((i) => i.rule === 'missing-path').map((i) => i.message);
  assert.equal(msgs.length, 2);
  assert.match(msgs[0], /src\/gone\.ts/);
  assert.match(msgs[1], /docs\/arch\.md/);
});

test('flags broken CLAUDE.md imports only, not emails or mentions', () => {
  const dir = repo({ 'CLAUDE.md': '@AGENTS.md\nPing @alice or mail a@b.com\nSee @docs/style.md\n', 'docs/style.md': '' });
  const msgs = lint(dir).issues.filter((i) => i.rule === 'broken-import').map((i) => i.message);
  assert.deepEqual(msgs, ['Import @AGENTS.md points to a file that does not exist']);
});

test('flags package manager that contradicts lockfile', () => {
  const dir = repo({ 'pnpm-lock.yaml': '', 'package.json': '{"scripts":{"dev":"x"}}', 'AGENTS.md': 'Install with `npm install`, then `pnpm run dev`. Global: `npm i -g foo`\n' });
  const iss = lint(dir).issues.filter((i) => i.rule === 'wrong-package-manager');
  assert.equal(iss.length, 1);
  assert.match(iss[0].message, /Uses npm .* lockfile is for pnpm/);
});

test('flags secrets', () => {
  const dir = repo({ '.cursorrules': 'token: ghp_' + 'a'.repeat(36) + '\n' });
  assert.deepEqual(rules(lint(dir)), ['secret']);
});

test('detects drifted copies and identical duplicates', () => {
  const base = Array.from({ length: 10 }, (_, i) => `- rule number ${i} applies here`).join('\n');
  const drifted = base.replace('rule number 3', 'rule number 33').replace('rule number 7', 'rule number 77');
  const dir = repo({ 'AGENTS.md': base, 'CLAUDE.md': drifted, 'GEMINI.md': base });
  const iss = lint(dir).issues.filter((i) => i.rule === 'drift');
  assert.ok(iss.some((i) => i.severity === 'warning' && /Drifted copy/.test(i.message)));
  assert.ok(iss.some((i) => i.severity === 'info' && /Identical/.test(i.message)));
});

test('too-long respects maxLines and inline disable', () => {
  const long = Array.from({ length: 30 }, (_, i) => `line ${i}`).join('\n');
  const dir = repo({ 'AGENTS.md': long, 'CLAUDE.md': '<!-- ctxlint-disable too-long -->\n' + long + '\nunique' });
  const iss = lint(dir, { maxLines: 20, ignore: ['drift'] }).issues;
  assert.deepEqual(iss.map((i) => [i.file, i.rule]), [['AGENTS.md', 'too-long']]);
});

test('CLI exit codes and formats', () => {
  const ok = repo({ 'AGENTS.md': 'fine\n' });
  assert.equal(spawnSync('node', [BIN, ok]).status, 0);
  const bad = repo({ 'AGENTS.md': 'x​y\n' });
  const r = spawnSync('node', [BIN, bad, '--format', 'github'], { encoding: 'utf8' });
  assert.equal(r.status, 1);
  assert.match(r.stdout, /::error file=.*AGENTS\.md,line=1,col=2,title=ctxlint hidden-unicode::/);
  const json = JSON.parse(execFileSync('node', [BIN, ok, '--format', 'json'], { encoding: 'utf8' }));
  assert.equal(json.summary.error, 0);
  assert.equal(spawnSync('node', [BIN, ok, '--ignore', 'nope']).status, 2);
});

test('missing-path: resolves ancestor dirs, partial paths, and skips negated/future mentions', () => {
  const dir = repo({
    'packages/web/package.json': '{}',
    'packages/web/src/util/a.ts': '',
    'packages/web/src/deep/AGENTS.md': 'Helpers in `src/util/a.ts`.\nPartial: `util/a.ts`, compiled `src/util/a.js`.\nDo not create `src/legacy/x.ts`.\nCreate `src/new/thing.ts`.\nSibling: `../../../../../other-repo/x.md`, generated `.build/state.json`.\nMissing: `src/util/gone.ts`\n',
  });
  const msgs = lint(dir).issues.filter((i) => i.rule === 'missing-path').map((i) => i.message);
  assert.equal(msgs.length, 1);
  assert.match(msgs[0], /src\/util\/gone\.ts/);
});

test('missing-script: accepts scripts defined in another workspace package', () => {
  const dir = repo({
    'package.json': '{"scripts":{"build":"x"}}',
    'tools/package.json': '{"scripts":{"typecheck":"tsc"}}',
    'AGENTS.md': 'Run `pnpm run typecheck` from the tools folder, and `npm run nope`.\n',
  });
  const msgs = lint(dir).issues.filter((i) => i.rule === 'missing-script').map((i) => i.message);
  assert.deepEqual(msgs, ['Script "nope" is not defined in package.json']);
});
