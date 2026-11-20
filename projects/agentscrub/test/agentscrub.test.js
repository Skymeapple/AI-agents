import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { detect } from '../src/detect.js';
import { scan, fix, scanSessions } from '../src/scan.js';

const BIN = path.join(path.dirname(fileURLToPath(import.meta.url)), '..', 'bin', 'agentscrub.js');
// Fake credentials assembled at runtime so no real-looking token is committed.
const GH = 'gh' + 'p_' + 'A1b2C3d4'.repeat(5);
const PW = 'Tr0ub4dor' + '&3xyz';
const JWT = 'eyJ' + 'hbGciOiJIUzI1NiJ9' + '.eyJ' + 'yb2xlIjoic2VydmljZV9yb2xlIn0' + '.' + 'c2lnbmF0dXJlLXZhbHVl';

function repo(files, { git = true } = {}) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentscrub-'));
  for (const [rel, content] of Object.entries(files)) {
    const p = path.join(dir, rel);
    fs.mkdirSync(path.dirname(p), { recursive: true });
    fs.writeFileSync(p, typeof content === 'string' ? content : JSON.stringify(content, null, 2));
  }
  if (git) {
    const g = (...a) => execFileSync('git', a, { cwd: dir, stdio: 'ignore' });
    g('init', '-q'); g('config', 'user.email', 't@t'); g('config', 'user.name', 't');
    g('config', 'core.excludesFile', '/dev/null');
  }
  return dir;
}
const g = (dir, ...a) => execFileSync('git', a, { cwd: dir, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });

test('detects credentials inside approved commands', () => {
  const cases = {
    [`Bash(psql postgres://admin:${PW}@db.example.com:5432/app)`]: 'url-credentials',
    [`Bash(PGPASSWORD=${PW} psql -h db -U admin)`]: 'env-assignment',
    [`Bash(export OPENAI_API_KEY=${PW}${PW})`]: 'env-assignment',
    [`Bash(mysql -u root -p${PW} app)`]: 'mysql-p',
    [`Bash(curl -u admin:${PW} https://api.x.io)`]: 'basic-auth-flag',
    [`Bash(curl -H "Authorization: Bearer ${PW}${PW}" https://api.x.io)`]: 'auth-header',
    [`Bash(sshpass -p ${PW} ssh host)`]: 'sshpass',
    [`Bash(git push https://${GH}@github.com/o/r)`]: 'github-token',
    [`Bash(supabase --key ${JWT})`]: 'jwt',
  };
  for (const [s, rule] of Object.entries(cases)) {
    const hits = detect(s);
    assert.ok(hits.some((h) => h.rule === rule), `${rule} not found in ${s}: ${JSON.stringify(hits)}`);
    for (const h of hits) assert.ok(!h.masked.includes(PW) && !h.masked.includes(GH), 'value leaked in output');
  }
});

test('ignores placeholders, env references and ordinary commands', () => {
  for (const s of [
    'Bash(psql postgres://admin:$DB_PASS@db/app)', 'Bash(PGPASSWORD=$PGPASSWORD psql)', 'Bash(export TOKEN=${TOKEN})',
    'Bash(curl -H "Authorization: Bearer $TOKEN" https://x)', 'Bash(npm run test:*)', 'Bash(git commit -m "fix password reset")',
    'Bash(mysql -u root -p app)', 'Bash(API_KEY=your-api-key-here node x.js)', 'Bash(curl -u user:<password> https://x)',
    'Read(//home/me/.ssh/**)', 'psql postgres://admin:\\$DB_PASS@db/app', "/\\bsshpass\\s+-p\\s*(\"[^\"]*\"|'[^']*'|\\S+)/", 'Bash(docker run -e POSTGRES_PASSWORD=postgres db)', 'Bash(make -j4)',
  ]) assert.deepEqual(detect(s), [], s);
});

test('scans Claude settings allowlist and grades exposure', () => {
  const dir = repo({
    '.claude/settings.local.json': { permissions: { allow: ['Bash(npm test)', `Bash(PGPASSWORD=${PW} psql -h db)`], deny: [] } },
    '.claude/settings.json': { permissions: { allow: ['Bash(npm run lint)'] } },
  });
  let res = scan(dir);
  assert.equal(res.findings.length, 1);
  assert.equal(res.findings[0].grade, 'at-risk');
  assert.equal(res.findings[0].location, 'permissions.allow[1]');
  assert.equal(res.findings[0].line, 5);
  assert.ok(res.findings[0].fixable);
  g(dir, 'add', '-A'); g(dir, 'commit', '-qm', 'x');
  res = scan(dir);
  assert.equal(res.findings[0].grade, 'committed');
  g(dir, 'remote', 'add', 'origin', 'https://example.com/r.git');
  assert.equal(scan(dir).findings[0].grade, 'committed+remote');
});

test('local grade when git-ignored and never committed', () => {
  const dir = repo({ '.gitignore': '.claude/settings.local.json\n', '.claude/settings.local.json': { permissions: { allow: [`Bash(mysql -p${PW})`] } } });
  assert.equal(scan(dir).findings[0].grade, 'local');
});

test('MCP env blocks: literal secrets flagged, ${VAR} references allowed', () => {
  const dir = repo({
    '.mcp.json': { mcpServers: { gh: { command: 'npx', env: { GITHUB_TOKEN: GH, OTHER: '${OTHER_TOKEN}' } }, db: { env: { DB_PASSWORD: PW } } } },
    '.cursor/mcp.json': { mcpServers: { x: { env: { API_KEY: '${env:API_KEY}' }, bearer_token_env_var: 'CODEX_BOARD_TOKEN', token_file: '/run/secrets/tok' } } },
  });
  const f = scan(dir).findings;
  assert.deepEqual(f.map((x) => x.location).sort(), ['mcpServers.db.env.DB_PASSWORD', 'mcpServers.gh.env.GITHUB_TOKEN']);
  assert.ok(f.every((x) => !x.fixable));
});

test('text histories: aider and specstory', () => {
  const dir = repo({ '.aider.chat.history.md': `> run: curl -u bob:${PW} https://x\n`, '.specstory/history/a.md': `export STRIPE_SECRET=${PW}\n` });
  assert.deepEqual(scan(dir).findings.map((x) => x.agent).sort(), ['aider', 'specstory']);
});

test('--fix removes only leaked entries, keeps the rest, git-ignores local settings', () => {
  const dir = repo({ '.claude/settings.local.json': { permissions: { allow: ['Bash(npm test)', `Bash(curl -u a:${PW} https://x)`], ask: [`Bash(PGPASSWORD=${PW} psql)`] }, env: { FOO: '1' } } });
  const actions = fix(scan(dir));
  const data = JSON.parse(fs.readFileSync(path.join(dir, '.claude/settings.local.json'), 'utf8'));
  assert.deepEqual(data.permissions, { allow: ['Bash(npm test)'], ask: [] });
  assert.deepEqual(data.env, { FOO: '1' });
  assert.match(fs.readFileSync(path.join(dir, '.gitignore'), 'utf8'), /^\.claude\/settings\.local\.json$/m);
  assert.equal(actions.length, 2);
  assert.equal(scan(dir).findings.length, 0);
});

test('--staged scans the index, not the working tree', () => {
  const dir = repo({ '.claude/settings.json': { permissions: { allow: [`Bash(mysql -p${PW})`] } } });
  assert.equal(scan(dir, { staged: true }).findings.length, 0);
  g(dir, 'add', '-A');
  assert.equal(scan(dir, { staged: true }).findings.length, 1);
  fs.writeFileSync(path.join(dir, '.claude/settings.json'), '{}');
  assert.equal(scan(dir, { staged: true }).findings.length, 1, 'staged content still has the secret');
});

test('hook blocks git commit when tracked agent files hold secrets, allows otherwise', () => {
  const dir = repo({ '.claude/settings.local.json': { permissions: { allow: [`Bash(PGPASSWORD=${PW} psql)`] } } });
  const run = (event, command, extra = []) => spawnSync('node', [BIN, 'hook', ...extra], {
    input: JSON.stringify({ hook_event_name: event, tool_name: 'Bash', tool_input: { command }, cwd: dir }), encoding: 'utf8' });
  let r = run('PreToolUse', 'git commit -am wip');
  assert.equal(r.status, 2);
  assert.match(r.stderr, /blocked this git command/);
  assert.ok(!r.stderr.includes(PW), 'secret leaked into hook output');
  assert.equal(run('PreToolUse', 'ls -la').status, 0);
  assert.equal(run('PostToolUse', 'git commit').status, 0);
  // --scrub on Stop removes the entry, after which commits are allowed
  assert.equal(run('Stop', '', ['--scrub']).status, 0);
  assert.equal(run('PreToolUse', 'git commit -am wip').status, 0);
});

test('session transcripts: JSONL scanned, values masked', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'agentscrub-s-'));
  fs.mkdirSync(path.join(dir, 'proj'));
  fs.writeFileSync(path.join(dir, 'proj', 's.jsonl'), [
    JSON.stringify({ type: 'user', message: { content: 'hi' } }),
    JSON.stringify({ type: 'tool_result', content: `DATABASE_URL=postgres://u:${PW}@h/db` }),
  ].join('\n'));
  const s = await scanSessions([dir]);
  assert.equal(s.findings.length, 1);
  assert.equal(s.findings[0].line, 2);
});

test('CLI: exit codes, json output, init', () => {
  const clean = repo({ '.claude/settings.json': { permissions: { allow: ['Bash(npm test)'] } } });
  assert.equal(spawnSync('node', [BIN, clean]).status, 0);
  const dirty = repo({ '.mcp.json': { mcpServers: { a: { env: { API_TOKEN: GH } } } } });
  const r = spawnSync('node', [BIN, dirty, '--format', 'json'], { encoding: 'utf8' });
  assert.equal(r.status, 1);
  const j = JSON.parse(r.stdout);
  assert.equal(j.findings[0].rule, 'github-token');
  assert.ok(!r.stdout.includes(GH));
  assert.equal(spawnSync('node', [BIN, '--bogus']).status, 2);
  const i = spawnSync('node', [BIN, 'init', clean], { encoding: 'utf8' });
  assert.equal(i.status, 0, i.stderr);
  const s = JSON.parse(fs.readFileSync(path.join(clean, '.claude/settings.json'), 'utf8'));
  assert.deepEqual(s.permissions.allow, ['Bash(npm test)']);
  assert.equal(s.hooks.PreToolUse[0].hooks[0].command, 'npx -y agentscrub hook');
  assert.match(fs.readFileSync(path.join(clean, '.git/hooks/pre-commit'), 'utf8'), /agentscrub --staged/);
  spawnSync('node', [BIN, 'init', clean]);
  assert.equal(JSON.parse(fs.readFileSync(path.join(clean, '.claude/settings.json'), 'utf8')).hooks.PreToolUse.length, 1, 'init is idempotent');
});
