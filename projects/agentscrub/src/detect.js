// Secret detection tuned for what coding agents save: approved shell commands,
// MCP server env blocks, and session transcripts. Two families:
//   1. Known key formats (provider prefixes, JWTs, private keys).
//   2. Credentials inside commands, which format-based scanners miss:
//      PGPASSWORD=..., mysql -p..., user:pass@host, curl -u, auth headers.

const PLACEHOLDER = /^(\$|%|\{\{|<|\[|\*{3,}|x{4,}$|X{4,}$|\.{3})|REDACTED|PLACEHOLDER|YOUR[_-]|your[_-]|CHANGE[_-]?ME|changeme|example|EXAMPLE|dummy|DUMMY|fake|FAKE|sample|SAMPLE|^(test|testing|password\d*|pass|passwd|secret|token|null|none|true|false|undefined|postgres|root|admin|guest|mysql|dev|local|localhost|development|qwerty|123456|1234|12345678|letmein)$|__[A-Z0-9_]+__|\$\{|\$\(|^\*+$/i;

export function isPlaceholder(v) {
  if (!v) return true;
  const s = String(v).replace(/^['"]|['"]$/g, '').replace(/^\\+/, ''); // \$VAR is still a reference
  if (/\[\^|\\[sSdDwWb]|\(\?[:!=]|\|\\/.test(s)) return true; // regex source, not a credential
  return s.length < 4 || PLACEHOLDER.test(s);
}

// [id, label, regex, group index of the secret value (0 = whole match)]
const FORMATS = [
  ['private-key', 'Private key block', /-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP |ENCRYPTED )?PRIVATE KEY-----/, 0],
  ['aws-access-key', 'AWS access key', /\b(?:AKIA|ASIA)[0-9A-Z]{16}\b/, 0],
  ['github-token', 'GitHub token', /\bgh[pousr]_[A-Za-z0-9]{36,}\b/, 0],
  ['github-pat', 'GitHub fine-grained token', /\bgithub_pat_[A-Za-z0-9_]{40,}\b/, 0],
  ['gitlab-token', 'GitLab token', /\bglpat-[A-Za-z0-9_-]{20,}\b/, 0],
  ['anthropic-key', 'Anthropic API key', /\bsk-ant-[A-Za-z0-9_-]{30,}/, 0],
  ['openai-key', 'OpenAI API key', /\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{32,}/, 0],
  ['slack-token', 'Slack token', /\bxox[abprs]-[A-Za-z0-9-]{10,}/, 0],
  ['stripe-key', 'Stripe secret key', /\b(?:sk|rk)_live_[0-9A-Za-z]{20,}\b/, 0],
  ['google-api-key', 'Google API key', /\bAIza[0-9A-Za-z_-]{35}\b/, 0],
  ['npm-token', 'npm token', /\bnpm_[A-Za-z0-9]{36}\b/, 0],
  ['huggingface-token', 'Hugging Face token', /\bhf_[A-Za-z0-9]{30,}\b/, 0],
  ['sendgrid-key', 'SendGrid API key', /\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b/, 0],
  ['jwt', 'JWT (e.g. Supabase service-role key)', /\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}/, 0],
];

const SECRET_NAME = '[A-Za-z0-9_]*(?:PASSWORD|PASSWD|PWD|PASS|SECRET|TOKEN|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|AUTH|CREDENTIALS?)[A-Za-z0-9_]*';

const COMMAND = [
  ['url-credentials', 'Password in URL', /\b[a-z][a-z0-9+.-]*:\/\/[^\s:/@'"]+:([^\s@/'"]+)@[^\s'"]+/i, 1],
  ['env-assignment', 'Secret in variable assignment', new RegExp(`(?:^|[\\s;&|('"])(?:export\\s+|set\\s+)?(${SECRET_NAME})=("[^"]*"|'[^']*'|[^\\s;&|)'"]+)`, 'i'), 2],
  ['password-flag', 'Password flag', /(?:^|\s)(?:--password|--passwd|--pass|--token|--api-key|--secret)(?:=|\s+)("[^"]*"|'[^']*'|[^\s'"]+)/i, 1],
  ['mysql-p', 'mysql -p<password>', /\b(?:mysql|mysqldump|mysqladmin|mariadb)\b[^\n]*?\s-p([^\s'"-][^\s'"]{3,})/, 1],
  ['sshpass', 'sshpass -p', /\bsshpass\s+-p\s*("[^"]*"|'[^']*'|\S+)/, 1],
  ['basic-auth-flag', 'curl -u user:password', /(?:^|\s)(?:-u|--user)\s+("[^"]*"|'[^']*'|[^\s:'"]+:[^\s'"]+)/, 1],
  ['auth-header', 'Authorization header', /\bAuthorization:\s*(?:Bearer|Basic|Token|Bot)\s+([^\s'"]{12,})/i, 1],
  ['api-key-header', 'API key header', /\b(?:x-api-key|api-key|x-auth-token|x-access-token|apikey|private-token):\s*([^\s'"]{12,})/i, 1],
];

const RULES = [...FORMATS, ...COMMAND];
export const RULE_IDS = RULES.map((r) => r[0]);

/** Masked form for reports. Provider keys keep their public prefix; passwords show at most 2 chars. */
export function mask(v, { prefix = false } = {}) {
  const s = String(v).replace(/^['"]|['"]$/g, '');
  const keep = prefix ? Math.min(4, Math.floor(s.length / 4)) : s.length >= 12 ? 2 : 0;
  return `${s.slice(0, keep)}…(${s.length} chars)`;
}

/**
 * Find secrets in one string. Returns [{rule, label, masked}], one per distinct value.
 * The secret value itself is never returned.
 */
export function detect(text, { names = false } = {}) {
  if (!text || typeof text !== 'string') return [];
  const out = [];
  const seen = new Set();
  for (const [id, label, re, group] of RULES) {
    const g = new RegExp(re.source, re.flags.includes('g') ? re.flags : re.flags + 'g');
    let m;
    while ((m = g.exec(text))) {
      const value = m[group] ?? m[0];
      if (group !== 0 && isPlaceholder(value)) continue;
      if (id === 'basic-auth-flag' && !String(value).includes(':')) continue;
      if (id === 'basic-auth-flag' && isPlaceholder(String(value).replace(/^['"]|['"]$/g, '').split(':').slice(1).join(':'))) continue;
      const key = String(value);
      if (seen.has(key)) continue;
      seen.add(key);
      out.push({ rule: id, label, masked: mask(value, { prefix: group === 0 }), ...(names ? { value } : {}) });
      if (m[0].length === 0) g.lastIndex++;
    }
  }
  // A value already reported by a format rule shouldn't be reported again by a command rule.
  return out.filter((f, i) => !out.some((o, j) => j < i && o.masked === f.masked));
}

/** For JSON env blocks: a key named like a secret holding a literal value. */
export function detectNamedValue(key, value) {
  if (typeof value !== 'string' || isPlaceholder(value)) return [];
  const direct = detect(value);
  if (direct.length) return direct;
  const refKey = /(?:env_?var|_env|Env|_name|Name|_file|File|_path|Path|_url|Url|header|Header|helper|Helper|_id|Id)$/.test(key);
  const envName = /^[A-Z][A-Z0-9_]{2,}$/.test(value); // the NAME of a variable, not its value
  if (!refKey && !envName && new RegExp(`^${SECRET_NAME}$`, 'i').test(key) && value.length >= 8 && !/\s/.test(value)) {
    return [{ rule: 'named-secret', label: `Literal value for ${key}`, masked: mask(value) }];
  }
  return [];
}
