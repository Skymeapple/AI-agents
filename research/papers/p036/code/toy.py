"""Toy instruction/data world for studying prompt injection mechanisms.

Sequence layout (length 16):
  [BOS, SYS, s, DAT, d_1 .. d_12, QRY]  -> predict answer at QRY position.
s is the trusted (system) instruction: one of 4 ops, or DELEG ("follow the
instruction found in the data, if it is legitimate").  Data contains digits,
filler tokens and possibly instruction tokens (legitimate or injected).
Each position also carries an out-of-band role id (0 = system/structural,
1 = untrusted data, 2 = trusted-provenance data) that the attacker cannot write.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- vocabulary ----------------
PAD, BOS, SYS, DAT, QRY, NONE, DELEG, SIG = range(8)
OPS = [8, 9, 10, 11]            # FIRST, LAST, MAX, MIN
OPNAMES = ["FIRST", "LAST", "MAX", "MIN"]
DIG0 = 12                       # digits 12..21
FIL0 = 22                       # filler 22..29
NFIL = 8
VOCAB = 30
NOOP_CLASS = 10                 # output classes: digits 0..9, NOOP=10
NCLS = 11
NDATA = 12
SEQ = 4 + NDATA + 1
DSTART = 4                      # index of first data position
# attacker may write any non-digit, non-PAD token (including forged structural tokens)
ATTACK_VOCAB = np.array([BOS, SYS, DAT, QRY, NONE, DELEG, SIG] + OPS + list(range(FIL0, FIL0 + NFIL)))


def apply_op(op_idx, digits, mask):
    """op_idx: (B,) in 0..3 ; digits: (B,NDATA) values; mask: (B,NDATA) bool digit positions."""
    B = digits.shape[0]
    big = 99
    first_pos = np.argmax(mask, axis=1)
    last_pos = NDATA - 1 - np.argmax(mask[:, ::-1], axis=1)
    first = digits[np.arange(B), first_pos]
    last = digits[np.arange(B), last_pos]
    mx = np.where(mask, digits, -big).max(1)
    mn = np.where(mask, digits, big).min(1)
    table = np.stack([first, last, mx, mn], 1)
    return table[np.arange(B), op_idx], table


def base_data(rng, B):
    toks = FIL0 + rng.integers(0, NFIL, size=(B, NDATA))
    digits = rng.integers(0, 10, size=(B, NDATA))
    nd = rng.integers(3, 7, size=B)
    ranks = np.argsort(rng.random((B, NDATA)), axis=1).argsort(axis=1)
    mask = ranks < nd[:, None]
    toks = np.where(mask, DIG0 + digits, toks)
    return toks, digits, mask


def free_span(rng, blocked, length):
    """for each row pick a random start s such that positions s..s+length-1 are unblocked; -1 if none."""
    B = blocked.shape[0]
    nst = NDATA - length + 1
    cs = np.concatenate([np.zeros((B, 1), int), np.cumsum(blocked, 1)], 1)
    ok = (cs[:, length:length + nst] - cs[:, :nst]) == 0
    sc = np.where(ok, rng.random((B, nst)), -1.0)
    st = sc.argmax(1)
    st[~ok.any(1)] = -1
    return st


def gen_spans(rng, B, lengths):
    """Return toks,digits,mask and a list of starts, one per requested length, spans disjoint and digit-free."""
    while True:
        toks, digits, mask = base_data(rng, B)
        blocked = mask.copy()
        starts = []
        okall = np.ones(B, bool)
        for L in lengths:
            st = free_span(rng, blocked, L)
            okall &= st >= 0
            idx = np.arange(NDATA)[None]
            blocked |= (idx >= st[:, None]) & (idx < st[:, None] + L) & (st[:, None] >= 0)
            starts.append(st)
        if okall.all():
            return toks, digits, mask, starts
        # regenerate failing rows
        idx = np.where(~okall)[0]
        sub = gen_spans(rng, len(idx), lengths)
        toks[idx], digits[idx], mask[idx] = sub[0], sub[1], sub[2]
        for i in range(len(lengths)):
            starts[i][idx] = sub[3][i][:]
        return toks, digits, mask, starts


def assemble(s_tok, data_toks, data_roles):
    B = len(s_tok)
    x = np.zeros((B, SEQ), np.int64)
    r = np.zeros((B, SEQ), np.int64)
    x[:, 0], x[:, 1], x[:, 2], x[:, 3] = BOS, SYS, s_tok, DAT
    x[:, DSTART:DSTART + NDATA] = data_toks
    x[:, -1] = QRY
    r[:, DSTART:DSTART + NDATA] = data_roles
    return x, r


TRAIN_TEMPLATES = [[], [NONE]]                     # prefix tokens before injected op
HELDOUT_TEMPLATES = [[SYS], [QRY], [DELEG], [DAT, SYS]]


def distinct_ops(rng, table, op_a):
    """sample op_b != op_a with table[op_b] != table[op_a]; returns op_b and validity mask."""
    B = table.shape[0]
    op_b = rng.integers(0, 4, size=B)
    for _ in range(20):
        bad = (op_b == op_a) | (table[np.arange(B), op_b] == table[np.arange(B), op_a])
        if not bad.any():
            break
        op_b[bad] = rng.integers(0, 4, size=bad.sum())
    valid = ~((op_b == op_a) | (table[np.arange(B), op_b] == table[np.arange(B), op_a]))
    return op_b, valid


def make_batch(rng, B, kind, q=1.0, inband=False, templates=TRAIN_TEMPLATES, forge_sig=False):
    """kind in {CLEAN, INJ, DEL_L, DEL_A, DEL_LA}.
    q: probability that the out-of-band provenance flag is correct (legit->trusted, injected->untrusted).
    inband: legit instructions marked with in-band SIG token instead of role flag (roles all untrusted).
    forge_sig: injected instruction preceded by SIG (adaptive in-band attacker).
    Returns x, roles, y, meta dict (with table, op ids, valid mask)."""
    tlen = max(2, max(len(t) for t in templates) + 1 + (1 if forge_sig else 0))  # >=2 so legit/injected spans match
    lens = []
    if kind in ("INJ", "DEL_A"):
        lens = [tlen]
    elif kind == "DEL_L":
        lens = [2]
    elif kind == "DEL_LA":
        lens = [2, tlen]
    toks, digits, mask, starts = gen_spans(rng, B, lens) if lens else (*base_data(rng, B), [])
    roles = np.ones((B, NDATA), np.int64)
    _, table = apply_op(np.zeros(B, int), digits, mask)
    meta = {"table": table}
    valid = np.ones(B, bool)
    ar = np.arange(B)

    def put_legit(st, op_l):
        trusted = rng.random(B) < q
        for b in range(B):
            p = st[b]
            if inband:
                toks[b, p] = SIG; toks[b, p + 1] = OPS[op_l[b]]
            else:
                toks[b, p] = OPS[op_l[b]]  # second slot stays filler
                roles[b, p] = 2 if trusted[b] else 1

    def put_inj(st, op_a):
        flip = rng.random(B) >= q  # noisy flag marks injected as trusted
        tid = rng.integers(0, len(templates), size=B)
        for b in range(B):
            seq = list(templates[tid[b]]) + ([SIG] if forge_sig else []) + [OPS[op_a[b]]]
            p = st[b]
            for j, t in enumerate(seq):
                toks[b, p + j] = t
            if not inband:
                roles[b, p + len(seq) - 1] = 2 if flip[b] else 1

    if kind == "CLEAN":
        op = rng.integers(0, 4, size=B)
        s = np.array([OPS[o] for o in op]); y = table[ar, op]
        meta["op_sys"] = op
    elif kind == "INJ":
        op = rng.integers(0, 4, size=B)
        op_a, valid = distinct_ops(rng, table, op)
        put_inj(starts[0], op_a)
        s = np.array([OPS[o] for o in op]); y = table[ar, op]
        meta.update(op_sys=op, op_att=op_a)
    elif kind == "DEL_L":
        op_l = rng.integers(0, 4, size=B)
        put_legit(starts[0], op_l)
        s = np.full(B, DELEG); y = table[ar, op_l]
        meta.update(op_leg=op_l)
    elif kind == "DEL_A":
        op_a = rng.integers(0, 4, size=B)
        put_inj(starts[0], op_a)
        s = np.full(B, DELEG); y = np.full(B, NOOP_CLASS)
        meta.update(op_att=op_a)
    elif kind == "DEL_LA":
        op_l = rng.integers(0, 4, size=B)
        op_a, valid = distinct_ops(rng, table, op_l)
        put_legit(starts[0], op_l); put_inj(starts[1], op_a)
        s = np.full(B, DELEG); y = table[ar, op_l]
        meta.update(op_leg=op_l, op_att=op_a)
    if inband:
        roles[:] = 1
    x, r = assemble(s, toks, roles)
    meta["valid"] = valid
    meta["starts"] = starts
    return x, r, y, meta


# ---------------- model ----------------
class TinyTransformer(nn.Module):
    def __init__(self, d=64, heads=4, layers=2, ff=128, use_roles=True):
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d)
        self.pos = nn.Embedding(SEQ, d)
        self.use_roles = use_roles
        self.role = nn.Embedding(3, d)
        enc = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, NCLS)

    def forward(self, x, r):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))[None]
        if self.use_roles:
            h = h + self.role(r)
        h = self.enc(h)
        return self.head(self.ln(h[:, -1]))


def train_model(mix, steps=2500, B=256, seed=0, use_roles=True, q=1.0, inband=False,
                weights=None, lr=2e-3, log=False, templates=TRAIN_TEMPLATES):
    """mix: dict kind->probability. weights: dict kind->loss weight."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = TinyTransformer(use_roles=use_roles)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    kinds = list(mix.keys())
    p = np.array([mix[k] for k in kinds]); p = p / p.sum()
    weights = weights or {}
    for step in range(steps):
        counts = rng.multinomial(B, p)
        xs, rs, ys, ws = [], [], [], []
        for k, c in zip(kinds, counts):
            if c == 0:
                continue
            x, r, y, _ = make_batch(rng, c, k, q=q, inband=inband, templates=templates)
            xs.append(x); rs.append(r); ys.append(y); ws.append(np.full(c, weights.get(k, 1.0)))
        x = torch.tensor(np.concatenate(xs)); r = torch.tensor(np.concatenate(rs))
        y = torch.tensor(np.concatenate(ys)); w = torch.tensor(np.concatenate(ws), dtype=torch.float32)
        loss = (F.cross_entropy(m(x, r), y, reduction="none") * w).sum() / w.sum()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log and step % 500 == 0:
            print(step, float(loss.detach()))
    m.eval()
    return m


@torch.no_grad()
def predict(m, x, r, bs=8192):
    out = []
    for i in range(0, len(x), bs):
        out.append(m(torch.tensor(x[i:i + bs]), torch.tensor(r[i:i + bs])))
    return torch.cat(out).numpy()
