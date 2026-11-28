"""Minimal Othello engine (bitboards) and synthetic-game generator.

Synthetic games follow the "synthetic" distribution of Li et al. (2023): every move is drawn
uniformly at random from the legal moves; a player with no legal move passes; the game ends
when neither player can move. Squares are indexed 0..63 (row*8+col). The 60 non-centre
squares form the token vocabulary (ids 0..59); passes are not tokens.
"""
import numpy as np

FULL = (1 << 64) - 1
NOT_A = 0xFEFEFEFEFEFEFEFE  # clears column 0 after a shift that moves +1 column
NOT_H = 0x7F7F7F7F7F7F7F7F
DIRS = [(1, NOT_A), (-1, NOT_H), (8, FULL), (-8, FULL), (9, NOT_A), (7, NOT_H), (-7, NOT_A), (-9, NOT_H)]
CENTRE = [27, 28, 35, 36]
SQ2TOK = {}
TOK2SQ = []
for s in range(64):
    if s not in CENTRE:
        SQ2TOK[s] = len(TOK2SQ); TOK2SQ.append(s)
TOK2SQ = np.array(TOK2SQ)


def shift(b, d, mask):
    return ((b << d) & mask & FULL) if d > 0 else ((b >> -d) & mask)


def legal(me, op):
    empty = ~(me | op) & FULL
    moves = 0
    for d, mask in DIRS:
        x = shift(me, d, mask) & op
        for _ in range(5):
            x |= shift(x, d, mask) & op
        moves |= shift(x, d, mask) & empty
    return moves


def flips(me, op, sq):
    m = 1 << sq
    f = 0
    for d, mask in DIRS:
        x = 0
        y = shift(m, d, mask)
        while y & op:
            x |= y
            y = shift(y, d, mask)
        if y & me:
            f |= x
    return f


def bits(b):
    out = []
    while b:
        l = b & -b
        out.append(l.bit_length() - 1)
        b ^= l
    return out


def random_game(rng):
    black = (1 << 28) | (1 << 35)   # e4, d5
    white = (1 << 27) | (1 << 36)   # d4, e5
    me, op, player = black, white, 0    # player 0 = black to move
    moves, boards, changed = [], [], []
    while True:
        lm = legal(me, op)
        if not lm:
            if not legal(op, me):
                break
            me, op, player = op, me, 1 - player     # pass
            continue
        cand = bits(lm)
        sq = cand[rng.integers(len(cand))]
        f = flips(me, op, sq)
        me |= (1 << sq) | f
        op &= ~f
        me, op, player = op, me, 1 - player
        moves.append(sq)
        changed.append((1 << sq) | f)
        # board after the move, relative to the player who moves NEXT (after any forced passes)
        nme, nop = me, op
        if not legal(me, op) and legal(op, me):
            nme, nop = op, me
        boards.append((nme, nop))
    return moves, boards, changed


def board_labels(boards):
    """(L, 64) int8: 0 empty, 1 'mine' (player to move next), 2 'theirs'."""
    lab = np.zeros((len(boards), 64), np.int8)
    for i, (me, op) in enumerate(boards):
        for s in bits(me): lab[i, s] = 1
        for s in bits(op): lab[i, s] = 2
    return lab


def ages_of(changed):
    """(L, 64) int8: number of moves since the square last changed (placed or flipped);
    0 = changed by the current move; -1 = still empty. Centre squares count from move -1."""
    L = len(changed)
    last = np.full(64, -100, np.int32)
    last[CENTRE] = -1
    out = np.full((L, 64), -1, np.int8)
    for t, c in enumerate(changed):
        for s in bits(c): last[s] = t
        occ = last > -100
        out[t, occ] = np.minimum(t - last[occ], 127)
    return out


def generate(n, seed, with_labels=True):
    rng = np.random.default_rng(seed)
    L = 60
    toks = np.full((n, L), -1, np.int16)
    lens = np.zeros(n, np.int16)
    labs = np.zeros((n, L, 64), np.int8) if with_labels else None
    ages = np.full((n, L, 64), -1, np.int8) if with_labels else None
    for g in range(n):
        mv, bd, ch = random_game(rng)
        toks[g, :len(mv)] = [SQ2TOK[s] for s in mv]
        lens[g] = len(mv)
        if with_labels:
            labs[g, :len(mv)] = board_labels(bd)
            ages[g, :len(mv)] = ages_of(ch)
    return toks, labs, lens, ages


if __name__ == "__main__":
    import time, sys
    t0 = time.time(); t, l, n, a = generate(500, 0); print(time.time() - t0, n.mean(), n.min())
    # sanity: final disc counts sum to 4 + number of moves
    print(((l[np.arange(500), n - 1] > 0).sum(1) == n + 4).all())
    print(a[0, :5, 19:22], a[0, 59].max())
