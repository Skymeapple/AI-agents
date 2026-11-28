"""Torus gridworld with coloured cells; sequences interleave observations and moves.

Token layout for a walk of T steps:
    obs_0, m_1, obs_1, m_2, obs_2, ..., m_T, obs_T      (length 2T+1)
Moves are tokens 0..3 (U, D, L, R). Colours are tokens 4..4+C-1.
The latent world state at the position of token m_t is pos_t (the cell after the move);
the next token obs_t = colour(pos_t) is a deterministic function of that state.
"""
import numpy as np

MOVES = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])


class TorusWorld:
    def __init__(self, n=5, mode="balanced2", seed=0):
        """mode: 'unique' (every cell its own colour: state is 1-local),
        'balanced2' (two colours, balanced random colouring),
        'landmark' (three landmark cells among n*n; state is strongly non-local)."""
        rng = np.random.default_rng(1000 + seed)
        self.n = n
        self.S = n * n
        self.mode = mode
        if mode == "unique":
            n_colors = self.S
            self.colors = rng.permutation(self.S)
        elif mode == "balanced2":
            n_colors = 2
            self.colors = rng.permutation(np.arange(self.S) % 2)
        elif mode == "landmark":
            n_colors = 2
            self.colors = np.zeros(self.S, dtype=np.int64)
            self.colors[rng.choice(np.arange(1, self.S), 3, replace=False)] = 1
        else:
            raise ValueError(mode)
        self.C = n_colors
        self.vocab = 4 + n_colors
        # transition table: next_cell[s, a]
        self.next_cell = np.zeros((self.S, 4), dtype=np.int64)
        for s in range(self.S):
            r, c = divmod(s, n)
            for a in range(4):
                rr, cc = (r + MOVES[a, 0]) % n, (c + MOVES[a, 1]) % n
                self.next_cell[s, a] = rr * n + cc

    def sample(self, N, T, rng, start=0):
        moves = rng.integers(0, 4, size=(N, T))
        pos = np.zeros((N, T + 1), dtype=np.int64)
        pos[:, 0] = start
        for t in range(T):
            pos[:, t + 1] = self.next_cell[pos[:, t], moves[:, t]]
        obs = self.colors[pos]
        toks = np.zeros((N, 2 * T + 1), dtype=np.int64)
        toks[:, 0::2] = obs + 4
        toks[:, 1::2] = moves
        return toks, pos, moves, obs

    def suffix_bayes(self, moves, obs, t_idx, k):
        """Exact posterior over pos_t given only the last k (obs, move) pairs before obs_t,
        i.e. tokens obs_{t-k}, m_{t-k+1}, ..., obs_{t-1}, m_t, starting from the uniform
        (stationary) prior at time t-k. Returns posterior array (N, S).
        moves: (N,T) with moves[:, j-1] = m_j ; obs: (N,T+1)."""
        N = moves.shape[0]
        t = t_idx
        p = np.ones((N, self.S)) / self.S
        if k == 0:
            return p
        onehot_col = (self.colors[None, :] == obs[:, t - k][:, None]).astype(float)
        p = p * onehot_col
        for j in range(t - k + 1, t + 1):
            a = moves[:, j - 1]
            q = np.zeros_like(p)
            # push mass along transitions
            dest = self.next_cell[:, a].T  # (N, S): destination of each s under a_n
            np.put_along_axis(q, dest, p, axis=1)  # transitions are permutations
            p = q
            if j < t:
                p = p * (self.colors[None, :] == obs[:, j][:, None])
        p = p / p.sum(1, keepdims=True)
        return p

    def rollout_colors(self, start_cells, moves_future):
        """Colours seen when executing moves_future (N, r) from start_cells (N,).
        Returns (N, r+1): colour at start then after each move."""
        N, r = moves_future.shape
        cells = np.zeros((N, r + 1), dtype=np.int64)
        cells[:, 0] = start_cells
        for j in range(r):
            cells[:, j + 1] = self.next_cell[cells[:, j], moves_future[:, j]]
        return self.colors[cells]


def marginal_from_start(world, steps, start=0):
    """Exact distribution of pos_steps for a uniform random walk started at `start`."""
    p = np.zeros(world.S)
    p[start] = 1.0
    for _ in range(steps):
        q = np.zeros(world.S)
        for a in range(4):
            np.add.at(q, world.next_cell[:, a], p / 4)
        p = q
    return p


def suffix_bayes_exact(world, moves, obs, t, k, start=0):
    """Exact posterior P(pos_t | obs_{t-k}, m_{t-k+1}, ..., obs_{t-1}, m_t) under the TRUE
    data distribution (known start cell, uniform random moves): the prior at time t-k is the
    exact finite-time marginal of the walk, not the stationary distribution.  At k = t the
    window is the full history and the posterior is a point mass."""
    N = moves.shape[0]
    prior = marginal_from_start(world, t - k, start)
    p = np.tile(prior, (N, 1))
    if k == 0:
        return p
    p = p * (world.colors[None, :] == obs[:, t - k][:, None])
    for j in range(t - k + 1, t + 1):
        a = moves[:, j - 1]
        q = np.zeros_like(p)
        dest = world.next_cell[:, a].T
        np.put_along_axis(q, dest, p, axis=1)
        p = q
        if j < t:
            p = p * (world.colors[None, :] == obs[:, j][:, None])
    return p / p.sum(1, keepdims=True)


def exact_curve(world, mv, ob, ts, kmax=24):
    """Acc*_k averaged over ts, with the exact finite-time prior (Definition 1 of the paper).
    For k > t the window is truncated at the start of the sequence (full history)."""
    out = []
    for k in range(kmax + 1):
        out.append(float(np.mean([suffix_bayes_exact(world, mv, ob, t, min(k, t)).max(1).mean()
                                  for t in ts])))
    return out
