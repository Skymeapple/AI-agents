"""Core model: learning with an AI assistant (offloading model).

State  s_t in [0,1]   : unassisted skill
Action a_t in [0,1]   : share of the practice task offloaded to the AI
Practice score        : P_t = (1-a_t) s_t + a_t q             (q = AI quality on the task)
Learning              : s_{t+1} = s_t + eta * g_t * (1 - s_t),
                        g_t = (1-a_t) + a_t * kappa(s_t)       (kappa = retained generative engagement)
kappa(s) = k0 - k1*s  (k1>0 encodes expertise reversal; k1=0 constant kappa)

Student objective (course of T practice periods):
   sum_t [ (1-w)/T * P_t - c*(1-a_t) ]  +  beta*(w + L) * s_T
   w    : weight of the proctored (AI-free) exam in the course grade
   L    : long-run value of skill (in units of course grade)
   beta : present-bias / salience discount on terminal skill
   c    : per-period effort cost of doing the work oneself
Planner objective: maximise s_T (equivalently L*s_T).
"""
import numpy as np

S_GRID = np.linspace(0.0, 1.0, 401)
A_GRID = np.linspace(0.0, 1.0, 21)


def kappa_fn(s, k0, k1=0.0):
    return k0 - k1 * s


def step(s, a, eta, k0, k1=0.0):
    g = (1 - a) + a * kappa_fn(s, k0, k1)
    g = np.maximum(g, 0.0)
    return np.clip(s + eta * g * (1 - s), 0.0, 1.0)


def practice(s, a, q):
    return (1 - a) * s + a * q


def solve_student(T, q, eta, k0, k1=0.0, w=0.3, L=1.0, beta=0.5, c=0.02,
                  s_grid=S_GRID, a_grid=A_GRID, allowed=None):
    """Backward induction. Returns policy array pol[t, i] = optimal a at s_grid[i].
    allowed(t, s) -> bool array mask of permitted a values (optional constraint)."""
    V = beta * (w + L) * s_grid
    pol = np.zeros((T, len(s_grid)))
    S, A = np.meshgrid(s_grid, a_grid, indexing="ij")  # (ns, na)
    for t in range(T - 1, -1, -1):
        Sn = step(S, A, eta, k0, k1)
        cont = np.interp(Sn, s_grid, V)
        flow = (1 - w) / T * practice(S, A, q) - c * (1 - A)
        tot = flow + cont
        if allowed is not None:
            mask = allowed(t, S, A)
            tot = np.where(mask, tot, -np.inf)
        # tie-break toward less offloading (small penalty)
        tot = tot - 1e-12 * A
        idx = np.argmax(tot, axis=1)
        pol[t] = a_grid[idx]
        V = tot[np.arange(len(s_grid)), idx]
    return pol


def simulate(s0, pol_or_fn, T, q, eta, k0, k1=0.0, s_grid=S_GRID):
    """Simulate one or many students (s0 array) under a policy array or a function (t,s)->a."""
    s = np.atleast_1d(np.asarray(s0, float)).copy()
    P = np.zeros((T, s.size)); A = np.zeros((T, s.size)); Sp = np.zeros((T + 1, s.size))
    Sp[0] = s
    for t in range(T):
        if callable(pol_or_fn):
            a = pol_or_fn(t, s)
        else:
            idx = np.clip(np.searchsorted(s_grid, s), 0, len(s_grid) - 1)
            # nearest grid point
            idx2 = np.clip(idx - 1, 0, len(s_grid) - 1)
            pick = np.where(np.abs(s_grid[idx2] - s) < np.abs(s_grid[idx] - s), idx2, idx)
            a = pol_or_fn[t, pick]
        a = np.broadcast_to(a, s.shape).astype(float)
        P[t] = practice(s, a, q); A[t] = a
        s = step(s, a, eta, k0, k1)
        Sp[t + 1] = s
    return dict(P=P, A=A, S=Sp, sT=s, Pbar=P.mean(0))
