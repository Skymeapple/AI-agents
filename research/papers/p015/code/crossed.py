"""Joint crossed agent x task frailty model (cloglog), fitted with PyTorch autograd.

  P(success of a run of agent a on task i) = exp(-exp(eta_ai)),
  eta_ai = mu_a + k_a * log(T_i/60) + lam_a * u_i + tau_a * w_ai,
  u_i ~ G (shared task difficulty; Gaussian or discrete grid with free weights),
  w_ai ~ N(0,1) (agent-specific task frailty, independent across pairs).

Optional "invalid task" latent class: with probability pi0 a task has p = 0 for every agent
(broken, mis-scored or mis-timed beyond reach).  The shared effect lam_a*u_i absorbs any error in
log T_i that is common to all agents (it enters as k_a*e_i, i.e. a shared task effect), so tau_a is
free of shared label noise.

Conditioning on u_i and w_ai, the task-level cumulative hazard is proportional to T^{k_a}.
"""
import numpy as np
import torch
from common import T_REF

torch.set_default_dtype(torch.float64)


def build_arrays(d, agents=None, tasks=None):
    agents = agents if agents is not None else sorted(d.alias.unique())
    tasks = tasks if tasks is not None else sorted(d.task_id.unique())
    ai = {a: j for j, a in enumerate(agents)}
    ti = {t: j for j, t in enumerate(tasks)}
    g = d[d.task_id.isin(ti) & d.alias.isin(ai)].groupby(["alias", "task_id"]).y.agg(["size", "sum"]).reset_index()
    n = np.zeros((len(agents), len(tasks)))
    s = np.zeros_like(n)
    n[g.alias.map(ai), g.task_id.map(ti)] = g["size"]
    s[g.alias.map(ai), g.task_id.map(ti)] = g["sum"]
    T = d.groupby("task_id").T.first().reindex(tasks).values
    return agents, tasks, n, s, T


class Crossed:
    def __init__(self, n, s, T, G="gauss", invalid=False, nu=25, nw=15, grid=None, common_k=False, fixed_k=None):
        self.common_k, self.fixed_k = common_k, fixed_k
        self.n = torch.tensor(n)
        self.s = torch.tensor(s)
        self.lt = torch.tensor(np.log(T / T_REF))
        self.A, self.I = n.shape
        self.G, self.invalid = G, invalid
        xw, ww = np.polynomial.hermite_e.hermegauss(nw)
        self.xw, self.lww = torch.tensor(xw), torch.tensor(np.log(ww / ww.sum()))
        if G == "gauss":
            xu, wu = np.polynomial.hermite_e.hermegauss(nu)
            self.xu, self.lwu0 = torch.tensor(xu), torch.tensor(np.log(wu / wu.sum()))
        else:
            self.xu = torch.tensor(grid if grid is not None else np.linspace(-4, 4, 21))
        # parameter vector layout
        A = self.A
        self.sl = dict(mu=slice(0, A), k=slice(A, 2 * A), loglam=slice(2 * A, 3 * A), logtau=slice(3 * A, 4 * A))
        m = 4 * A
        if G != "gauss":
            self.sl["wlog"] = slice(m, m + len(self.xu) - 1)
            m += len(self.xu) - 1
        if invalid:
            self.sl["logit_pi0"] = slice(m, m + 1)
            m += 1
        self.npar = m

    def init(self, mu=None, k=None):
        th = torch.zeros(self.npar)
        th[self.sl["mu"]] = torch.tensor(mu) if mu is not None else 0.0
        th[self.sl["k"]] = torch.tensor(k) if k is not None else 1.0
        th[self.sl["loglam"]] = np.log(2.0)
        th[self.sl["logtau"]] = np.log(1.0)
        if self.invalid:
            th[self.sl["logit_pi0"]] = -3.0
        return th

    def lwu(self, th):
        if self.G == "gauss":
            return self.lwu0
        z = torch.cat([torch.zeros(1), th[self.sl["wlog"]]])
        return z - torch.logsumexp(z, 0)

    def task_ll(self, th):
        """Per-task marginal log-likelihood (length I)."""
        mu, k = th[self.sl["mu"]], th[self.sl["k"]]
        if self.fixed_k is not None:
            k = torch.full_like(k, float(self.fixed_k))
        elif self.common_k:
            k = k[:1].expand_as(k)
        lam, tau = torch.exp(th[self.sl["loglam"]]), torch.exp(th[self.sl["logtau"]])
        # eta[a, i, u, w]
        eta = (mu[:, None] + k[:, None] * self.lt[None, :])[:, :, None, None] \
            + (lam[:, None] * self.xu[None, :])[:, None, :, None] \
            + (tau[:, None] * self.xw[None, :])[:, None, None, :]
        e = torch.exp(torch.clamp(eta, -30, 6))
        lp = -e
        lq = torch.log(-torch.expm1(-e) + 1e-300)
        s, n = self.s[:, :, None, None], self.n[:, :, None, None]
        ll = s * lp + (n - s) * lq
        lai = torch.logsumexp(ll + self.lww, dim=3)  # [a, i, u]
        li = torch.logsumexp(lai.sum(0) + self.lwu(th)[None, :], dim=1)  # [i]
        if self.invalid:
            pi0 = torch.sigmoid(th[self.sl["logit_pi0"]])[0]
            allzero = (self.s.sum(0) == 0)
            li = torch.where(allzero, torch.logaddexp(torch.log1p(-pi0) + li, torch.log(pi0).expand_as(li)),
                             torch.log1p(-pi0) + li)
        return li

    def nll(self, th):
        return -self.task_ll(th).sum()

    def fit(self, th0, iters=400, outer=4):
        th = th0.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([th], lr=1, max_iter=iters, tolerance_grad=1e-7, tolerance_change=1e-11,
                                history_size=50, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            l = self.nll(th)
            l.backward()
            return l
        for _ in range(outer):
            opt.step(closure)
        return th.detach(), float(self.nll(th.detach()))

    def task_posterior_invalid(self, th):
        """Posterior probability that each task is in the invalid class."""
        if not self.invalid:
            return None
        with torch.no_grad():
            pi0 = torch.sigmoid(th[self.sl["logit_pi0"]])[0]
            inv = self.invalid
            self.invalid = False
            lc = self.task_ll(th)
            self.invalid = inv
            allzero = (self.s.sum(0) == 0)
            post = torch.where(allzero, pi0 / (pi0 + (1 - pi0) * torch.exp(lc)), torch.zeros_like(lc))
        return post.numpy()

    def wald_se(self, th, keys=("k",)):
        H = torch.autograd.functional.hessian(self.nll, th)
        cov = torch.linalg.pinv(H)
        se = torch.sqrt(torch.clamp(torch.diag(cov), min=0))
        return {key: se[self.sl[key]].numpy() for key in keys}
