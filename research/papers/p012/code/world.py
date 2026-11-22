"""Synthetic knowledge world with controllable familiarity, coverage and aleatoric ambiguity."""
import numpy as np

def make_world(seed=0, n_ent=4000, n_rel=8, n_alea=2, n_val=32, n_types=8, rho=0.6,
               coverage=0.75, frac_unseen=0.10, zipf=1.0, n_train=150_000):
    rng = np.random.default_rng(seed)
    types = rng.integers(0, n_types, n_ent)
    default = rng.integers(0, n_val, (n_types, n_rel))
    alea = np.zeros(n_rel, bool); alea[n_rel - n_alea:] = True
    # ground-truth conditional p*(.|e,r)
    pstar = np.zeros((n_ent, n_rel, n_val))
    for e in range(n_ent):
        for r in range(n_rel):
            top = default[types[e], r] if rng.random() < rho else rng.integers(n_val)
            if not alea[r]:
                pstar[e, r, top] = 1.0
            else:
                others = rng.choice(np.setdiff1d(np.arange(n_val), [top]), 2, replace=False)
                pstar[e, r, top] = 0.5; pstar[e, r, others[0]] = 0.3; pstar[e, r, others[1]] = 0.2
    documented = rng.random((n_ent, n_rel)) < coverage
    documented[np.arange(n_ent), rng.integers(0, n_rel, n_ent)] = True  # >=1 documented rel
    unseen = rng.random(n_ent) < frac_unseen
    ranks = rng.permutation((~unseen).sum()) + 1
    pop = np.zeros(n_ent); pop[~unseen] = ranks ** (-zipf); pop /= pop.sum()
    # sample training corpus
    es = rng.choice(n_ent, n_train, p=pop)
    rs = np.empty(n_train, int)
    for i, e in enumerate(es):
        rs[i] = rng.choice(np.flatnonzero(documented[e]))
    cum = pstar[es, rs].cumsum(1)
    ys = (rng.random(n_train)[:, None] > cum).sum(1)
    counts = np.zeros((n_ent, n_rel), int)
    np.add.at(counts, (es, rs), 1)
    return dict(types=types, default=default, alea=alea, pstar=pstar, documented=documented,
                unseen=unseen, pop=pop, es=es, rs=rs, ys=ys, counts=counts,
                ent_count=counts.sum(1), n_ent=n_ent, n_rel=n_rel, n_val=n_val, rho=rho, seed=seed)
