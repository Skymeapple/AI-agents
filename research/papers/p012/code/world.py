"""Synthetic knowledge world with controllable familiarity, coverage and aleatoric ambiguity."""
import numpy as np

def make_world(seed=0, n_ent=4000, n_rel=8, n_alea=2, n_val=32, n_types=8, rho=0.6,
               coverage=0.75, frac_unseen=0.10, zipf=1.0, n_train=150_000,
               matched=False, comp_names=False, n_sub=300, type_bind=0.8):
    """matched=True: deterministic values are drawn once per entity from a type-specific 3-value prior
    (0.5/0.3/0.2), and aleatoric facts have p* equal to that same prior (so an ideal Bayesian learner's
    predictive distribution is identical for 'unknown deterministic' and 'known aleatoric' facts: delta=0).
    comp_names=True: each entity name is a pair of sub-tokens (A,B) from a vocabulary of n_sub; A is tied
    to the entity's latent type with prob type_bind. Unseen names are then composed of trained sub-tokens."""
    rng = np.random.default_rng(seed)
    types = rng.integers(0, n_types, n_ent)
    default = rng.integers(0, n_val, (n_types, n_rel))
    alea = np.zeros(n_rel, bool); alea[n_rel - n_alea:] = True
    # ground-truth conditional p*(.|e,r)
    pstar = np.zeros((n_ent, n_rel, n_val))
    if matched:
        cand = np.stack([rng.choice(n_val, 3, replace=False) for _ in range(n_types * n_rel)]).reshape(n_types, n_rel, 3)
        default = cand[:, :, 0]
        prior3 = np.array([0.5, 0.3, 0.2])
    for e in range(n_ent):
        for r in range(n_rel):
            if matched:
                cc = cand[types[e], r]
                if alea[r]: pstar[e, r, cc] = prior3
                else: pstar[e, r, cc[rng.choice(3, p=prior3)]] = 1.0
                continue
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
    ent_tok = None
    if comp_names:
        r2 = np.random.default_rng(seed + 7919)
        home = np.arange(n_sub) % n_types          # each A-token has a home type
        used, ent_tok = set(), np.zeros((n_ent, 2), int)
        for e in range(n_ent):
            while True:
                if r2.random() < type_bind: A = r2.choice(np.flatnonzero(home == types[e]))
                else: A = r2.integers(n_sub)
                B = n_sub + r2.integers(n_sub)
                if (A, B) not in used: used.add((A, B)); break
            ent_tok[e] = (A, B)
    prior = None
    if matched:
        prior = np.zeros((n_types, n_rel, n_val))
        for t in range(n_types):
            for r in range(n_rel): prior[t, r, cand[t, r]] = prior3
    else:
        prior = np.full((n_types, n_rel, n_val), (1 - rho) / n_val)
        for t in range(n_types):
            for r in range(n_rel): prior[t, r, default[t, r]] += rho
    return dict(ent_tok=ent_tok, prior=prior, matched=matched, types=types, default=default, alea=alea, pstar=pstar, documented=documented,
                unseen=unseen, pop=pop, es=es, rs=rs, ys=ys, counts=counts,
                ent_count=counts.sum(1), n_ent=n_ent, n_rel=n_rel, n_val=n_val, rho=rho, seed=seed)
