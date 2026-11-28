"""Fast checks of the pieces the conclusions rest on: the Bayes-optimal
reference, the baselines, and the estimators."""
import numpy as np

from wu.common import powerlaw_fit, twonn_id
from wu.e1_world_model import EgoGrid, ngram_nll
from wu.e6_incontext import DX, K, SIGMA, dmmse_pred, make_batch, ridge_pred


def test_world_is_hidden_markov_and_filter_is_consistent():
    w = EgoGrid(6, 3, 6, 3)
    rng = np.random.default_rng(0)
    toks, states = w.sample(200, 15, rng)
    p_obs, prior = w.bayes(toks)
    # the true state always has positive belief, and observations are never "impossible"
    for t in range(16):
        assert (prior[np.arange(200), t, states[:, t]] > 0).all()
    assert (p_obs > 0).all() and (p_obs <= 1 + 1e-9).all()
    # observations are deterministic given the state
    assert ((toks[:, 1::2] - 4) == w.obs_of[states]).all()


def test_ngram_cannot_beat_bayes():
    w = EgoGrid(6, 3, 6, 3)
    rng = np.random.default_rng(1)
    tr, _ = w.sample(5000, 10, rng)
    te, _ = w.sample(500, 10, rng)
    p, _ = w.bayes(te)
    bayes = -np.log(p[:, 1:]).mean()
    for n in (1, 3, 5):
        assert ngram_nll(tr, te, n, w.vocab, w.n_obs) >= bayes - 0.02


def test_twonn_recovers_dimension_of_embedded_manifold():
    rng = np.random.default_rng(0)
    z = rng.uniform(size=(3000, 3))
    X = np.tanh(z @ rng.normal(size=(3, 40)) * 2) @ rng.normal(size=(40, 50))
    assert abs(twonn_id(X) - 3) < 0.5


def test_powerlaw_fit_exact():
    x = np.array([1, 10, 100.0])
    s, c, r2 = powerlaw_fit(x, 3 * x ** -0.7)
    assert abs(s + 0.7) < 1e-9 and r2 > 0.999


def test_bayes_estimators():
    rng = np.random.default_rng(0)
    X, Y, W = make_batch(2000, rng)
    mse_ridge = ((ridge_pred(X, Y, SIGMA ** 2) - Y) ** 2).mean(0)
    assert mse_ridge[-1] < 0.2 and mse_ridge[-1] < mse_ridge[1]
    pool = rng.normal(size=(4, DX))
    X, Y, _ = make_batch(2000, rng, pool)
    d = ((dmmse_pred(X, Y, pool) - Y) ** 2).mean(0)
    r = ((ridge_pred(X, Y, SIGMA ** 2) - Y) ** 2).mean(0)
    assert d[3] < r[3]  # on its own prior, the pool-Bayes estimator wins early
