"""Revision (R1): train an ensemble for one (variant, world) and cache member predictions + member-0 hidden states.
Usage: python3 train_cache.py <variant> <world_seed>   ->  ../cache/<variant>_w<seed>.npz
Variants (all use the same world seeds and member seeds 100*seed+m as run_all.py):
  base          original world & model (30 epochs)                     -- reproduces run_all.py member predictions
  early         original world, 3 epochs (imperfect, familiarity-correlated memorization)
  small         original world, width 32, emb 16, 10 epochs (capacity-limited)
  comp          compositional names (2 sub-tokens, one tied to type), 20 epochs
  matched       shape-matched world (unknown-deterministic and known-aleatoric share the same Bayes predictive), 30 epochs
  matched_early shape-matched world, 3 epochs
  matched_ls    shape-matched world, label smoothing 0.2, 10 epochs"""
import numpy as np, sys, os, time
import world as W, mlp
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache')
VARIANTS = {
    'base': (dict(), dict(epochs=30)),
    'early': (dict(), dict(epochs=3)),
    'small': (dict(), dict(epochs=10, h=32, d=16)),
    'comp': (dict(comp_names=True), dict(epochs=20, comp=True)),
    'matched': (dict(matched=True), dict(epochs=30)),
    'matched_early': (dict(matched=True), dict(epochs=3)),
    'matched_ls': (dict(matched=True), dict(epochs=10, ls=0.2)),
}
NM = 5
if __name__ == '__main__':
    var, ws = sys.argv[1], int(sys.argv[2]); t0 = time.time()
    wkw, tkw = VARIANTS[var]
    w = W.make_world(seed=ws, **wkw); nE, nR = w['n_ent'], w['n_rel']
    e = np.repeat(np.arange(nE), nR); r = np.tile(np.arange(nR), nE)
    Ps = []
    for m in range(NM):
        net = mlp.train(w, seed=100 * ws + m, **tkw)
        P, H = net.predict(e, r); Ps.append(P.astype(np.float32))
        if m == 0: H0 = H.astype(np.float16)
    np.savez_compressed(f'{CACHE}/{var}_w{ws}.npz', P=np.stack(Ps), H0=H0)
    print(f'{var} w{ws} done in {time.time()-t0:.0f}s', flush=True)
