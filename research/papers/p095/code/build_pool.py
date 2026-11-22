"""Build a pool of 'research candidates' on real data (CPU only).

Each candidate is a randomly proposed pipeline modification (MLP architecture,
optimiser and preprocessing choices). For every candidate we train it with
SEEDS independent seeds; each seed draws its own training subsample and
initialisation. We store
  * val_correct[c, s, :]  boolean correctness on a fixed validation pool
  * test_acc[c, s]        accuracy on a large disjoint test set
so that downstream experiments can replay any "evaluate N candidates with
m runs each" research protocol without retraining.

Usage: python3 build_pool.py <dataset> [n_configs] [n_seeds]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, gzip, time, warnings, json
import numpy as np
from multiprocessing import Pool
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, QuantileTransformer, MinMaxScaler
from sklearn.exceptions import ConvergenceWarning

ROOT = "/home/user/AI-agents/research/papers/p095"
N_TRAIN = 2000          # training subsample per run (data-sampling noise)
N_TRAINPOOL, N_VALPOOL = 8000, 4000


def load(name):
    with gzip.open(f"{ROOT}/data/{name}.tsv.gz", "rt") as f:
        header = f.readline().strip().split("\t")
        arr = np.loadtxt(f, delimiter="\t")
    X, y = arr[:, :-1], arr[:, -1].astype(int)
    assert header[-1] == "target"
    rng = np.random.default_rng(12345)
    p = rng.permutation(len(y))
    return X[p], y[p]


def sample_config(rng):
    depth = rng.integers(1, 4)
    width = int(2 ** rng.uniform(3, 8))          # 8 .. 256
    return dict(
        hidden=tuple([width] * depth),
        activation=str(rng.choice(["relu", "tanh", "logistic"])),
        lr=float(10 ** rng.uniform(-4, -1)),
        alpha=float(10 ** rng.uniform(-6, 0)),
        batch=int(rng.choice([32, 64, 128, 256])),
        max_iter=int(rng.choice([5, 10, 20, 40])),
        scaler=str(rng.choice(["standard", "quantile", "minmax", "none"])),
    )


def run_one(args):
    cfg, seed, data = args
    Xtr_pool, ytr_pool, Xva, yva, Xte, yte = data
    rng = np.random.default_rng(1000 + seed)
    idx = rng.choice(len(ytr_pool), N_TRAIN, replace=False)
    Xtr, ytr = Xtr_pool[idx], ytr_pool[idx]
    sc = {"standard": StandardScaler(), "quantile": QuantileTransformer(n_quantiles=200),
          "minmax": MinMaxScaler(), "none": None}[cfg["scaler"]]
    if sc is not None:
        sc.fit(Xtr); Xtr_, Xva_, Xte_ = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    else:
        Xtr_, Xva_, Xte_ = Xtr, Xva, Xte
    clf = MLPClassifier(hidden_layer_sizes=cfg["hidden"], activation=cfg["activation"],
                        learning_rate_init=cfg["lr"], alpha=cfg["alpha"],
                        batch_size=cfg["batch"], max_iter=cfg["max_iter"],
                        random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            clf.fit(Xtr_, ytr)
            pv, pt = clf.predict(Xva_), clf.predict(Xte_)
        except Exception:
            # diverged / numerical failure counts as a failed idea: predict majority class
            maj = np.bincount(ytr).argmax()
            pv, pt = np.full(len(yva), maj), np.full(len(yte), maj)
    return (pv == yva), float(np.mean(pt == yte))


def main():
    name = sys.argv[1]
    n_cfg = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    n_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    X, y = load(name)
    a, b = N_TRAINPOOL, N_TRAINPOOL + N_VALPOOL
    data = (X[:a], y[:a], X[a:b], y[a:b], X[b:], y[b:])
    rng = np.random.default_rng(7)
    cfgs = [sample_config(rng) for _ in range(n_cfg)]
    jobs = [(c, s, data) for c in cfgs for s in range(n_seed)]
    t0 = time.time()
    with Pool(4) as p:
        out = p.map(run_one, jobs, chunksize=4)
    val = np.array([o[0] for o in out]).reshape(n_cfg, n_seed, -1)
    test = np.array([o[1] for o in out]).reshape(n_cfg, n_seed)
    np.savez_compressed(f"{ROOT}/results/pool_{name}.npz", val_correct=val, test_acc=test)
    with open(f"{ROOT}/results/pool_{name}_configs.json", "w") as f:
        json.dump(cfgs, f)
    print(f"{name}: {n_cfg} configs x {n_seed} seeds, test set {len(y)-b}, "
          f"time {time.time()-t0:.0f}s, mean test acc {test.mean():.4f}, "
          f"best config mean {test.mean(1).max():.4f}")


if __name__ == "__main__":
    main()
