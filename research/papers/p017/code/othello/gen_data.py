"""Generate synthetic Othello datasets (seeded) into ../../results/othello/data_*.npz."""
import sys, os, numpy as np
from engine import generate
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "othello")
os.makedirs(OUT, exist_ok=True)
name, n, seed, lab = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4] == "1"
t, l, n_, a = generate(n, seed, lab)
if lab:
    np.savez_compressed(os.path.join(OUT, f"data_{name}.npz"), toks=t, labs=l, lens=n_, ages=a)
else:
    np.savez_compressed(os.path.join(OUT, f"data_{name}.npz"), toks=t, lens=n_)
print(name, "done")
