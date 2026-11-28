"""Extra seeds for E6 at selected task-diversity levels: python -m wu.e6_seeds M seed"""
import sys

from .common import save_json
from .e6_incontext import run_one

if __name__ == "__main__":
    M = None if sys.argv[1] == "inf" else int(sys.argv[1])
    seed = int(sys.argv[2])
    r = run_one(M, seed=seed)
    save_json(f"e6_extra_M{sys.argv[1]}_seed{seed}", r)
