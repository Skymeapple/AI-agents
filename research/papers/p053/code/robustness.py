"""Architecture robustness: rerun the full analysis with (a) an LSTM (H=64) and (b) a small GRU (H=32).
Usage: python3 robustness.py   -> results/robustness_raw.json"""
import json, os, sys
import torch, torch.nn as nn
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
import emo
import run_experiments as rx

class LSTMNet(nn.Module):
    """LSTM wrapper exposing a GRU-like interface (h0 is hidden state; cell starts at 0)."""
    def __init__(self, H=64):
        super().__init__()
        self.lstm = nn.LSTM(emo.IN_DIM, H, batch_first=True); self.rep = nn.Linear(H, 3); self.act = nn.Linear(H, 2)
        outer = self
        class G:  # analyse() calls net.gru(x, h0); emulate it, carrying the cell state across calls via cache
            def __call__(self, x, h0=None):
                if h0 is None:
                    hs, (h, c) = outer.lstm(x); outer._cprefix = c
                else:
                    # steering perturbs the hidden state; the cell state is the one cached from the matching prefix
                    hs, (h, c) = outer.lstm(x, (h0, outer._cprefix))
                return hs, h
        self.gru = G()
    def forward(self, x, h0=None):
        hs, _ = self.lstm(x); return hs, self.rep(hs), self.act(hs)

ARCH = sys.argv[1] if len(sys.argv) > 1 else "all"

def job(arch, world, w, seed):
    torch.set_num_threads(1)
    if arch == "gru32":
        emo.Net = lambda: _gru(32)
    else:
        emo.Net = lambda: LSTMNet(64)
    r = rx.analyse(world, w, seed); r["arch"] = arch; return r

class _GRUNet(nn.Module):
    def __init__(self, H):
        super().__init__(); self.gru = nn.GRU(emo.IN_DIM, H, batch_first=True); self.rep = nn.Linear(H, 3); self.act = nn.Linear(H, 2)
    def forward(self, x, h0=None):
        hs, _ = self.gru(x, h0); return hs, self.rep(hs), self.act(hs)
def _gru(H): return _GRUNet(H)

if __name__ == "__main__":
    jobs = [(a, wd, w, s) for a in ("gru32", "lstm64") for wd, w in [("D", 0.0), ("X50", 0.5), ("G", 1.0), ("M", 0.0)] for s in (0, 1, 2)]
    with Pool(4) as p:
        out = p.starmap(job, jobs)
    json.dump(out, open(os.path.join(rx.OUT, "robustness_raw.json"), "w"), indent=1)
