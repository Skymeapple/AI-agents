"""Predicting regulatory DNA activity (lentiMPRA) in three human cell lines.

Public proxy for HFM's "Predicting human regulatory DNA activity with Evo 2 7B and
lentiMPRA" track. Data: Agarwal et al., "Massively parallel characterization of
transcriptional regulatory elements" (Zenodo record 10558183, CC-BY 4.0), the
per-cell tables in human_legnet-main/datasets/original/*_averaged.tsv: 230 bp
constructs (200 bp insert plus fixed adapters) with mean log2 RNA/DNA activity in
HepG2, K562 and WTC11, and the authors' ten folds.

It tests two ideas from research/human-frontier-mission.md section 4.1, with a small
CNN trained from scratch on CPU (no Evo 2, so absolute scores are far below HFM's):

  1. one multi-task model with a shared trunk and one head per cell line, against
     one model per cell line;
  2. averaging the prediction over the insert and its reverse complement.

Leakage control: the three libraries share sequences, and the authors' folds differ
between cell lines. Every sequence that is in *any* cell line's test fold is removed
from *all* training sets, otherwise a multi-task model would see test sequences
through another cell line's labels.

Fold 10 is the test fold, folds 1-8 train (at most 40,000 sequences per cell line
to fit the CPU budget). Two seeds per model.

Usage: python research/hfm-experiments/mpra_multitask.py <dir-with-*_averaged.tsv>
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

CELLS = ["HepG2", "K562", "WTC11"]
ADAPT = 15
MAX_TRAIN = int(os.environ.get("MPRA_MAX_TRAIN", 40000))   # sequences per cell line
EPOCHS = int(os.environ.get("MPRA_EPOCHS", 3))
SEEDS = tuple(range(int(os.environ.get("MPRA_SEEDS", 2))))
LUT = np.full(256, 4, np.int64)
for i, ch in enumerate("ACGT"):
    LUT[ord(ch)] = i


def encode(seqs):
    idx = LUT[np.frombuffer("".join(seqs).encode(), np.uint8)].reshape(len(seqs), -1)
    return torch.from_numpy(idx)


def onehot(idx):
    return torch.nn.functional.one_hot(idx.clamp(max=3), 4).float().transpose(1, 2) * (idx < 4).unsqueeze(1)


def revcomp_insert(idx):
    """Reverse-complement the 200 bp insert, leaving the adapters in place."""
    out = idx.clone()
    ins = idx[:, ADAPT:-ADAPT].flip(1)
    out[:, ADAPT:-ADAPT] = torch.where(ins < 4, 3 - ins, ins)
    return out


class Net(nn.Module):
    def __init__(self, heads):
        super().__init__()
        def block(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2))
        self.trunk = nn.Sequential(block(4, 96, 9), block(96, 128, 7), block(128, 160, 5), nn.Dropout(0.1))
        self.fc = nn.Sequential(nn.Linear(320, 192), nn.ReLU())
        self.out = nn.Linear(192, heads)

    def forward(self, idx):
        h = self.trunk(onehot(idx))
        return self.out(self.fc(torch.cat([h.mean(2), h.amax(2)], 1)))


def train(X, Y, M, heads, seed):
    """Y, M: (N, heads) targets and mask of which heads have a label."""
    torch.manual_seed(seed)
    net = Net(heads)
    steps = EPOCHS * (len(X) // 256)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=steps)
    g = torch.Generator().manual_seed(seed)
    net.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(len(X), generator=g)
        for b in range(len(X) // 256):
            i = perm[b * 256:(b + 1) * 256]
            x = X[i]
            flip = torch.rand(len(i), generator=g) < 0.5  # reverse-complement augmentation
            x = torch.where(flip.unsqueeze(1), revcomp_insert(x), x)
            loss = (((net(x) - Y[i]) ** 2) * M[i]).sum() / M[i].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    return net.eval()


@torch.no_grad()
def predict(net, X, rc):
    out = []
    for b in range(0, len(X), 1024):
        x = X[b:b + 1024]
        p = net(x)
        if rc:
            p = 0.5 * (p + net(revcomp_insert(x)))
        out.append(p)
    return torch.cat(out).numpy()


def main():
    torch.set_num_threads(4)
    d = sys.argv[1]
    tabs = {}
    for c in CELLS:
        t = pd.read_csv(os.path.join(d, f"{c}_averaged.tsv"), sep="\t")
        t = t[t.rev == 0].drop_duplicates("seq")
        t["z"] = (t.mean_value - t.mean_value.mean()) / t.mean_value.std()
        tabs[c] = t
    test_seqs = set().union(*[set(t.seq[t.fold == 10]) for t in tabs.values()])
    rng = np.random.default_rng(0)
    train_t, test_t = {}, {}
    for c, t in tabs.items():
        tr = t[(t.fold <= 8) & ~t.seq.isin(test_seqs)]
        train_t[c] = tr.iloc[rng.permutation(len(tr))[:MAX_TRAIN]]
        test_t[c] = t[t.fold == 10]
        print(f"{c}: train {len(train_t[c])}, test {len(test_t[c])}", flush=True)

    # multi-task table: union of training sequences, one column per cell line, masked where unmeasured
    allseq = pd.concat([train_t[c][["seq"]] for c in CELLS]).drop_duplicates()
    for c in CELLS:
        allseq = allseq.merge(train_t[c][["seq", "z"]].rename(columns={"z": c}), on="seq", how="left")
    Xm = encode(allseq.seq.tolist())
    Ym = torch.tensor(allseq[CELLS].fillna(0).values, dtype=torch.float32)
    Mm = torch.tensor(allseq[CELLS].notna().values, dtype=torch.float32)
    print(f"multi-task training sequences: {len(allseq)} (labels per cell line: {Mm.sum(0).int().tolist()})", flush=True)

    rows = []
    for seed in SEEDS:
        t0 = time.time()
        multi = train(Xm, Ym, Mm, 3, seed)
        print(f"  seed {seed}: multi-task trained in {time.time() - t0:.0f}s", flush=True)
        for j, c in enumerate(CELLS):
            t0 = time.time()
            Xs = encode(train_t[c].seq.tolist())
            Ys = torch.tensor(train_t[c].z.values, dtype=torch.float32).unsqueeze(1)
            single = train(Xs, Ys, torch.ones_like(Ys), 1, seed)
            Xt, yt = encode(test_t[c].seq.tolist()), test_t[c].z.values
            for name, net, col in (("single-task", single, 0), ("multi-task", multi, j)):
                for rc in (False, True):
                    rows.append(dict(seed=seed, cell=c, model=name, rc=rc, rho=spearmanr(predict(net, Xt, rc)[:, col], yt)[0]))
            print(f"  seed {seed}: {c} single-task trained in {time.time() - t0:.0f}s", flush=True)
    r = pd.DataFrame(rows)
    tab = r.pivot_table(index=["model", "rc"], columns="cell", values="rho")
    tab["macro"] = tab[CELLS].mean(axis=1)
    per_seed = r.pivot_table(index=["model", "rc"], columns="seed", values="rho")
    print("\nSpearman correlation on the test fold (mean over seeds); rc = reverse-complement averaging")
    print(tab.round(4).to_string())
    print("\nmacro Spearman by seed")
    print(per_seed.round(4).to_string())


if __name__ == "__main__":
    main()
