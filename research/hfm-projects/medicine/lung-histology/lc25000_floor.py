"""Lung adenocarcinoma against squamous cell carcinoma on public patches, on CPU, with a
check for near-duplicate leakage.

Stand-in for HFM's "Patient-level lung cancer histology classification from H&E slides
with Phikon-v2" track, which needs a GPU and hundreds of GB of whole-slide images. The
public LC25000 set (Borkowski et al. 2019) has 5,000 lung adenocarcinoma, 5,000 squamous
cell carcinoma and 5,000 benign lung patches. Its authors made them by augmenting 250
original images per class (rotations and flips), so a random train/test split puts
copies of the same picture on both sides. Published accuracies near 100% may be measuring
that.

Steps
  1. Features: an ImageNet ResNet-18 (torchvision), averaged over the eight rotations and
     flips of each patch so that augmented copies of one original map close together.
  2. Near-duplicate groups: link patches whose invariant features have cosine
     similarity above a threshold, and take connected components.
  3. Logistic regression, adenocarcinoma against squamous (and three classes), scored by
     five-fold cross-validation twice: random folds, and folds that keep each
     near-duplicate group together.

Usage: python lc25000_floor.py <out dir> [hf dataset id]
"""
import io
import json
import os
import sys
import urllib.request

import numpy as np

AGENT = {"User-Agent": "research-script"}


def files(repo):
    with urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/api/datasets/{repo}/parquet", headers=AGENT), timeout=120) as r:
        listing = json.load(r)
    urls = []
    for config in listing.values():
        for split in config.values():
            urls += split
    return urls


def load(repo, cache):
    import pyarrow.parquet as pq
    from PIL import Image
    imgs, labels = [], []
    names = None
    for i, url in enumerate(files(repo)):
        path = os.path.join(cache, f"part{i}.parquet")
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)
        t = pq.read_table(path)
        if i == 0:
            print("schema:", t.schema, flush=True)
            try:
                meta = json.loads(t.schema.metadata[b"huggingface"])
                names = meta["info"]["features"]["label"]["names"]
            except Exception:
                names = None
        cols = t.column_names
        img_col = next(c for c in cols if "image" in c.lower())
        lab_col = next(c for c in cols if c != img_col)
        for cell, lab in zip(t.column(img_col).to_pylist(), t.column(lab_col).to_pylist()):
            name = names[lab] if names and isinstance(lab, int) else str(lab)
            if not name.startswith("lung"):
                continue
            im = Image.open(io.BytesIO(cell["bytes"])).convert("RGB").resize((224, 224))
            imgs.append(np.asarray(im))
            labels.append(name)
        print(f"  {url.split('/')[-1]}: lung patches so far {len(imgs)}", flush=True)
    return np.stack(imgs), np.array(labels)


def embed(imgs):
    import torch
    import torchvision
    torch.set_num_threads(4)
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = torch.nn.Identity()
    net.eval()
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    out = []
    with torch.no_grad():
        for s in range(0, len(imgs), 64):
            x = (torch.from_numpy(imgs[s:s + 64]).permute(0, 3, 1, 2).float() / 255 - mean) / std
            views = [torch.rot90(v, k, (2, 3)) for v in (x, x.flip(3)) for k in range(4)]
            out.append(torch.stack([net(v) for v in views]).mean(0).numpy())
            if s % 1280 == 0:
                print(f"  embedded {s + len(x)} of {len(imgs)}", flush=True)
    return np.concatenate(out)


def groups(z, thr):
    zn = z / np.linalg.norm(z, axis=1, keepdims=True)
    parent = list(range(len(z)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for s in range(0, len(z), 1000):
        sim = zn[s:s + 1000] @ zn.T
        for i, j in zip(*np.nonzero(sim > thr)):
            a, b = find(s + i), find(j)
            if a != b:
                parent[a] = b
    return np.array([find(i) for i in range(len(z))])


def evaluate(X, y, grp, folds, rng):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if grp is None:
        fold = rng.permutation(len(y)) % folds
    else:
        ug = rng.permutation(np.unique(grp))
        fmap = {g: i % folds for i, g in enumerate(ug)}
        fold = np.array([fmap[g] for g in grp])
    correct = 0
    for f in range(folds):
        tr, te = fold != f, fold == f
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000, C=0.5).fit(sc.transform(X[tr]), y[tr])
        correct += (m.predict(sc.transform(X[te])) == y[te]).sum()
    return correct / len(y)


def main():
    out = sys.argv[1]
    repo = sys.argv[2] if len(sys.argv) > 2 else "marlygotti/LC25000"
    os.makedirs(out, exist_ok=True)
    imgs, labels = load(repo, out)
    print("classes:", {c: int((labels == c).sum()) for c in np.unique(labels)}, flush=True)
    z = embed(imgs)
    zn = z / np.linalg.norm(z, axis=1, keepdims=True)
    rng = np.random.default_rng(0)
    probe = rng.choice(len(z), 2000, replace=False)
    sim = zn[probe] @ zn.T
    sim[np.arange(len(probe)), probe] = -1
    nn = sim.max(1)
    print("nearest-neighbour cosine similarity, quantiles 1/10/50/90%:", np.round(np.quantile(nn, [0.01, 0.1, 0.5, 0.9]), 4), flush=True)
    tumour = np.isin(labels, [l for l in np.unique(labels) if "n" != l.split("_")[-1]])
    for thr in (0.97, 0.95, 0.92, 0.90):
        g = groups(z, thr)
        sizes = np.bincount(np.unique(g, return_inverse=True)[1])
        print(f"\n===== link threshold {thr}: {len(sizes)} groups (expected about 750 if augmentation copies were found); "
              f"median size {np.median(sizes):.0f}, largest {sizes.max()}; mixed-class groups {sum(len(set(labels[g == k])) > 1 for k in np.unique(g))} =====", flush=True)
        for name, mask in (("adenocarcinoma vs squamous", tumour), ("three classes", np.ones(len(labels), bool))):
            r = evaluate(z[mask], labels[mask], None, 5, rng)
            gg = evaluate(z[mask], labels[mask], g[mask], 5, rng)
            print(f"RESULT thr {thr} {name}: random folds accuracy {r:.4f}, near-duplicate-grouped folds {gg:.4f}", flush=True)


if __name__ == "__main__":
    main()
