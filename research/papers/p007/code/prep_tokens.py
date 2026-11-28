"""Tokenise four real data sources into disjoint ranges of one shared vocabulary.

  image   : MNIST digits, 2x2-average-pooled to 14x14, 4 grey levels, raster order
  audio   : Free Spoken Digit Dataset (FSDD) recordings, 8 kHz; log band energies
            (25 ms window, 10 ms hop, 24 log-spaced bands), per-utterance mean
            normalisation, vector-quantised with a 64-code k-means codebook
  text    : Tiny Shakespeare, characters
  fashion : Fashion-MNIST, same tokenizer as image

Every item (image / utterance / text line block) ends with a per-modality
separator token.  Outputs ../data_cache/tokens.npz with one train stream and
one held-out stream per modality (held-out = last 10% of items; for audio the
held-out items are the recordings with index 45-49 of every speaker/digit).
Downloads (not redistributed) are expected in ../data_cache (see README).
Run: python3 prep_tokens.py
"""
import gzip, os, glob
import numpy as np
from scipy.io import wavfile
from scipy.cluster.vq import kmeans2

DC = os.path.join(os.path.dirname(__file__), "..", "data_cache")
rng = np.random.default_rng(0)


def idx_images(path):
    with gzip.open(path) as f:
        b = f.read()
    n = int.from_bytes(b[4:8], "big")
    return np.frombuffer(b, np.uint8, offset=16).reshape(n, 28, 28)


def image_tokens(imgs, levels=4):
    x = imgs.reshape(-1, 14, 2, 14, 2).mean((2, 4)) / 256.0
    q = np.minimum((x * levels).astype(np.int64), levels - 1)
    return q.reshape(len(imgs), -1)


def stream(items, sep):
    out = []
    for it in items:
        out.append(np.asarray(it, np.int64)); out.append(np.array([sep]))
    return np.concatenate(out)


def audio_feats(path, nb=24):
    sr, w = wavfile.read(path)
    w = w.astype(np.float64)
    if w.ndim > 1:
        w = w.mean(1)
    w = w / (np.abs(w).max() + 1e-9)
    win, hop = int(0.025 * sr), int(0.010 * sr)
    if len(w) < win:
        w = np.pad(w, (0, win - len(w)))
    nf = 1 + (len(w) - win) // hop
    fr = np.stack([w[i * hop: i * hop + win] for i in range(nf)]) * np.hanning(win)
    P = np.abs(np.fft.rfft(fr, n=256, axis=1)) ** 2
    edges = np.unique(np.geomspace(2, P.shape[1] - 1, nb + 1).astype(int))
    B = np.stack([P[:, a:b].sum(1) for a, b in zip(edges[:-1], edges[1:])], 1)
    L = np.log(B + 1e-6)
    return L - L.mean(0)


if __name__ == "__main__":
    out = {}
    V = 0
    vocab = {}
    # image + fashion
    for name, fn in [("image", "train-images-idx3-ubyte.gz"), ("fashion", "fashion-train-images-idx3-ubyte.gz")]:
        imgs = idx_images(os.path.join(DC, fn))[:20000]
        tok = image_tokens(imgs) + V
        sep = V + 4
        ntr = int(0.9 * len(tok))
        out[name + "_train"] = stream(tok[:ntr], sep); out[name + "_val"] = stream(tok[ntr:], sep)
        vocab[name] = (V, V + 5); V += 5
    # audio
    files = sorted(glob.glob(os.path.join(DC, "fsdd", "*.wav")))
    feats = [(os.path.basename(f), audio_feats(f)) for f in files]
    is_val = [int(b.split("_")[2].split(".")[0]) >= 45 for b, _ in feats]
    trainF = np.concatenate([F for (b, F), v in zip(feats, is_val) if not v])
    sub = trainF[rng.choice(len(trainF), min(40000, len(trainF)), replace=False)]
    cb, _ = kmeans2(sub, 64, seed=1, minit="++", iter=30)
    def vq(F):
        return np.argmin(((F[:, None, :] - cb[None]) ** 2).sum(-1), 1)
    toks_tr = [vq(F) + V for (b, F), v in zip(feats, is_val) if not v]
    toks_va = [vq(F) + V for (b, F), v in zip(feats, is_val) if v]
    order = rng.permutation(len(toks_tr))
    out["audio_train"] = stream([toks_tr[i] for i in order], V + 64)
    out["audio_val"] = stream(toks_va, V + 64)
    vocab["audio"] = (V, V + 65); V += 65
    # text
    txt = open(os.path.join(DC, "shakespeare.txt")).read()
    chars = sorted(set(txt))
    ids = np.array([chars.index(c) for c in txt], np.int64) + V
    ntr = int(0.9 * len(ids))
    out["text_train"] = ids[:ntr]; out["text_val"] = ids[ntr:]
    vocab["text"] = (V, V + len(chars)); V += len(chars)
    # one modality-tag token per modality (prepended to every training window)
    for name in ["image", "audio", "text", "fashion"]:
        vocab[name + "_tag"] = V; V += 1
    out["vocab_size"] = np.array(V)
    for k, (a, b) in [(k, v) for k, v in vocab.items() if not k.endswith("_tag")]:
        out[k + "_range"] = np.array([a, b])
        out[k + "_tag"] = np.array(vocab[k + "_tag"])
    np.savez_compressed(os.path.join(DC, "tokens.npz"), **out)
    for k, v in out.items():
        print(k, v.shape if v.ndim else v)
