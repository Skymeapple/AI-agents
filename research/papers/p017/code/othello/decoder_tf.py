"""Amortised window decoder: one bidirectional transformer trained with board supervision on
random crops of the last k tokens (k ~ log-uniform in 1..60; absolute positions kept), then
evaluated at fixed k. It is k-suffix-measurable at evaluation time, so its accuracy is a lower
bound on Acc*_k (plus positional information). Resumable: --minutes per segment.
Output: decoder_tf.pt, decoder_tf.json"""
import argparse, json, os, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import OUT, load_lab, valid_mask, stratified
ap = argparse.ArgumentParser(); ap.add_argument("--minutes", type=float, default=25); ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--eval", action="store_true"); ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args(); torch.set_num_threads(a.threads)
PAD = 60


class Dec(nn.Module):
    def __init__(self, d=128, nl=4, h=4):
        super().__init__()
        self.emb = nn.Embedding(61, d); self.pos = nn.Parameter(torch.randn(60, d) * 0.02)
        self.q = nn.Parameter(torch.randn(d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, h, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, nl); self.ln = nn.LayerNorm(d); self.out = nn.Linear(d, 64 * 3)

    def forward(self, toks, posi, padmask):
        # toks/posi: (B, K) window tokens and their absolute positions; last column = query position t
        x = self.emb(toks) + self.pos[posi]
        x = self.enc(x, src_key_padding_mask=padmask)
        return self.out(self.ln(x[:, -1])).view(-1, 64, 3)


def crops(toks, gi, ti, ks, K=60):
    B = len(gi)
    off = np.arange(K)[None, :] - (K - 1)               # -59..0
    tt = ti[:, None] + off                              # absolute positions
    keep = (off > -ks[:, None]) & (tt >= 0)
    tk = np.where(keep, toks[gi[:, None], np.maximum(tt, 0)], PAD)
    return torch.as_tensor(tk), torch.as_tensor(np.maximum(tt, 0)), torch.as_tensor(~keep)


ck = os.path.join(OUT, "decoder_tf.pt")
torch.manual_seed(0)
m = Dec()
if not a.eval:
    tr = load_lab(["lab1", "lab2"]); v = valid_mask(tr[3]); GI, TI = np.nonzero(v)
    opt = torch.optim.AdamW(m.parameters(), 1e-3, weight_decay=0.01)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=a.steps, pct_start=0.05)
    step = 0
    if os.path.exists(ck):
        s = torch.load(ck); m.load_state_dict(s["m"]); opt.load_state_dict(s["o"]); sch.load_state_dict(s["s"]); step = s["step"]
    rng = np.random.default_rng(step); t0 = time.time()
    while step < a.steps and time.time() - t0 < a.minutes * 60:
        j = rng.integers(0, len(GI), 256)
        ks = np.round(np.exp(rng.uniform(0, np.log(60), 256))).astype(int)
        tk, ps, pm = crops(tr[0], GI[j], TI[j], ks)
        y = torch.as_tensor(tr[1][GI[j], TI[j]].astype(np.int64))
        loss = F.cross_entropy(m(tk, ps, pm).reshape(-1, 3), y.reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step(); step += 1
        if step % 200 == 0:
            print(f"step {step} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
            torch.save({"m": m.state_dict(), "o": opt.state_dict(), "s": sch.state_dict(), "step": step}, ck)
    torch.save({"m": m.state_dict(), "o": opt.state_dict(), "s": sch.state_dict(), "step": step}, ck)
else:
    m.load_state_dict(torch.load(ck)["m"]); m.eval()
    te = load_lab(["lab4"]); v = valid_mask(te[3][:2000]); GI, TI = np.nonzero(v)
    res = {}
    for k in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 60]:
        preds = []
        with torch.no_grad():
            for i in range(0, len(GI), 4096):
                tk, ps, pm = crops(te[0], GI[i:i + 4096], TI[i:i + 4096], np.full(len(GI[i:i + 4096]), k))
                preds.append(m(tk, ps, pm).argmax(-1).numpy())
        corr = np.concatenate(preds) == te[1][GI, TI]
        res[str(k)] = stratified(corr, te[2][GI, TI])
        print(k, {x: round(y[0], 4) for x, y in res[str(k)].items()}, flush=True)
    json.dump(res, open(os.path.join(OUT, "decoder_tf.json"), "w"), indent=1)
