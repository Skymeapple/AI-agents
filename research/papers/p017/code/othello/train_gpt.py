"""Train the small Othello GPT on next-move prediction (synthetic games), resumable.
Usage: python train_gpt.py --minutes 28   (repeat to continue from the checkpoint)
Checkpoint: ../../results/othello/gpt.pt ; log: ../../results/othello/train_log.txt"""
import argparse, os, time, numpy as np, torch, torch.nn.functional as F
from gpt import GPT, PAD
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "othello")
ap = argparse.ArgumentParser(); ap.add_argument("--minutes", type=float, default=28)
ap.add_argument("--total_steps", type=int, default=12000); ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)
toks = np.concatenate([np.load(os.path.join(OUT, f"data_lm{i}.npz"))["toks"] for i in range(1, 5)]).astype(np.int64)
toks[toks < 0] = PAD
X = torch.as_tensor(toks)
torch.manual_seed(0)
m = GPT()
opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.total_steps, pct_start=0.05)
ck = os.path.join(OUT, "gpt.pt")
step = 0
if os.path.exists(ck):
    s = torch.load(ck); m.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"]); step = s["step"]
g = torch.Generator().manual_seed(1234 + step)
log = open(os.path.join(OUT, "train_log.txt"), "a")
t0 = time.time(); run = 0.0
m.train()
while step < a.total_steps and time.time() - t0 < a.minutes * 60:
    idx = torch.randint(0, X.shape[0], (a.bs,), generator=g)
    b = X[idx]
    lg = m(b[:, :-1])
    loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), b[:, 1:].reshape(-1), ignore_index=PAD)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sched.step()
    step += 1; run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
    if step % 200 == 0:
        msg = f"step {step} loss {run:.4f} elapsed {time.time()-t0:.0f}s"; print(msg, flush=True); log.write(msg + "\n"); log.flush()
        torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, ck)
torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, ck)
print("stopped at step", step)
