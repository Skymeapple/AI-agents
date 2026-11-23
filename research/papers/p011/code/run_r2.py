"""Round-2 pre-specified seed sweep (see sweep_r2_spec.json).
Trains one free model, logs behavioural tests every 250 steps, and saves weights at each step in SAVE.
usage: python3 run_r2.py ETA DELTA SEED STEPS SAVE1,SAVE2,..."""
import sys, json, time, os
import numpy as np, torch, torch.nn.functional as F
import cotlab as C
eta, delta, seed, steps = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
save = set(int(s) for s in sys.argv[5].split(","))
P = 23
tag = os.environ.get("R2TAG") or f"r2_e{eta:g}_d{delta:g}_s{seed}"
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "runs_r2")
os.makedirs(out, exist_ok=True)
t0 = time.time()
torch.manual_seed(seed)
rng = np.random.default_rng(seed + 1000)
Dtr, Dte = C.make_data(P, "add", seed)
model = C.TinyTF(P, d=64, L=2, regime="free")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.5, betas=(0.9, 0.98))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: min(1.0, (t + 1) / 200))
hist = []
for t in range(steps):
    bi = rng.integers(0, len(Dtr), 256)
    X, s, y = C.build_batch(Dtr[bi], P, eta, delta, rng)
    o = model(X)
    loss = F.cross_entropy(o[:, 3], s) + F.cross_entropy(o[:, 5], y)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if (t + 1) % 250 == 0:
        r = C.evaluate(model, Dte, P, "add"); r["step"] = t + 1; hist.append(r)
    if (t + 1) in save:
        torch.save(model.state_dict(), os.path.join(out, f"{tag}_step{t+1}.pt"))
rec = dict(tag=tag, eta=eta, delta=delta, seed=seed, steps=steps, saved=sorted(save), p=P, d=64, L=2,
           history=hist, seconds=time.time() - t0)
json.dump(rec, open(os.path.join(out, tag + ".json"), "w"), indent=1)
print(tag, round(time.time() - t0), hist[-1], flush=True)
