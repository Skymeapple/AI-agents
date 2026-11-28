"""Train one model and save evaluation history + final weights.
usage: python3 run_one.py TAG TASK REGIME ETA DELTA L SEED STEPS"""
import sys, json, time, os, torch
import cotlab as C
tag, task, reg = sys.argv[1], sys.argv[2], sys.argv[3]
eta, delta = float(sys.argv[4]), float(sys.argv[5])
L, seed, steps = int(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
P = 23
out = os.path.join(os.path.dirname(__file__), "..", "results", "runs")
os.makedirs(out, exist_ok=True)
t = time.time()
m, Dtr, Dte, hist = C.train(p=P, task=task, regime=reg, eta=eta, delta=delta, L=L, d=64, bs=256,
                            steps=steps, seed=seed, log_every=250, eval_fn=C.evaluate)
final = C.evaluate(m, Dte, P, task)
rec = dict(tag=tag, task=task, regime=reg, eta=eta, delta=delta, L=L, seed=seed, steps=steps,
           p=P, d=64, final=final, history=hist, seconds=time.time() - t)
json.dump(rec, open(os.path.join(out, tag + ".json"), "w"), indent=1)
torch.save(m.state_dict(), os.path.join(out, tag + ".pt"))
print(tag, round(time.time() - t), {k: round(v, 3) for k, v in final.items()}, flush=True)
