"""Prop. 3 is a statement about training-support inputs. Evaluate the behavioural tests on TRAINING
triples as well as held-out triples, for round-1 final free models and round-2 checkpoints.
Output: results/r2_trainsupport.json"""
import os, glob, json, re
import numpy as np, torch
import cotlab as C
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
P = 23
rows = []
items = []
for f in sorted(glob.glob(os.path.join(R, "runs", "add_free_*.json"))):
    rec = json.load(open(f)); items.append((rec["tag"], rec["eta"], rec["delta"], rec["seed"], rec["steps"], os.path.join(R, "runs", rec["tag"] + ".pt")))
for f in sorted(glob.glob(os.path.join(R, "runs_r2", "*_step*.pt"))):
    m = re.match(r"r2_e([\d.]+)_d([\d.]+)_s(\d+)_step(\d+)\.pt", os.path.basename(f))
    items.append((os.path.basename(f)[:-3], float(m[1]), float(m[2]), int(m[3]), int(m[4]), f))
for tag, eta, delta, seed, step, pt in items:
    m = C.TinyTF(P, d=64, L=2, regime="free"); m.load_state_dict(torch.load(pt)); m.eval()
    Dtr, Dte = C.make_data(P, "add", seed)
    rtr = C.evaluate(m, Dtr, P, "add"); rte = C.evaluate(m, Dte, P, "add")
    rows.append(dict(tag=tag, eta=eta, delta=delta, seed=seed, step=step,
                     train={k: rtr[k] for k in ("acc_clean", "acc_trunc", "follow", "ignore")},
                     test={k: rte[k] for k in ("acc_clean", "acc_trunc", "follow", "ignore")}))
json.dump(rows, open(os.path.join(R, "r2_trainsupport.json"), "w"), indent=1)
for r in rows:
    print(r["tag"], {k: round(v, 3) for k, v in r["train"].items()}, {k: round(v, 3) for k, v in r["test"].items()})
