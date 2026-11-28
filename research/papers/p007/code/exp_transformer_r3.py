"""Round-3 additions to the transformer study (Exp. H); same model/data/protocol as exp_transformer.py.

  seeds    : seeds 2,3,4 for the whole round-2 design (M=2 size x budget grid and the
             P4 modality-addition sweep), so every cell has 5 seeds.  Identical training
             code path (verified: `check` reproduces a round-2 run bit-for-bit); in addition
             the per-window test losses at each modality's best-validation checkpoint are
             stored, so that test-set sampling uncertainty can be bootstrapped.
             -> ../results/transformer_runs_r3.jsonl
  converge : optimisation control at d_model=32, D=25k: step cap raised from 3000 to 9000 and
             three learning rates (1e-3, 2e-3, 4e-3), seeds 0-2, unified(image+audio) and both
             specialists.  -> ../results/transformer_converge.jsonl

Usage: python3 exp_transformer_r3.py {seeds|converge|check} [part nparts]
"""
import json, os, sys, time, math, itertools
import numpy as np
import torch
import torch.nn.functional as F
import exp_transformer as E

R = os.path.join(E.HERE, "..", "results")
V, L, BM = E.V, E.L, E.BM


@torch.no_grad()
def evaluate(model, xb):
    model.eval()
    logits = model(xb[:, :-1])
    loss = F.cross_entropy(logits.reshape(-1, V), xb[:, 1:].reshape(-1))  # as in exp_transformer
    per = F.cross_entropy(logits.reshape(-1, V), xb[:, 1:].reshape(-1), reduction="none")
    per = per.reshape(xb.shape[0], -1).mean(1)  # mean nats/token of each held-out window
    model.train()
    return loss.item(), per.numpy().round(5).tolist()


def train(mods, D, dm, seed, lr0=E.LR, maxsteps=E.MAXSTEPS):
    """Copy of exp_transformer.train with lr / step cap as arguments and per-window test losses."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    nl, nh = E.SIZES[dm]
    model = E.GPT(dm, nl, nh)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr0, weight_decay=0.0, betas=(0.9, 0.98))
    T = E.T
    streams = {m: T[m + "_train"][:D] for m in mods}
    ho = {m: E.heldout(m) for m in mods}
    best, best_step, best_test, best_val = math.inf, 0, None, None
    pm = {m: dict(val=math.inf, test=None, step=0, win=None) for m in mods}
    last_improve = 0
    t0 = time.time()
    for step in range(1, maxsteps + 1):
        lr = lr0 * min(1.0, step / 200)
        for g in opt.param_groups:
            g["lr"] = lr
        xb = torch.cat([E.windows(streams[m], int(T[m + "_tag"]),
                                  rng.integers(0, D - L - 1, BM)) for m in mods])
        logits = model(xb[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, V), xb[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % E.EVAL_EVERY == 0:
            vl = {m: evaluate(model, ho[m]["val"])[0] for m in mods}
            tw = {m: evaluate(model, ho[m]["test"]) for m in mods}
            tl = {m: tw[m][0] for m in mods}
            v = float(np.mean(list(vl.values())))
            if v < best - 1e-4:
                best, best_step, best_test, best_val = v, step, tl, vl
                last_improve = step
            for m in mods:
                if vl[m] < pm[m]["val"] - 1e-4:
                    pm[m] = dict(val=vl[m], test=tl[m], step=step, win=tw[m][1]); last_improve = step
            if step - last_improve >= E.PATIENCE * E.EVAL_EVERY:
                break
    return dict(mods=list(mods), D=D, d_model=dm, seed=seed, lr=lr0, maxsteps=maxsteps, n_params=nparam,
                best_step=best_step, steps_run=step, val=best_val, test=best_test,
                val_per_mod_stop={m: pm[m]["val"] for m in mods},
                test_per_mod_stop={m: pm[m]["test"] for m in mods},
                step_per_mod_stop={m: pm[m]["step"] for m in mods},
                test_windows_per_mod_stop={m: pm[m]["win"] for m in mods},
                secs=round(time.time() - t0, 1), torch=torch.__version__)


def jobs(which):
    J = []
    order = ["image", "audio", "text", "fashion"]
    if which == "seeds":
        for seed in [2, 3, 4]:
            for D, dm in itertools.product([25000, 100000], [16, 32, 64, 128]):
                J += [(("image", "audio"), D, dm, seed, E.LR, E.MAXSTEPS), (("image",), D, dm, seed, E.LR, E.MAXSTEPS),
                      (("audio",), D, dm, seed, E.LR, E.MAXSTEPS)]
            J += [(tuple(order[:3]), 25000, 32, seed, E.LR, E.MAXSTEPS), (tuple(order[:4]), 25000, 32, seed, E.LR, E.MAXSTEPS),
                  (("text",), 25000, 32, seed, E.LR, E.MAXSTEPS), (("fashion",), 25000, 32, seed, E.LR, E.MAXSTEPS)]
        # longest first, so that round-robin parts are balanced
        J.sort(key=lambda j: -(j[2] ** 1.3) * (len(j[0]) + ("image" in j[0]) + ("fashion" in j[0])))
    if which == "converge":
        for seed in [0, 1, 2]:
            for lr in [1e-3, 2e-3, 4e-3]:
                for mods in [("image", "audio"), ("image",), ("audio",)]:
                    J.append((mods, 25000, 32, seed, lr, 9000))
    return J


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "check":  # must equal the round-2 line for (audio, 25k, d=32, seed 0)
        r = train(("audio",), 25000, 32, 0)
        old = [json.loads(l) for l in open(E.OUT)]
        o = next(o for o in old if o["mods"] == ["audio"] and o["D"] == 25000 and o["d_model"] == 32 and o["seed"] == 0)
        ok = all(r[k] == o[k] for k in ["best_step", "steps_run", "val", "test", "test_per_mod_stop", "step_per_mod_stop"])
        print("bit-identical to round-2 run:", ok, "| mean of window losses", np.mean(r["test_windows_per_mod_stop"]["audio"]),
              "vs", r["test_per_mod_stop"]["audio"], "| torch", torch.__version__)
        sys.exit(0 if ok else 1)
    part, nparts = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 1)
    out = os.path.join(R, "transformer_runs_r3.jsonl" if which == "seeds" else "transformer_converge.jsonl")
    for mods, D, dm, seed, lr, ms in jobs(which)[part::nparts]:
        done = set()
        if os.path.exists(out):
            for line in open(out):
                o = json.loads(line); done.add((tuple(o["mods"]), o["D"], o["d_model"], o["seed"], o["lr"]))
        if (tuple(mods), D, dm, seed, lr) in done:
            continue
        r = train(mods, D, dm, seed, lr, ms)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")
        print(json.dumps({k: v for k, v in r.items() if k != "test_windows_per_mod_stop"}), flush=True)
