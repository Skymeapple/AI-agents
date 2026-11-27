"""Does reinforcement learning on procedurally generated, test-checked tasks transfer?
Toy scale, on CPU.

Scaled-down experiment for HFM's "Agentic RL Training Recipe Exploration with verl and
agent-harness" track, which needs GPUs. That track's best idea is to replace a general
agentic data mix with procedurally synthesised terminal tasks, each carrying an
executable test, and train with GRPO. The published version of the idea (Endless
Terminals) lifted its own held-out set from 10.7% to 53.3% but Terminal-Bench 2.0 only
from 2.2% to 3.4%. The question that matters is transfer, and it can be asked of a
small model on a small task family.

Task: a "shell" of eight string commands. The model sees two input/output examples and
must write a command pipeline (1 to 4 commands). The reward is 1 if the pipeline
reproduces both examples and a third, hidden test case, else 0. Nothing checks the
program text, only its behaviour.

Model: a 3-layer transformer (width 128), first trained by supervised learning on
pipelines of one or two commands. Then one of these recipes runs for the same number of
updates (16 prompts x 8 samples each):

  grpo-narrow     GRPO on generated tasks of 2-3 commands from 4 of the 8 commands
  grpo-diverse    GRPO on generated tasks of 1-3 commands from all 8 commands
  grpo-dynamic    as grpo-diverse, keeping only prompts whose group of samples is
                  neither all right nor all wrong (DAPO-style dynamic sampling;
                  draws three times as many prompts, so it costs more generation)
  rft-diverse     rejection-sampling fine-tuning: supervised learning on the model's
                  own samples that pass the test, same tasks as grpo-diverse
  none            no second stage

Scores (greedy decoding, fixed held-out tasks):
  narrow / diverse   fresh tasks from each training generator (in-distribution)
  benchmark          3-4 command pipelines over all 8 commands: longer than anything
                     trained on, the stand-in for an external benchmark

Usage: python grpo_tiny.py <recipe> <rl steps> <seed> [<seed> ...]
"""
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA = "abcdef"
CMDS = {
    "R": lambda s: s[::-1],                                   # reverse
    "S": lambda s: "".join(sorted(s)),                        # sort
    "D": lambda s: "".join(dict.fromkeys(s)),                 # drop repeated letters
    "H": lambda s: s[:max(1, len(s) // 2)],                   # first half
    "T": lambda s: s[1:] if len(s) > 1 else s,                # drop first letter
    "C": lambda s: s[1:] + s[:1],                             # rotate left
    "A": lambda s: s.replace("a", "") or s,                   # delete every a
    "P": lambda s: (s + s)[:10],                              # repeat, capped at 10
}
NARROW = "RSDH"
VOCAB = list(ALPHA) + list(CMDS) + [">", ";", "|", "$", "_"]   # "$" ends a program, "_" pads
TOK = {c: i for i, c in enumerate(VOCAB)}
D, LAYERS, HEADS, CTX, MAXPROG = 128, 3, 4, 48, 5


def run(prog, s):
    for c in prog:
        s = CMDS[c](s)
    return s


def make_task(rng, cmds, lengths):
    """A pipeline whose three cases are non-trivial (output differs from input somewhere)."""
    while True:
        prog = "".join(rng.choice(cmds) for _ in range(rng.choice(lengths)))
        cases = ["".join(rng.choice(ALPHA) for _ in range(rng.randint(4, 8))) for _ in range(3)]
        outs = [run(prog, c) for c in cases]
        if any(o != c for o, c in zip(outs, cases)):
            return prog, list(zip(cases, outs))


GENERATORS = {"narrow": (NARROW, (2, 3)), "diverse": ("".join(CMDS), (1, 2, 3)),
              "benchmark": ("".join(CMDS), (3, 4)), "pretrain": ("".join(CMDS), (1, 2))}


def prompt(cases):
    return f"{cases[0][0]}>{cases[0][1]};{cases[1][0]}>{cases[1][1]}|"


def passes(prog, cases):
    return all(run(prog, a) == b for a, b in cases)


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb, self.pos = nn.Embedding(len(VOCAB), D), nn.Embedding(CTX, D)
        layer = nn.TransformerEncoderLayer(D, HEADS, 4 * D, dropout=0.0, batch_first=True, norm_first=True)
        self.body = nn.TransformerEncoder(layer, LAYERS)
        self.norm, self.head = nn.LayerNorm(D), nn.Linear(D, len(VOCAB))
        self.register_buffer("mask", torch.triu(torch.full((CTX, CTX), float("-inf")), 1))

    def forward(self, x):
        t = x.shape[1]
        h = self.body(self.emb(x) + self.pos(torch.arange(t)), mask=self.mask[:t, :t], is_causal=True)
        return self.head(self.norm(h))


def encode(texts):
    """Right-aligned prompts, so every prompt ends at the same position."""
    n = max(len(t) for t in texts)
    return torch.tensor([[TOK["_"]] * (n - len(t)) + [TOK[c] for c in t] for t in texts])


def sample(model, prompts, greedy=False, temp=1.0):
    """Programs (strings) and the summed log-probability of each, with gradients."""
    x = encode(prompts)
    logps = torch.zeros(len(prompts))
    done = torch.zeros(len(prompts), dtype=torch.bool)
    progs = [""] * len(prompts)
    allowed = torch.full((len(VOCAB),), float("-inf"))
    for c in list(CMDS) + ["$"]:
        allowed[TOK[c]] = 0.0
    for step in range(MAXPROG):
        logits = model(x)[:, -1] / temp + allowed
        if step == 0:
            logits[:, TOK["$"]] = float("-inf")                 # at least one command
        lp = logits.log_softmax(-1)
        nxt = lp.argmax(-1) if greedy else torch.multinomial(lp.exp(), 1)[:, 0]
        if step == MAXPROG - 1:
            nxt = torch.where(done, nxt, torch.full_like(nxt, TOK["$"]))
        logps = logps + torch.where(done, torch.zeros(()), lp.gather(1, nxt[:, None])[:, 0])
        for i, t in enumerate(nxt.tolist()):
            if not done[i] and VOCAB[t] != "$":
                progs[i] += VOCAB[t]
        done = done | (nxt == TOK["$"])
        x = torch.cat([x, nxt[:, None]], 1)
        if done.all():
            break
    return progs, logps


def sft_step(model, opt, pairs):
    """Teacher-forced loss on the program tokens of (prompt, program) pairs."""
    texts = [p + g + "$" for p, g in pairs]
    x = encode(texts)
    lens = torch.tensor([len(g) + 1 for _, g in pairs])
    logits = model(x[:, :-1])
    tgt = x[:, 1:]
    pos = torch.arange(tgt.shape[1])[None]
    keep = pos >= tgt.shape[1] - lens[:, None]
    loss = F.cross_entropy(logits.reshape(-1, len(VOCAB)), tgt.reshape(-1), reduction="none").view_as(tgt)
    loss = (loss * keep).sum() / keep.sum()
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, tasks):
    model.eval()
    out = {}
    for name, ts in tasks.items():
        progs, _ = sample(model, [prompt(c) for _, c in ts], greedy=True)
        out[name] = sum(passes(p, c) for p, (_, c) in zip(progs, ts)) / len(ts)
    model.train()
    return out


def main():
    recipe, rl_steps = sys.argv[1], int(sys.argv[2])
    torch.set_num_threads(4)
    trng = random.Random(99)
    tasks = {k: [make_task(trng, *GENERATORS[k]) for _ in range(400)] for k in ("narrow", "diverse", "benchmark")}
    group, prompts_per_step = 8, 16
    for seed in map(int, sys.argv[3:]):
        torch.manual_seed(seed)
        rng = random.Random(seed)
        model = GPT()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        t0 = time.time()
        for step in range(3000):                                   # supervised stage: 1-2 commands
            pairs = []
            for _ in range(64):
                prog, cases = make_task(rng, *GENERATORS["pretrain"])
                pairs.append((prompt(cases), prog))
            sft_step(model, opt, pairs)
        base = evaluate(model, tasks)
        print(f"  seed {seed} after supervised stage: " + ", ".join(f"{k} {v:.3f}" for k, v in base.items()) + f" ({time.time() - t0:.0f}s)", flush=True)
        if recipe != "none":
            gen = GENERATORS["narrow" if recipe == "grpo-narrow" else "diverse"]
            opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)
            informative = 0
            for step in range(1, rl_steps + 1):
                batch = [make_task(rng, *gen) for _ in range(prompts_per_step * (3 if recipe == "grpo-dynamic" else 1))]
                ps = [prompt(c) for _, c in batch for _ in range(group)]
                progs, logps = sample(model, ps)
                r = torch.tensor([float(passes(p, batch[i // group][1])) for i, p in enumerate(progs)]).view(-1, group)
                if recipe == "grpo-dynamic":
                    keep = ((r.std(1) > 0).nonzero()[:, 0])[:prompts_per_step]
                    if len(keep) == 0:
                        continue
                    r, logps = r[keep], logps.view(-1, group)[keep].reshape(-1)
                informative += int((r.std(1) > 0).sum())
                if recipe.startswith("grpo"):
                    adv = ((r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)).reshape(-1)
                    loss = -(adv * logps).mean()
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                else:                                                     # rft: learn from own passing samples
                    good = [(ps[i], progs[i]) for i in range(len(ps)) if r.view(-1)[i] > 0]
                    good = list(dict.fromkeys(good))
                    if good:
                        sft_step(model, opt, good)
                if step in (rl_steps // 4, rl_steps // 2, 3 * rl_steps // 4):
                    m = evaluate(model, tasks)
                    print(f"  seed {seed} {recipe} step {step}: " + ", ".join(f"{k} {v:.3f}" for k, v in m.items())
                          + f", informative groups {informative / (step * prompts_per_step):.2f}, {time.time() - t0:.0f}s", flush=True)
        final = evaluate(model, tasks)
        print(f"RESULT {recipe} seed {seed}: base " + ", ".join(f"{k} {v:.3f}" for k, v in base.items())
              + " -> final " + ", ".join(f"{k} {v:.3f}" for k, v in final.items()) + f", time {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
