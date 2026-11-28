"""ScienceWorld with no training: random actions, a small open model, and the same model snapped to valid actions.

HFM's track fine-tunes a 7B model on ScienceWorld (30 text-based elementary-science
tasks) and scores 35.8 of 100. That needs a GPU. This stand-in measures, on CPU and
with no training, what three simple agents score on one test variation of each task:

  random   a uniformly random valid action each step
  free     a small open model (Qwen2.5-3B, 4-bit) writes a command; it is sent as written
  snap     the same model, but its command is replaced by the most similar valid action

"snap" tests in an interactive setting what grammar-constrained decoding showed for
TOPv2 parsing: how much is lost to commands the simulator cannot parse. The gold
action sequence is replayed first as a check that the environment scores 100.

Usage: python scienceworld_small_model.py <model.gguf or -> <random|free|snap|gold> <steps> <out.jsonl> [part parts]
"""
from __future__ import annotations

import difflib
import json
import os
import random
import sys
import time

SYSTEM = ("You are playing a text-based science simulation. Each turn, reply with exactly one command and nothing else. "
          "Commands look like: look around; go to kitchen; teleport to kitchen; open cupboard; pick up thermometer; "
          "put ice in metal pot; pour water into cup; activate stove; deactivate stove; use thermometer on water; "
          "focus on plant; examine apple; read book; mix cup; wait; connect wire to battery; inventory.")


def main():
    model_path, mode, steps, out_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    part, parts = (int(sys.argv[5]), int(sys.argv[6])) if len(sys.argv) > 6 else (0, 1)
    from scienceworld import ScienceWorldEnv
    env = ScienceWorldEnv("", envStepLimit=steps + 5)
    tasks = sorted(env.get_task_names())
    llm = None
    if mode in ("free", "snap"):
        from llama_cpp import Llama
        llm = Llama(model_path=model_path, n_ctx=4096, n_threads=os.cpu_count(), verbose=False)
    out = open(out_path, "a")
    rnd = random.Random(0)
    total = []
    for ti, task in enumerate(tasks):
        if ti % parts != part:
            continue
        env.load(task, 0, "easy", generateGoldPath=True)
        var = env.get_variations_test()[0]
        runs = 5 if mode == "random" else 1
        scores, parsed, n_act = [], 0, 0
        for run in range(runs):
            env.load(task, var, "easy", generateGoldPath=True)
            obs, info = env.reset()
            desc = info.get("taskDesc") or env.get_task_description()
            history, score, t0 = [], 0, time.time()
            gold = env.get_gold_action_sequence() if mode == "gold" else None
            for step in range(len(gold) if gold else steps):
                valid = info.get("valid") or env.get_valid_action_object_combinations()
                if mode == "gold":
                    act = gold[step]
                elif mode == "random":
                    act = rnd.choice(valid)
                else:
                    recent = "\n".join(f"> {a}\n{o[:300]}" for a, o in history[-4:])
                    user = f"Task: {desc}\n\n{recent}\n\nCurrent observation: {obs[:700]}\n\nYour next command:"
                    r = llm.create_chat_completion(messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], max_tokens=24, temperature=0.0)
                    act = r["choices"][0]["message"]["content"].strip().split("\n")[0].strip("`>. ").lower()
                    if history and sum(a == act for a, _ in history[-3:]) >= 2:   # stuck repeating: take a random valid action
                        act = rnd.choice(valid)
                    elif mode == "snap" and act not in valid:
                        m = difflib.get_close_matches(act, valid, n=1, cutoff=0.0)
                        act = m[0] if m else rnd.choice(valid)
                n_act += 1
                parsed += act in valid
                obs, reward, done, info = env.step(act)
                score = max(score, info.get("score", 0))
                history.append((act, obs))
                if done:
                    break
            scores.append(max(score, 0))
        rec = {"task": task, "variation": var, "mode": mode, "score": sum(scores) / len(scores), "valid_share": parsed / max(n_act, 1), "actions": n_act, "last_actions": [a for a, _ in history[-6:]]}
        total.append(rec["score"])
        out.write(json.dumps(rec) + "\n"); out.flush()
        print(f"{mode:<7}{task:<52} score {rec['score']:>5.1f}  valid commands {100 * rec['valid_share']:>3.0f}%  {time.time() - t0:.0f}s", flush=True)
    print(f"\n{mode}: mean score {sum(total) / max(len(total), 1):.1f} over {len(total)} tasks, up to {steps} steps each")


if __name__ == "__main__":
    main()
