"""Extract structural (LLM-free) data from the AgentDojo benchmark.

For every user task: ground-truth tool calls (name + args) and, for each call,
whether its output contains an injection canary (i.e. untrusted, attacker-
controllable content enters the context at that step).
For every injection task: ground-truth tool calls (name + args).
Output: results/agentdojo_trajectories.json (per-step outputs and canaries; used by permission_reachability.py)
"""
import json, sys, os, copy
from agentdojo.task_suite.load_suites import get_suites
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.types import get_text_content_as_str

VERSION = sys.argv[1] if len(sys.argv) > 1 else "v1.2.2"
OUT = os.path.join(os.path.dirname(__file__), "..", "results", "agentdojo_trajectories.json")

def jsonable(x):
    try:
        json.dumps(x); return x
    except TypeError:
        return str(x)

data = {"version": VERSION, "suites": {}}
for sname, suite in get_suites(VERSION).items():
    defaults = suite.get_injection_vector_defaults()
    canaries = {k: f"<CANARY---{k}--->" for k in defaults}
    env0 = suite.load_and_inject_default_environment(canaries)
    users = []
    for ut in suite.user_tasks.values():
        env = env0.copy(deep=True)
        runtime = FunctionsRuntime(suite.tools)
        calls = ut.ground_truth(env.copy(deep=True))
        steps = []
        for c in calls:
            res, err = runtime.run_function(env, c.function, dict(c.args), raise_on_error=False)
            out = str(res) if res is not None else ""
            seen = [k for k, v in canaries.items() if v in out]
            steps.append({"tool": c.function, "args": {k: jsonable(v) for k, v in dict(c.args).items()},
                          "canaries": seen, "error": err, "output": out})
        users.append({"id": ut.ID, "prompt": ut.PROMPT, "steps": steps})
    injs = []
    envc = suite.load_and_inject_default_environment({})
    for it in suite.injection_tasks.values():
        calls = it.ground_truth(envc.copy(deep=True))
        injs.append({"id": it.ID, "goal": it.GOAL,
                     "steps": [{"tool": c.function, "args": {k: jsonable(v) for k, v in dict(c.args).items()}} for c in calls]})
    data["suites"][sname] = {"tools": [t.name for t in suite.tools], "user_tasks": users,
                             "injection_tasks": injs, "n_vectors": len(defaults)}
    print(sname, len(users), len(injs), sum(any(s["canaries"] for s in u["steps"]) for u in users), "injectable")
json.dump(data, open(OUT, "w"), indent=1)
print("wrote", OUT)
