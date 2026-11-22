# p024 — Permission boundaries for LLM agents: reproducible code

Requirements: Python 3.13, numpy, matplotlib, `pip install agentdojo==0.1.35` (benchmark v1.2.2 task definitions; no LLM or API key is used).

Run (total < 2 min CPU):

    cd code
    python extract_agentdojo.py      # -> results/agentdojo_structure.json (ground-truth tool calls, injection vectors seen)
    python analyze_permissions.py    # -> results/permission_analysis.json, pair_feasibility.csv, trifecta_by_task.csv,
                                     #    target_provenance.csv, fig_granularity_frontier.png
    python fig_policies.py           # -> results/fig_policies.png
    python fatigue_and_holdout.py    # -> results/approval_fatigue.json, fig_approval_fatigue.png, profile_holdout.json

Hand annotations: 9 injection tasks ship with an empty ground truth in AgentDojo (workspace injection_task_6..13,
travel injection_task_6); their minimal tool requirements are annotated in `MANUAL_INJECTION` in extract_agentdojo.py.
Tool labels (private / sink / state-changing) and consequential-argument lists are in analyze_permissions.py (LABELS, TARGET).
Random seed for the hold-out experiment: 20261122.
    python provenance_strict.py      # -> results/provenance_strict.json
