# p024 — Permission boundaries for LLM agents: reproducible code

Requirements: Python 3.13, numpy, matplotlib, `pip install agentdojo==0.1.35` (benchmark v1.2.2 task
definitions; no LLM and no API key is used anywhere). Total runtime < 2 min CPU.

    cd code
    python extract_agentdojo.py          # -> results/agentdojo_structure.json  (ground-truth calls, injection vectors seen)
    python extract_trajectories.py       # -> results/agentdojo_trajectories.json (per-step outputs + canaries, for taint position)
    python analyze_permissions.py        # -> permission_analysis.json, pair_feasibility.csv, trifecta_by_task.csv,
                                         #    target_provenance.csv, fig_granularity_frontier.png
    python permission_reachability.py    # -> reachability_summary.json/pairs.json/table.csv (TOOL/DEST/BIND, taint revocation,
                                         #    sink-only mode, cluster-bootstrap CIs)
    python fig_policies.py               # -> fig_policies.png
    python provenance_strict.py          # -> provenance_strict.json
    python fatigue_and_holdout.py        # -> approval_fatigue.json, fig_approval_fatigue.png, profile_holdout.json
    python bridge_confirmation_load.py   # -> confirmation_load.json
    python attention_budget.py           # -> attention_budget.json, fig_attention_budget.png (Theorem 3 check)

Hand annotations: 9 injection tasks ship with an empty ground truth in AgentDojo v1.2.x (workspace
injection_task_6..13, travel injection_task_6). Their minimal requirements are annotated in
`MANUAL_INJECTION` (extract_agentdojo.py; with alternatives) and `MANUAL` (permission_reachability.py;
single canonical list). Tool labels and consequential-argument lists are in analyze_permissions.py
(LABELS, TARGET) and permission_reachability.py (EGRESS, WRITE, PRIVATE, DEST_ARGS, CONTENT_ARGS).
Seeds: 20261122 (hold-out), 0 (bootstrap), 7 (attention-budget instances).
