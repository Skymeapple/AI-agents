"""Oracle permission-boundary analysis of AgentDojo (no LLM involved).

Computes, for every (user task u, injection task i) security case:
  * whether the attack is still *feasible* under several permission policies,
    i.e. whether some minimal call list that achieves the injection goal is
    admitted by the policy;
  * trifecta (private data / untrusted content / external sink) structure of
    each user task;
  * provenance of the consequential arguments of each legitimate task
    (user-specified vs. derived from tool outputs);
  * the granularity frontier: feasible attacks vs. number k of static
    permission profiles (greedy agglomerative design);
  * the approval-fatigue model of Section 5 instantiated on these task sets.

Feasibility is a *ceiling* statement: an attack that is feasible may still fail
because the model does not follow the injection. Infeasible attacks cannot
succeed under the stated policy, whatever the model does (given the policy is
enforced outside the model).

Outputs JSON/CSV/PNG files under results/.
"""
import json, os, itertools, csv, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
D = json.load(open(os.path.join(RES, "agentdojo_structure.json")))

# ---------------------------------------------------------------- tool labels
# P: returns the principal's private data; E: can move information/value to a
# party other than the principal (an exfiltration/side-effect sink);
# W: state-changing (integrity-relevant) action inside the principal's domain.
LABELS = {
 "workspace": {
  "P": {"get_unread_emails", "get_sent_emails", "get_received_emails", "get_draft_emails", "search_emails",
        "search_contacts_by_name", "search_contacts_by_email", "search_calendar_events", "get_day_calendar_events",
        "search_files_by_filename", "get_file_by_id", "list_files", "search_files"},
  "E": {"send_email", "share_file", "create_calendar_event", "add_calendar_event_participants",
        "reschedule_calendar_event", "cancel_calendar_event"},
  "W": {"delete_email", "append_to_file", "create_file", "delete_file", "create_calendar_event",
        "cancel_calendar_event", "reschedule_calendar_event", "add_calendar_event_participants"}},
 "travel": {
  "P": {"get_user_information", "search_calendar_events", "get_day_calendar_events"},
  "E": {"send_email", "reserve_hotel", "reserve_car_rental", "reserve_restaurant", "create_calendar_event"},
  "W": {"reserve_hotel", "reserve_car_rental", "reserve_restaurant", "create_calendar_event", "cancel_calendar_event"}},
 "banking": {
  "P": {"get_iban", "get_balance", "get_most_recent_transactions", "get_scheduled_transactions", "read_file", "get_user_info"},
  "E": {"send_money", "schedule_transaction", "update_scheduled_transaction"},
  "W": {"send_money", "schedule_transaction", "update_scheduled_transaction", "update_password", "update_user_info"}},
 "slack": {
  "P": {"get_channels", "read_channel_messages", "read_inbox", "get_users_in_channel"},
  "E": {"send_direct_message", "send_channel_message", "post_webpage", "invite_user_to_slack",
        "add_user_to_channel", "get_webpage"},  # get_webpage: URL query strings can carry data out
  "W": {"add_user_to_channel", "invite_user_to_slack", "remove_user_from_slack", "post_webpage",
        "send_channel_message", "send_direct_message"}},
}
# Arguments that name *whom/what* a consequential action targets.
TARGET = {
 "send_email": ["recipients", "cc", "bcc"], "share_file": ["email", "file_id"],
 "create_calendar_event": ["participants"], "add_calendar_event_participants": ["participants", "event_id"],
 "delete_file": ["file_id"], "delete_email": ["email_id"], "append_to_file": ["file_id"],
 "cancel_calendar_event": ["event_id"], "reschedule_calendar_event": ["event_id"],
 "send_money": ["recipient", "amount"], "schedule_transaction": ["recipient", "amount"],
 "update_scheduled_transaction": ["recipient", "id", "amount"], "update_password": ["password"],
 "update_user_info": ["first_name", "last_name", "street", "city"],
 "reserve_hotel": ["hotel"], "reserve_restaurant": ["restaurant"], "reserve_car_rental": ["company"],
 "send_direct_message": ["recipient"], "send_channel_message": ["channel"], "post_webpage": ["url"],
 "get_webpage": ["url"], "invite_user_to_slack": ["user", "user_email"],
 "add_user_to_channel": ["user", "channel"], "remove_user_from_slack": ["user"],
}


def req_tools(calls):
    return frozenset(c["tool"] for c in calls)


def allowed_targets(calls):
    """argument-level grant: tool -> arg -> set of permitted values."""
    a = {}
    for c in calls:
        t = c["tool"]
        a.setdefault(t, {})
        for arg in TARGET.get(t, []):
            if arg in c["args"]:
                a[t].setdefault(arg, set()).update(c["args"][arg])
    return a


def allowed_targets_realizable(calls, prompt):
    """argument-level grant that a policy author can write *before* the agent
    reads any tool output: an argument is pinned only if all its legitimate
    values appear in the trusted user prompt; otherwise it is left free."""
    a = allowed_targets(calls)
    for t, args in a.items():
        for arg, vals in list(args.items()):
            if not all(in_prompt(v, prompt) for v in vals):
                args[arg] = None  # wildcard
    return a


def call_admitted_arg(call, grant):
    t = call["tool"]
    if t not in grant:
        return False
    for arg in TARGET.get(t, []):
        if arg in call["args"]:
            vals = set(call["args"][arg])
            allowed = grant[t].get(arg, set())
            if allowed is None:  # wildcard (unpinnable argument)
                continue
            if not vals <= allowed:
                return False
    return True


def feasible_tool(alts, G):
    return any(frozenset(c["tool"] for c in alt) <= G for alt in alts)


def feasible_arg(alts, grant):
    return any(all(call_admitted_arg(c, grant) for c in alt) for alt in alts)


def in_prompt(val, prompt):
    v = str(val).strip().lower()
    if not v:
        return True
    p = prompt.lower()
    if v in p:
        return True
    try:  # numeric values: accept e.g. 1100 vs 1100.0
        f = float(v)
        for cand in {str(int(f)) if f.is_integer() else v, f"{f:.2f}", f"{f:,.2f}", f"{f:,.0f}"}:
            if cand in p:
                return True
    except ValueError:
        pass
    return False


def main():
    out = {"suites": {}, "totals": {}}
    rows = []
    tot = dict(cases=0)
    policy_names = ["full_suite", "suite_union", "readonly_split", "task_tool_LP", "task_arg_LP_realizable", "task_arg_LP"]
    for p in policy_names:
        tot[p] = 0
    tri_rows = []
    prov_rows = []
    for sname, S in D["suites"].items():
        L = LABELS[sname]
        tools = frozenset(S["tools"])
        U = S["user_tasks"]; I = S["injection_tasks"]
        union = frozenset().union(*[req_tools(u["calls"]) for u in U.values()])
        readonly = frozenset(t for t in tools if t not in L["E"] | L["W"])
        cnt = {p: 0 for p in policy_names}; n = 0
        for uid, u in U.items():
            R = req_tools(u["calls"])
            grant_arg = allowed_targets(u["calls"])
            grant_real = allowed_targets_realizable(u["calls"], u["prompt"])
            # read-only split: a task that needs no E/W tool runs in read-only profile
            G_split = readonly if not (R & (L["E"] | L["W"])) else tools
            for iid, it in I.items():
                alts = it["alternatives"]
                f = {"full_suite": feasible_tool(alts, tools), "suite_union": feasible_tool(alts, union),
                     "readonly_split": feasible_tool(alts, G_split), "task_tool_LP": feasible_tool(alts, R),
                     "task_arg_LP_realizable": feasible_arg(alts, grant_real),
                     "task_arg_LP": feasible_arg(alts, grant_arg)}
                n += 1
                for p in policy_names:
                    cnt[p] += f[p]
                rows.append([sname, uid, iid, it["manual"]] + [int(f[p]) for p in policy_names])
            # trifecta
            has_P = bool(R & L["P"]); has_U = bool(u["injection_vectors"]); has_E = bool(R & L["E"])
            tri_rows.append([sname, uid, int(has_P), int(has_U), int(has_E), int(has_P and has_U and has_E),
                             int(bool(R & L["W"]))])
            # provenance of consequential target arguments
            derived = []
            consequential = False
            for c in u["calls"]:
                if c["tool"] in (L["E"] | L["W"]):
                    consequential = True
                    for arg in TARGET.get(c["tool"], []):
                        for v in c["args"].get(arg, []):
                            if not in_prompt(v, u["prompt"]):
                                derived.append(f"{c['tool']}.{arg}={v[:40]}")
            prov_rows.append([sname, uid, int(consequential), int(len(derived) > 0), "; ".join(sorted(set(derived)))])
        out["suites"][sname] = {"n_tools": len(tools), "n_union": len(union), "n_readonly": len(readonly),
                                "cases": n, **{p: cnt[p] for p in policy_names}}
        tot["cases"] += n
        for p in policy_names:
            tot[p] += cnt[p]
    out["totals"] = tot

    with open(os.path.join(RES, "pair_feasibility.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["suite", "user_task", "injection_task", "manual_annotation"] + policy_names); w.writerows(rows)
    with open(os.path.join(RES, "trifecta_by_task.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["suite", "user_task", "private_read", "untrusted_input", "external_sink", "trifecta", "state_change"]); w.writerows(tri_rows)
    with open(os.path.join(RES, "target_provenance.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["suite", "user_task", "consequential", "data_derived_target", "derived_args"]); w.writerows(prov_rows)

    # ---- per-injection-task breakdown under task_tool_LP
    inj_break = {}
    for r in rows:
        k = (r[0], r[2]); inj_break.setdefault(k, [0, 0, 0, 0]); inj_break[k][0] += 1
        inj_break[k][1] += r[4 + policy_names.index("task_tool_LP")]
        inj_break[k][2] += r[4 + policy_names.index("task_arg_LP_realizable")]
        inj_break[k][3] += r[4 + policy_names.index("task_arg_LP")]
    out["by_injection_task"] = {f"{a}/{b}": {"pairs": v[0], "feasible_tool_LP": v[1], "feasible_arg_LP_realizable": v[2],
                                "feasible_arg_LP_oracle": v[3]} for (a, b), v in sorted(inj_break.items())}

    # ---- trifecta & provenance summaries
    tri = np.array([r[2:] for r in tri_rows])
    out["trifecta"] = {"tasks": len(tri_rows), "private": int(tri[:, 0].sum()), "untrusted": int(tri[:, 1].sum()),
                       "sink": int(tri[:, 2].sum()), "trifecta": int(tri[:, 3].sum()), "state_change": int(tri[:, 4].sum()),
                       "by_suite": {s: int(sum(r[5] for r in tri_rows if r[0] == s)) for s in D["suites"]}}
    pv = np.array([r[2:4] for r in prov_rows])
    out["provenance"] = {"consequential_tasks": int(pv[:, 0].sum()), "data_derived": int(pv[:, 1].sum()),
                         "by_suite": {s: [int(sum(r[2] for r in prov_rows if r[0] == s)), int(sum(r[3] for r in prov_rows if r[0] == s))] for s in D["suites"]}}

    # ---- residual feasible attacks under task_arg_LP: classify reason
    resid = [r for r in rows if r[4 + policy_names.index("task_arg_LP")]]
    out["arg_LP_residual"] = [r[:3] for r in resid]

    # ---- granularity frontier
    frontier = {}
    for sname, S in D["suites"].items():
        U = S["user_tasks"]; I = S["injection_tasks"]
        uids = list(U)
        Rs = {u: req_tools(U[u]["calls"]) for u in uids}

        def cost(G, members):
            return sum(feasible_tool(I[i]["alternatives"], G) for _ in members for i in I)

        clusters = [([u], Rs[u]) for u in uids]
        cur = sum(cost(G, m) for m, G in clusters)
        seq = [(len(clusters), cur)]
        while len(clusters) > 1:
            best = None
            for a, b in itertools.combinations(range(len(clusters)), 2):
                ma, Ga = clusters[a]; mb, Gb = clusters[b]
                G = Ga | Gb
                delta = cost(G, ma + mb) - cost(Ga, ma) - cost(Gb, mb)
                key = (delta, len(G))
                if best is None or key < best[0]:
                    best = (key, a, b, G)
            (_, a, b, G) = best
            merged = (clusters[a][0] + clusters[b][0], G)
            clusters = [c for j, c in enumerate(clusters) if j not in (a, b)] + [merged]
            cur = sum(cost(Gc, m) for m, Gc in clusters)
            seq.append((len(clusters), cur))
        frontier[sname] = {"k": [s[0] for s in seq][::-1], "feasible": [s[1] for s in seq][::-1],
                           "cases": len(uids) * len(I)}
    out["frontier"] = frontier

    json.dump(out, open(os.path.join(RES, "permission_analysis.json"), "w"), indent=1)

    # ---- figure: frontier
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    colors = {"workspace": "#2a6fb0", "travel": "#c2571a", "banking": "#2e8b57", "slack": "#8a3fa0"}
    for s, fr in frontier.items():
        ax.plot(fr["k"], np.array(fr["feasible"]) / fr["cases"], "-o", ms=3, label=s, color=colors[s])
    ax.set_xscale("log")
    ax.set_xlabel("number of static permission profiles k (log)")
    ax.set_ylabel("fraction of security cases still feasible")
    ax.set_title("Granularity frontier (greedy profile design, AgentDojo v1.2.2)")
    ax.grid(alpha=.3); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(RES, "fig_granularity_frontier.png"), dpi=160)

    print(json.dumps({k: out[k] for k in ["suites", "totals", "trifecta", "provenance"]}, indent=1))
    print("arg-LP residual:", out["arg_LP_residual"])


if __name__ == "__main__":
    main()
