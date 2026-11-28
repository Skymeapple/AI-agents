"""The trajectory report: what the agent has learned, and whether it is real."""

from __future__ import annotations

from .config import AgentConfig
from .state import StateDir


def _score(scores: dict, bench: str, split: str, key: str = "mean") -> str:
    value = scores.get(bench, {}).get(split, {}).get(key)
    return "-" if value is None else f"{value:.3f}"


def render(state: StateDir, seed: AgentConfig) -> str:
    cycles = state.read_jsonl("cycles.jsonl")
    history = state.read_jsonl("history.jsonl")
    champion, scores = state.load_champion()
    knowledge = state.load_knowledge()

    lines = [
        "# autoscout trajectory",
        "",
        "Regenerated every cycle. `tune` is what proposals are selected on, `holdout` "
        "gates the one finalist per cycle, `sealed` is never used in any decision. "
        "If tune rises and sealed does not, the agent is overfitting its benchmark.",
        "",
        "Triage scores are mean per-paper utility (hit +1, wrong-topic hit +0.5, "
        "false alarm -0.5); `share` is that as a fraction of the best achievable. "
        "`online recall` is measured on the papers the agent actually harvested, "
        "and only exists for the synthetic stream, which carries hidden labels.",
        "",
        "| cycle | source | new | flagged | mined | evaluated | passed | promoted | "
        "triage tune | holdout | sealed | sealed share | reading load | online recall |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for c in cycles:
        s = c["scores"]
        oracle = c.get("online_oracle") or {}
        promoted = c["promoted"]["proposer"] if c.get("promoted") else ""
        lines.append(
            f"| {c['cycle']} | {c['source']} | {c['new_papers']} | {c['flagged']} | "
            f"{c['mined_terms']} | {c['evaluated']} | {c['passed_tune']} | {promoted} | "
            f"{_score(s, 'triage', 'tune')} | {_score(s, 'triage', 'holdout')} | "
            f"{_score(s, 'triage', 'sealed')} | {_score(s, 'triage', 'sealed', 'utility_share')} | "
            f"{_score(s, 'triage', 'sealed', 'reading_load')} | "
            f"{oracle.get('recall', '-')} |"
        )

    lines += ["", "## Current champion", "",
              f"Fingerprint `{champion.fingerprint()}`. Scores by benchmark and split:", ""]
    lines += ["| benchmark | split | mean | n | details |", "| --- | --- | --- | --- | --- |"]
    for bench, splits in scores.items():
        for split, values in splits.items():
            details = ", ".join(f"{k} {v}" for k, v in values.items() if k not in ("mean", "n"))
            lines.append(f"| {bench} | {split} | {values['mean']:.4f} | {values['n']} | {details} |")

    diff = seed.diff(champion)
    lines += ["", "### Changes from the seed agenda", ""]
    lines += [f"- {d}" for d in diff] if diff else ["- none yet"]

    lines += ["", "## Proposer track record", "",
              "Acceptance here means passing the tune stage; the bandit allocates "
              "evaluation budget on it.", "",
              "| proposer | tried | passed tune | rate | promoted |", "| --- | --- | --- | --- | --- |"]
    promotions: dict[str, int] = {}
    for c in cycles:
        if c.get("promoted"):
            promotions[c["promoted"]["proposer"]] = promotions.get(c["promoted"]["proposer"], 0) + 1
    for name, s in sorted(knowledge.proposer_stats.items()):
        rate = s["accepted"] / s["tried"] if s["tried"] else 0.0
        lines.append(f"| {name} | {s['tried']} | {s['accepted']} | {rate:.0%} | {promotions.get(name, 0)} |")

    holdout_rejections = [h for h in history if h.get("stage") == "holdout" and not h["accepted"]]
    lines += ["", "## Promotions blocked at holdout", "",
              "Candidates that won on tune and failed holdout -- the overfitting the "
              "two-stage gate exists to catch.", ""]
    lines += [f"- cycle {h['cycle']} ({h['proposer']}): {h['reason']}" for h in holdout_rejections] \
        or ["- none"]
    return "\n".join(lines) + "\n"
