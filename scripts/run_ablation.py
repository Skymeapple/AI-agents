"""Ablation study for autoscout on the synthetic stream.

Runs the full offline stream under several arms, across seeds, and writes a
markdown table of the results. The arms answer the research plan's questions:

``frozen``          no iteration at all -- the seed agenda, forever (baseline)
``mutation``        parameter search only; learns nothing from the literature
``literature``      vocabulary mining only; never touches parameters
``full``            both, bandit-allocated (the default agent)
``full-naive-gate`` both, but plain hill-climbing acceptance (no bootstrap
                    interval, no holdout) -- what the statistical gate buys

The headline metric is triage utility share on the **sealed** split, which no
arm ever reads during its run. All of this is on synthetic data whose drift was
planted by the same author who wrote the agent; read the output as a test of
mechanism, not as evidence about real literature.

Run:
    AUTOSCOUT_DISABLE_LLM=1 python scripts/run_ablation.py [--seeds 5] [--out examples/autoscout-ablation.md]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from autoscout.bench import default_benchmarks  # noqa: E402
from autoscout.config import AgentConfig  # noqa: E402
from autoscout.gate import GatePolicy  # noqa: E402
from autoscout.loop import CycleOptions, run_cycle, score_all  # noqa: E402
from autoscout.proposers import LiteratureProposer, MutationProposer  # noqa: E402
from autoscout.state import StateDir  # noqa: E402

ARMS = {
    "mutation": (lambda: [MutationProposer()], GatePolicy()),
    "literature": (lambda: [LiteratureProposer()], GatePolicy()),
    "full": (lambda: [MutationProposer(), LiteratureProposer()], GatePolicy()),
    "full-naive-gate": (lambda: [MutationProposer(), LiteratureProposer()], GatePolicy(naive=True)),
}


def run_arm(name: str, seed: int, cycles: int, candidates: int) -> dict:
    make_proposers, policy = ARMS[name]
    with tempfile.TemporaryDirectory() as tmp:
        state = StateDir(Path(tmp) / "state")
        summaries = []
        for _ in range(cycles):
            opts = CycleOptions(seed=seed, candidates=candidates, policy=policy,
                                proposers=make_proposers())
            summaries.append(run_cycle(state, opts))
        last = summaries[-1]
        return {
            "sealed_share": last["scores"]["triage"]["sealed"]["utility_share"],
            "tune_share": last["scores"]["triage"]["tune"]["utility_share"],
            "sealed_load": last["scores"]["triage"]["sealed"]["reading_load"],
            "discovery_sealed": last["scores"]["discovery"]["sealed"]["mean"],
            "promotions": sum(1 for s in summaries if s["promoted"]),
            "evaluations": sum(s["evaluated"] for s in summaries),
            "late_online_recall": float(np.mean([s["online_oracle"]["recall"] for s in summaries[-4:]])),
        }


def frozen() -> dict:
    scores = score_all(AgentConfig.from_agenda(), default_benchmarks())
    return {
        "sealed_share": scores["triage"]["sealed"]["utility_share"],
        "tune_share": scores["triage"]["tune"]["utility_share"],
        "sealed_load": scores["triage"]["sealed"]["reading_load"],
        "discovery_sealed": scores["discovery"]["sealed"]["mean"],
        "promotions": 0,
        "evaluations": 0,
        "late_online_recall": float("nan"),
    }


def fmt(values: list[float], digits: int = 3) -> str:
    arr = np.array(values, dtype=float)
    if np.all(np.isnan(arr)):
        return "-"
    if len(arr) == 1 or np.allclose(arr, arr[0]):
        return f"{arr[0]:.{digits}f}"
    return f"{arr.mean():.{digits}f} ± {arr.std(ddof=1):.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--out", default="examples/autoscout-ablation.md")
    args = parser.parse_args()
    os.environ["AUTOSCOUT_DISABLE_LLM"] = "1"

    results: dict[str, list[dict]] = {"frozen": [frozen()]}
    for arm in ARMS:
        results[arm] = []
        for seed in range(args.seeds):
            results[arm].append(run_arm(arm, seed, args.cycles, args.candidates))
            print(f"{arm} seed {seed}: {results[arm][-1]}", file=sys.stderr)

    keys = [("sealed_share", "sealed utility share"), ("tune_share", "tune utility share"),
            ("sealed_load", "sealed reading load"), ("late_online_recall", "online recall, last 4 weeks"),
            ("discovery_sealed", "discovery sealed"), ("promotions", "promotions"),
            ("evaluations", "candidates evaluated")]
    lines = [
        "# autoscout ablation (synthetic stream)", "",
        f"{args.cycles} cycles over the 12-week synthetic stream, {args.candidates} candidates "
        f"per cycle, {args.seeds} seeds per arm (mean ± sd). LLM proposer disabled. "
        "Generated by `scripts/run_ablation.py`.", "",
        "**Synthetic data, planted drift.** This tests whether the mechanism works, "
        "not how it performs on real papers.", "",
        "| arm | " + " | ".join(label for _, label in keys) + " |",
        "| --- | " + " | ".join("---" for _ in keys) + " |",
    ]
    for arm, runs in results.items():
        cells = [fmt([r[k] for r in runs], 0 if k in ("promotions", "evaluations") else 3)
                 for k, _ in keys]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
