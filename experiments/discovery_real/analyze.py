"""Analysis for PROTOCOL.md sections 6-7.

    python experiments/discovery_real/analyze.py freeze-baseline   # before AI arms run
    python experiments/discovery_real/analyze.py report            # after
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import T_C, T_R, T_S, cost  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
BASELINES, AI_ARMS = ["B0", "B1", "B2"], ["A1", "A2", "A3"]
HUMAN_ONLY_S_PER_TITLE = 12.0


def load(dataset: str, arm: str) -> dict | None:
    p = OUT / f"{dataset}_{arm}.json"
    return json.loads(p.read_text()) if p.exists() else None


def flags_of(rep: dict) -> np.ndarray:
    return np.unpackbits(np.frombuffer(bytes.fromhex(rep["flags"]), dtype=np.uint8))[: rep["n"]].astype(bool)


def costs(rec: dict, q: str = "q90", **k) -> np.ndarray:
    if not k:
        return np.array([r[q]["total_usd"] for r in rec["replicates"]])
    qv = {"q90": 0.90, "q95": 0.95}[q]
    return np.array([cost(flags_of(r), r["n_clusters"], r["wall_s"], qv, **k)["total_usd"]
                     for r in rec["replicates"]])


def freeze_baseline() -> None:
    frozen = {}
    for d in ["BIO", "MAG"]:
        means = {a: costs(load(d, a)).mean() for a in BASELINES if load(d, a)}
        best = min(means, key=means.get)
        frozen[d] = {"best_baseline": best, "mean_q90_usd": means}
    (HERE / "BASELINE_FROZEN.json").write_text(json.dumps(frozen, indent=1))
    print(json.dumps(frozen, indent=1))


def boot_ci(ratio: np.ndarray, n: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    samples = ratio[rng.integers(0, len(ratio), (n, len(ratio)))]
    g = np.exp(np.log(samples).mean(1))
    return float(np.percentile(g, 2.5)), float(np.percentile(g, 97.5))


def table_row(name: str, rec: dict) -> str:
    r = rec["replicates"]
    m = lambda f: np.mean([f(x) for x in r])  # noqa: E731
    return (f"| {name} | {rec['param']} | {m(lambda x: x['metrics']['nmi']):.3f} "
            f"| {m(lambda x: x['metrics']['ari']):.3f} | {m(lambda x: x['metrics']['hungarian_acc']):.3f} "
            f"| {m(lambda x: x['q90']['raw_acc']):.3f} | {m(lambda x: x['metrics']['coverage']):.2f} "
            f"| {m(lambda x: x['n_clusters']):.0f} | {m(lambda x: x['metrics']['recovery_rate']):.2f}/"
            f"{m(lambda x: x['metrics']['mean_purity']):.2f} "
            f"| {m(lambda x: x['wall_s']):.1f} | {rec['peak_rss_mb']:.0f} "
            f"| {m(lambda x: x['q90']['human_hours']):.2f} | ${m(lambda x: x['q90']['total_usd']):.2f} "
            f"| ${m(lambda x: x['q95']['total_usd']):.2f} |")


def report() -> None:
    frozen = json.loads((HERE / "BASELINE_FROZEN.json").read_text())
    lines = []
    for d in ["BIO", "MAG"]:
        recs = {a: load(d, a) for a in BASELINES + AI_ARMS if load(d, a)}
        lines += [f"\n### {d}\n",
                  "| Arm | param | NMI | ARI | Hung. ACC | raw acc (maj.) | coverage | clusters "
                  "| recovery/purity | wall s | peak MB | analyst h (Q90) | total $ Q90 | total $ Q95 |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        lines += [table_row(a, r) for a, r in recs.items()]
        n = recs["B0"]["replicates"][0]["n"]
        k = {"BIO": 20, "MAG": 7}[d]
        human_only = (n * HUMAN_ONLY_S_PER_TITLE + k * T_C) / 3600 * 60
        lines.append(f"\nHuman-only reference (manual coding, 12 s/title): ${human_only:.2f}\n")

        base = frozen[d]["best_baseline"]
        ai_present = [a for a in AI_ARMS if a in recs]
        if not ai_present:
            continue
        best_ai = min(ai_present, key=lambda a: costs(recs[a]).mean())
        cb, ca = costs(recs[base]), costs(recs[best_ai])
        ratio = cb / ca
        g = float(np.exp(np.log(ratio).mean()))
        lo, hi = boot_ci(ratio)
        wins = int((ca < cb).sum())
        real = wins == len(cb) and lo > 1.0
        cb95, ca95 = costs(recs[base], "q95"), costs(recs[best_ai], "q95")
        lines += [f"**Frozen best baseline: {base}. Best AI arm: {best_ai}.**\n",
                  f"- Paired cost ratio per seed: {', '.join(f'{x:.2f}' for x in ratio)}",
                  f"- G (geometric mean) = **{g:.2f}x**, 95% bootstrap CI [{lo:.2f}, {hi:.2f}]; "
                  f"AI cheaper on {wins}/{len(cb)} seeds -> real gain: **{'YES' if real else 'NO'}**",
                  f"- Q95: G = {np.exp(np.log(cb95 / ca95).mean()):.2f}x",
                  f"- Distance to 10x: **{10 / g:.1f}x** further reduction needed",
                  f"- vs human-only: baseline {human_only / cb.mean():.1f}x cheaper, "
                  f"AI {human_only / ca.mean():.1f}x cheaper\n"]

        # Stage breakdown and Amdahl ceilings for the best AI arm.
        parts = {p: np.mean([r["q90"]["parts_usd"][p] for r in recs[best_ai]["replicates"]])
                 for p in ["machine", "review", "screen", "fix"]}
        bparts = {p: np.mean([r["q90"]["parts_usd"][p] for r in recs[base]["replicates"]])
                  for p in parts}
        total = sum(parts.values())
        lines += ["| Stage | baseline $ | AI $ | AI share | G if this stage cost 0 |", "|---|---|---|---|---|"]
        for p in parts:
            lines.append(f"| {p} | {bparts[p]:.2f} | {parts[p]:.2f} | {parts[p] / total:.0%} "
                         f"| {cb.mean() / (total - parts[p]):.2f}x |")
        human_floor = parts["review"]
        lines.append(f"\nIf screening and fixing were free (perfect AI), the cost floor is the cluster "
                     f"review alone: G ceiling = {cb.mean() / (parts['machine'] + human_floor):.1f}x.\n")

        # Sensitivity (PROTOCOL section 7).
        lines += ["| Sensitivity | G |", "|---|---|"]
        for name, base_v in [("t_c", T_C), ("t_s", T_S), ("t_r", T_R)]:
            for f in (0.5, 2.0):
                kw = {"t_c": T_C, "t_s": T_S, "t_r": T_R, name: base_v * f}
                gs = np.exp(np.log(costs(recs[base], **kw) / costs(recs[best_ai], **kw)).mean())
                lines.append(f"| {name} x{f} | {gs:.2f}x |")
    text = "\n".join(lines)
    (HERE / "out/report_tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    {"freeze-baseline": freeze_baseline, "report": report}[sys.argv[1]]()
