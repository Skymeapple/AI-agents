# p039 — Disagreement, tractability, and the value of information in loss-of-control mitigation spending

Reproduce everything (CPU, ~10 s, seeded):

    python3 code/run_all.py > results/run_log.txt

Requires Python 3 with numpy, scipy, matplotlib.

- `code/model.py` — closed-form model (optimal spend, minimised cost, threshold belief, pooling, EVPI/EVSI, robust rules).
- `code/run_all.py` — experiments E0–E9; writes `results/results.json` and `results/fig1..fig6*.png`.
- `paper.md` — the paper. Every model-derived number in it is in `results/results.json`.

Sourced inputs (see paper Table 1): US DOT VSL 2024 = $13.7M; UN WPP 2024 world population 8.2B;
XPT medians (AI extinction by 2100): superforecasters 0.38%, domain experts 3%;
Grace et al. (2024) extinction/disempowerment question: median 5%, mean 16.2%;
McAleese (LessWrong, 2025 update) AI-safety funding estimate for 2024: ~$189M/yr.
Assumptions (k, beta, m, discounting, horizon) are labelled in code and paper.
