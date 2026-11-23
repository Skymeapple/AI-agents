# p035 — Bounding deployment behaviour of evaluation-aware models (revision 1)

Reproduce all numbers and figures (CPU only; fixed seeds):

```
cd /home/user/AI-agents/research/papers/p035
# round 0 (110 MLP organisms, ~4 min on 4 cores)
python3 -W ignore code/run_experiments.py results/raw.json
python3 code/analyze.py                       # -> results/summary.{txt,json}
python3 -W ignore code/analyze2.py            # -> results/diagnostics.json, fig1-3
# revision 1 (280 MLP organisms, ~21 min on 4 shared cores; 18 transformer organisms, ~6 min)
python3 -W ignore code/run_v2.py results/raw_v2.json
python3 -W ignore code/run_token.py results/raw_token.json
python3 -W ignore code/analyze_v2.py          # -> results/summary_v2.txt, analysis_v2.json, fig4, fig5
python3 code/analyze_token.py                 # -> results/summary_token.{txt,json}
```

Files
- `code/organism.py` round-0 world, MLP organism, estimators, diagnostics
- `code/organism_v2.py` revision: saturation regimes, interaction detector, tent (A2-violating) policy,
  six rewriters (perfect/partial/overshoot/task-altering), A3 check on benign traffic, stronger
  white-box baselines (risky-target steering at L1/L2, mean matching, MRDE), within-config diagnostics
- `code/token_organism.py` 2-layer transformer organisms on token sequences (order-dependent cues)
- `code/analyze_v2.py` stratified (within-configuration) diagnostic analysis, LB validity without slack,
  and re-analysis of round-0 raw.json

Dependencies: numpy, scipy, scikit-learn 1.9, matplotlib, torch (CPU).
