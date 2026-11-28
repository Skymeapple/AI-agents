# Exploratory, unfinished, NOT used in paper.md

These files were left by an interrupted revision session. They explore a "first-order speed limit"
(polynomial-method lower bounds for momentum / Chebyshev / CG) and a heavy-ball overshoot check.
No theorem in `paper.md` depends on them, the runs were cut off before the main output
(`linear_speedlimit.json`) was written, and the corresponding results in
`results/exploratory_unfinished/` are partial (`ls_part_*_log.txt` are truncated logs; `ls_test.json`
is a 2-seed smoke test; `ls_hbrho.json` and `prop4_check.json` are complete but unreviewed).
Nothing here has been checked against a written proof. The scripts write to `../results/` relative
to the old location, so paths must be adjusted before re-running.
