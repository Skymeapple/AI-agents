# Closed discovery loop

_16 directions, 768 claims available, 192 checked._

**Claim source: bank.** Synthetic. Candidates are real verifier payloads; their attachment to works and each direction's yield are planted. See scripts/make_claim_bank.py. The ranking below shows the mechanism working; it says nothing about the real fields these directions are named after.

## Directions re-ranked by verified priority

Verified priority is sciscout's priority times the share of the direction's claims estimated to be true. The interval is the 90% credible range on that share. A direction with no checks sits at the prior mean of one half, with an interval that says so.

| # | Direction | Was | Priority | Checks | Passed | Yield (90%) | Verified priority |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | error / correction / quantum | 3 | 0.65 | 48 | 42 | 0.96 (0.88-1.00) | 0.63 (0.57-0.65) |
| 2 | sodium / solid-state / batteries | 4 | 0.64 | 48 | 30 | 0.73 (0.59-0.86) | 0.46 (0.38-0.55) |
| 3 | carbon / capture / air | 6 | 0.62 | 33 | 20 | 0.69 (0.53-0.85) | 0.43 (0.33-0.53) |
| 4 | topological / qubits / materials | 13 | 0.52 | 6 | 5 | 0.79 (0.49-0.98) | 0.41 (0.26-0.51) |
| 5 | hydrogen / electrolysis / catalysts | 10 | 0.59 | 12 | 6 | 0.56 (0.29-0.83) | 0.33 (0.17-0.49) |
| 6 | fermentation / proteins / precision | 12 | 0.54 | 6 | 3 | 0.54 (0.19-0.89) | 0.29 (0.10-0.47) |
| 7 | perovskite / tandem / photovoltaics | 7 | 0.62 | 5 | 2 | 0.45 (0.12-0.82) | 0.28 (0.07-0.50) |
| 8 | crops / field / gene-edited | 14 | 0.52 | 0 | 0 | 0.50 (0.05-0.95) | 0.26 (0.03-0.49) |
| 9 | superconductivity / transition / ambient-pressure | 15 | 0.50 | 0 | 0 | 0.50 (0.05-0.95) | 0.25 (0.03-0.48) |
| 10 | targeted / degradation / vitro | 9 | 0.59 | 4 | 3 | 0.41 (0.03-0.91) | 0.24 (0.02-0.54) |
| 11 | design / structure / models | 1 | 0.68 | 17 | 13 | 0.33 (0.02-0.83) | 0.22 (0.01-0.56) |
| 12 | solid / tumour / car-t | 11 | 0.59 | 4 | 1 | 0.36 (0.07-0.73) | 0.21 (0.04-0.43) |
| 13 | editing / base / prime | 5 | 0.63 | 2 | 0 | 0.31 (0.02-0.77) | 0.19 (0.01-0.48) |
| 14 | therapeutics / gut / microbiome | 16 | 0.50 | 1 | 0 | 0.37 (0.03-0.85) | 0.18 (0.01-0.43) |
| 15 | photonic / computing / neuromorphic | 8 | 0.61 | 2 | 0 | 0.29 (0.02-0.73) | 0.18 (0.01-0.45) |
| 16 | emulators / learning / machine | 2 | 0.67 | 4 | 0 | 0.21 (0.01-0.57) | 0.14 (0.01-0.38) |

## How much a pass is worth

Error rates measured in `aisci-verifiers/results/i1_calibration.json`. The last column is the chance a passing claim is true when half of all claims are.

| Verifier | Wrong candidates passed | Correct candidates rejected | A pass is right |
| --- | --- | --- | --- |
| matmul_tensor | 0.02 | 0.20 | 97% |
| python_tests[strong tests] | 0.10 | 0.33 | 87% |
| python_tests[weak tests] | 0.70 | 0.33 | 49% |
| symbolic_identity | 0.03 | 0.11 | 97% |

The rates are posterior means under a uniform prior on small calibration sets (one to forty cases), so they are deliberately not zero where no error was seen.

## Record

192 checks recorded (67 negative, 125 success); hash chain intact; 0 issued but unwritten.

## Does closing the loop help?

Each policy spends 192 checks on the same 768 claims across 16 directions; 20 seeds. Value is the sciscout priority of each true claim confirmed, summed. Trap share is the share of checks spent on the two directions planted with low yield and lenient checking.

| Policy | Value found | Share of oracle | Trap share | Loop's lead (90%) |
| --- | --- | --- | --- | --- |
| loop (this module) | 70.2 | 80% | 12% | - |
| loop believing every pass | 58.3 | 66% | 29% | +11.8 (+11.3 to +12.3) |
| sciscout priority only | 52.6 | 60% | 14% | +17.5 (+16.6 to +18.5) |
| uniform | 53.1 | 60% | 13% | +17.0 (+16.1 to +17.9) |
| oracle (knows true yields) | 87.9 | 100% | 0% | -17.7 (-18.1 to -17.4) |

The bank's yields and traps were planted by the author of the loop, so this shows the mechanism does what it claims on a case built to test it. It is not evidence about real research claims.

## Where the numbers came from

| Input | Value | Grade | Basis |
| --- | --- | --- | --- |
| design / structure / models: priority | 0.6778 | observed | sciscout priority; 88% of it rests on evidence |
| design / structure / models: yield | 0.3281 | observed | 13 of 17 checks passed, weighted by each verifier's measured error rates |
| emulators / learning / machine: priority | 0.6672 | observed | sciscout priority; 88% of it rests on evidence |
| emulators / learning / machine: yield | 0.2072 | observed | 0 of 4 checks passed, weighted by each verifier's measured error rates |
| error / correction / quantum: priority | 0.6544 | observed | sciscout priority; 88% of it rests on evidence |
| error / correction / quantum: yield | 0.9556 | observed | 42 of 48 checks passed, weighted by each verifier's measured error rates |
| sodium / solid-state / batteries: priority | 0.6357 | observed | sciscout priority; 88% of it rests on evidence |
| sodium / solid-state / batteries: yield | 0.731 | observed | 30 of 48 checks passed, weighted by each verifier's measured error rates |
| editing / base / prime: priority | 0.6279 | observed | sciscout priority; 88% of it rests on evidence |
| editing / base / prime: yield | 0.3062 | observed | 0 of 2 checks passed, weighted by each verifier's measured error rates |
| carbon / capture / air: priority | 0.6207 | observed | sciscout priority; 88% of it rests on evidence |
| carbon / capture / air: yield | 0.6935 | observed | 20 of 33 checks passed, weighted by each verifier's measured error rates |
| perovskite / tandem / photovoltaics: priority | 0.6187 | observed | sciscout priority; 88% of it rests on evidence |
| perovskite / tandem / photovoltaics: yield | 0.4492 | observed | 2 of 5 checks passed, weighted by each verifier's measured error rates |
| photonic / computing / neuromorphic: priority | 0.6111 | observed | sciscout priority; 88% of it rests on evidence |
| photonic / computing / neuromorphic: yield | 0.2919 | observed | 0 of 2 checks passed, weighted by each verifier's measured error rates |
| targeted / degradation / vitro: priority | 0.5918 | observed | sciscout priority; 88% of it rests on evidence |
| targeted / degradation / vitro: yield | 0.4113 | observed | 3 of 4 checks passed, weighted by each verifier's measured error rates |
| hydrogen / electrolysis / catalysts: priority | 0.5914 | observed | sciscout priority; 88% of it rests on evidence |
| hydrogen / electrolysis / catalysts: yield | 0.5596 | observed | 6 of 12 checks passed, weighted by each verifier's measured error rates |
| solid / tumour / car-t: priority | 0.5912 | observed | sciscout priority; 88% of it rests on evidence |
| solid / tumour / car-t: yield | 0.3576 | observed | 1 of 4 checks passed, weighted by each verifier's measured error rates |
| fermentation / proteins / precision: priority | 0.5367 | observed | sciscout priority; 88% of it rests on evidence |
| fermentation / proteins / precision: yield | 0.5398 | observed | 3 of 6 checks passed, weighted by each verifier's measured error rates |
| topological / qubits / materials: priority | 0.521 | observed | sciscout priority; 88% of it rests on evidence |
| topological / qubits / materials: yield | 0.7895 | observed | 5 of 6 checks passed, weighted by each verifier's measured error rates |
| crops / field / gene-edited: priority | 0.5195 | observed | sciscout priority; 88% of it rests on evidence |
| crops / field / gene-edited: yield | 0.5 | assumed | no claims checked yet; uniform prior **needs review** |
| superconductivity / transition / ambient-pressure: priority | 0.5039 | observed | sciscout priority; 88% of it rests on evidence |
| superconductivity / transition / ambient-pressure: yield | 0.5 | assumed | no claims checked yet; uniform prior **needs review** |
| therapeutics / gut / microbiome: priority | 0.5021 | observed | sciscout priority; 88% of it rests on evidence |
| therapeutics / gut / microbiome: yield | 0.3662 | observed | 0 of 1 checks passed, weighted by each verifier's measured error rates |
| matmul_tensor: false pass | 0.02381 | observed | 0 of 40 wrong candidates passed in aisci calibration, posterior mean under a uniform prior |
| matmul_tensor: false reject | 0.2 | observed | 0 of 3 correct candidates rejected in aisci calibration, posterior mean under a uniform prior |
| python_tests[strong tests]: false pass | 0.1 | observed | 0 of 8 wrong candidates passed in aisci calibration, posterior mean under a uniform prior |
| python_tests[strong tests]: false reject | 0.3333 | observed | 0 of 1 correct candidates rejected in aisci calibration, posterior mean under a uniform prior |
| python_tests[weak tests]: false pass | 0.7 | observed | 6 of 8 wrong candidates passed in aisci calibration, posterior mean under a uniform prior |
| python_tests[weak tests]: false reject | 0.3333 | observed | 0 of 1 correct candidates rejected in aisci calibration, posterior mean under a uniform prior |
| symbolic_identity: false pass | 0.02857 | observed | 0 of 33 wrong candidates passed in aisci calibration, posterior mean under a uniform prior |
| symbolic_identity: false reject | 0.1111 | observed | 0 of 7 correct candidates rejected in aisci calibration, posterior mean under a uniform prior |

