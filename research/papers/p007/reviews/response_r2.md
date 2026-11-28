# Response to Reviewer (round 2)

Thank you for re-running the code, re-deriving the proofs and checking the references. The review found no fatal issue and five major weaknesses (W1–W5), plus nine minor points. Below, each is listed with exactly what changed. Where we could not resolve a point we say so.

**State of the revision before this response.** The round-2 revision had been started and interrupted. A read-only comparison of the two commits around the interruption showed that the only change on disk was the addition of `reviews/review_r2.md`; `paper.md`, `code/` and `results/` were still in their post-round-1 state and internally consistent (as your reproduction also found). Everything described below was therefore done in this pass.

**New scripts** (all run in this environment, CPU only; outputs in `results/`):

| Script | Output | Purpose |
|---|---|---|
| `exp_valrank.py` | `valrank_results.json` | oracle vs validation-selected vs fixed rank (W5) |
| `exp_headint.py` | `headint_results.json` | direct test of Cor. 3 (m3) |
| `exp_mlp_narrow.py` | `mlp_narrow_results.json` | ReLU control with no private features (W2) |
| `exp_hetero.py` | `hetero_results.json` | linear analogue of the ReLU teacher with unequal feature strengths (W2) |
| `analyze_r3.py` | `analysis_r3.json` | E3 slope fragility (m7); ReLU ratio vs Cor. 2 formula (W2) |
| `exp_transformer_r3.py` | `transformer_runs_r3.jsonl`, `transformer_converge.jsonl` | seeds 2–4 for the whole transformer design (84 runs); 9000-step, 3-learning-rate control (27 runs) (W3) |
| `analyze_transformer_r3.py` | `transformer_summary_r3.json`, `figH_transformer_r3.png` | 5-seed statistics, sign tests, test-window bootstrap (W3) |

No existing script or result file was modified. `exp_transformer_r3.py check` confirms that the new code path reproduces a round-2 run bit for bit (torch 2.15.0). Two operational notes, for transparency. The machine restarted during the control runs; the five unfinished runs were relaunched, and a duplicated worker set appended five duplicate lines to `transformer_converge.jsonl`. The duplicates were verified identical to the first copies and removed, leaving 27 lines. The longest single run took 849 s on the shared machine.

---

## Major weaknesses

### W1. The research question is answered only inside the toy model; the real-data evidence is too small and mostly negative.

**We agree, and we cannot fix this with the compute available** (no GPU; a mid-scale test with tens of millions of parameters is out of reach). What we did instead is reduce the claims to what the evidence supports, and strengthen the small-scale evidence so that at least its statistics are sound.

- *Claims narrowed.* The Introduction ("Scope and claim") now says that the transformer level is far too small to speak about production-scale models, that video and action are not studied, and that the paper is a solvable instance of the capacity-competition picture, not an answer to where real unified models stop paying. The abstract ends with the same statement, as does the Conclusion. A new Limitations item ("The research question is not answered for real systems") says it again.
- *The Cor. 3 link is withdrawn.* You pointed out that the image-side competition is predicted by almost any capacity- or optimisation-sharing account, that Cor. 3 is about oracle width, ample capacity and per-modality heads, and that the effect shrinks to zero at the largest size. All three points are correct. Sec. 5.8 item 1 now withdraws both "what Cor. 3 predicts" and "$\phi\approx0$", and the status table row for Cor. 3 reads "not tested" for transformers. The abstract no longer says "compatible with the head-interference prediction".
- *"Persistent competition" is withdrawn.* With 5 seeds the largest model at $D$=100k shows −0.001 ± 0.002, and the per-seed size slope is negative in 5 of 5 seeds at both budgets. Sec. 5.8 item 2 now says the competition declines with size and is not persistent.
- *The audio/image asymmetry* remains unexplained; the paper says so.
- *Positioning against work at scale.* See W4: the paper now points to Han et al. [2026] for evidence at scale.

**Not resolved:** no mid-scale test, no video or action data, and P2, P3 and P5 remain untested on real data.

### W2. The theory's quantitative content does not transfer to the one-hidden-layer ReLU model.

**Agreed. The presentation is changed as you asked, and we add a new experiment and a small new result on the competition below $k_s$.**

- *Cor. 2 is now reported as a failed quantitative prediction.* The status table row reads "quantitative prediction fails (ratio 3–82% above $\Gamma$; competition at $M=8$); only the weak lower bound holds". Sec. 5.7 item (c) now says that respecting a floor is a weak test and that we do not count it as support. Item (f) gives the numbers from `analysis_r3.json`: the oracle-width ratio at $h=64$ exceeds the formula value by 3%, 29%, 82% ($n=1024$; $M=2,4,8$) and 15%, 17%, 68% ($n=4096$); the bootstrap CI contains the formula value in one of six cells; for $M=8$ both CIs lie above 1 at $h=64>k_s+Mk_p$. The abstract, Sec. 6 item 2, Limitations and the Conclusion were changed to match.
- *Competition below $k_s$: our previous guess was wrong.* We had guessed that a narrow shared layer suffers because it must serve the private features. A control experiment (`exp_mlp_narrow.py`; round-2 protocol, $n=4096$, widths 2/4/6, 4 seeds) with a teacher that has **no private features** still shows competition below $k_s$: 1.28/1.40 ($M=2$), 1.29/1.78 ($M=4$), 1.31/1.72 ($M=8$) at $h=2/4$. The same script with $k_p=6$ replicates the round-2 values (1.63, 2.03, 2.05 at $h=6$ against 1.65, 2.02, 2.05).
- *A partial explanation (new Remark 3).* Prop. 1's tie below $k_s$ depends on equal-strength features with orthonormal heads. Lemma 1 holds for any $W_m$, and Ky Fan's inequality gives $\Delta_{\rm pop}(r)\ge0$ at every $r$, with equality only if all modalities share a top-$r$ eigenspace. The ReLU teacher has Gaussian heads, so modalities weight the shared features differently. The linear model with heads from the same distribution (`exp_hetero.py`) predicts competition below $k_s$ for both teachers, competition at $k_s$ with private features, and a switch to pooling synergy at $k_s$ without them. That matches the sign of the ReLU point estimate in all 18 cells (17 clear; in one the ReLU CI [0.55, 1.55] includes 1).
- *What this does not do.* The magnitudes do not transfer: the linear analogue overstates the competition by a factor 1.03–1.35 at $h=2$, 1.65–2.3 at $h=4$ and 3.5–5.5 at $h=6$ with private features. We say this in Sec. 5.7 and in the status table ("sign explained …, magnitude not"). This is a step towards your option (b), not a nonlinear theory, and the elementary inequality in Remark 3 is not a new technique.

### W3. Statistical treatment of the transformer study.

**Agreed on every point. We added seeds, replaced the statistics, and added an optimisation control.**

- *Seeds.* Seeds 2–4 were run for the whole design (M=2 grid and P4 sweep), so every cell has 5 seeds (140 models). The Sec. 5.8 table is replaced by 5-seed means with seed SE, $t$-based 95% CIs on 4 degrees of freedom, and the number of positive seeds.
- *The "more than 2 SE in 7/8 cells" claim is withdrawn* in the text, with the reason you gave.
- *Paired sign count.* We confirm your 15 of 16 on the 2-seed data. With 5 seeds: $\Delta_{\rm image}>0$ in 36 of 40 comparisons (exact two-sided binomial $p=2\cdot10^{-7}$), with all four negatives in the 100k/$d_{\rm model}$=128 cell; audio 17 of 40 ($p=0.43$). Seed-level mean over the 8 cells: image +0.014 [+0.010, +0.019], audio −0.003 [−0.008, +0.001].
- *Test-sampling uncertainty.* For seeds 2–4 we stored per-window test losses and report a paired bootstrap over test windows: half-widths of about 0.004 (image, 160 windows) and 0.010 (audio, 98 windows). The image interval excludes zero in 7 of 8 cells; the audio interval in 3 of 8, with both signs. Seed and test-window uncertainty are reported separately, not combined, and the 16 cell-level intervals are not corrected for multiplicity; both caveats are in the text and in Limitations.
- *P4.* The conclusion is reworded from "not supported" to "not supported, and the test is weak … an underpowered null, not evidence against a modality budget". With 5 seeds the seed-paired changes from $M=2$ to $M=4$ are +0.000 [−0.015, +0.015] (image) and −0.007 [−0.018, +0.004] (audio), and −0.010 [−0.042, +0.022] for text from $M=3$ to $M=4$. The mean over modalities present rises (+0.005 to +0.014) only because the added modalities are themselves in competition. The two oddities you noted are addressed explicitly: the text drop is now +0.029 → +0.018, inside its CI, and the image drop has disappeared (+0.018 at both $M=2$ and $M=4$). Both were seed noise.
- *Learning rate and step cap.* We count 83 of 90 image-containing models with best image step ≥2400 and state that the size axis mixes capacity with optimisation. In one cell ($d_{\rm model}=32$, $D$=25k, seeds 0–2) we raised the cap to 9000 steps and selected among three learning rates per model and modality. No run reached the new cap. $\Delta_{\rm image}$ is +0.010 ± 0.003 (positive in 3 of 3 seeds, $t$-CI [−0.002, +0.022]) against +0.017 ± 0.003 for the same seeds under the main protocol, so about 40% of the effect in that cell was an optimisation effect. We report this in the abstract and in Sec. 5.8, and we no longer treat the size trend as a clean test of P1.

**Not resolved:** the converged control covers one cell with 3 seeds. Extending it to $d_{\rm model}=128$ would need single runs well over the 15-minute budget.

### W4. Missing closely related work.

**Added, after verifying each paper by web search.**

- Han, Tong, Fan, Chen, Torr, Kokkinos and Lewis (2026), arXiv:2608.05000: verified (arXiv listing; authors, title and abstract-level findings). Discussed in the Introduction, in a new Related Work paragraph ("Controlled studies of synergy and competition at scale"), in Sec. 6 and in the Conclusion. Like you, we could read only the abstract and summaries, and the paper says so; we do not claim that their results confirm or refute any formula here. We state that their study addresses P1- and P6-type questions at a scale we cannot reach and that our architectural discussion adds nothing empirical to theirs.
- Uni-X, Hao, Liu, Xiao, Huang and Yu, ICLR 2026, arXiv:2509.24365: verified (ICLR proceedings page). Discussed in Related Work and Sec. 6, with the caveat that gradient conflict is an optimisation quantity our static model does not contain.
- Khan (2026), "Capacity and Redundancy Trade-offs in Multi-Task Learning", arXiv:2607.16554, UAI 2026: verified (arXiv listing; the comments field gives UAI 2026). We describe it from its abstract as a capacity–redundancy identity and present Cor. 1 as its explicit form in one linear model. We could not verify the PMLR volume number, so the reference omits it.
- Ilbert, Tiomoko, Louart, Odonnat, Feofanov, Palpanas and Redko, NeurIPS 2024, arXiv:2406.10327: verified (NeurIPS proceedings). This is the random-matrix analysis with $W_t=W_0+V_t$ that you mention. We note that their finite-data results are rigorous where ours are a heuristic.
- The "Novelty, stated plainly" paragraph now says that the qualitative picture is established empirically and information-theoretically, and that our addition is closed-form constants in one linear instance.

### W5. Oracle-width convention; validation-selected rank not implemented.

**Implemented.** `exp_valrank.py` simulates the Exp. A configuration under three conventions: oracle rank, rank selected on an independent validation estimate ($n_{\rm val}=n/4$), and fixed rank (new table in Sec. 5.4).

- Validation-selected rank reproduces the oracle ratio to within 0.2% at every capacity for $n\ge1000$ and within 1.0% at $n=200$ (e.g. $r=24$, $n=1000$: 0.964 oracle, 0.963 validation). So the oracle convention is not what makes Cor. 2 optimistic.
- Fixed rank gives 0.553 ($r=20$) and 0.684 [0.681, 0.687] ($r=24$) at $n=1000$, as Remark 2 says. The fixed-rank df heuristic is accurate at $r=20$ (0.527) and 10–12% too low at $r=24$ (0.609); at $n=200$ it is 22–24% too low. We report this and did not investigate the cause.
- The text now states your point directly: published comparisons are fixed-size, so for them the relevant statement is Remark 2, not "bounded synergy" (Model section, Remark 2, Sec. 5.4, Sec. 6 item 2, Limitations, abstract).

---

## Minor issues

- **m1 (Cor. 2 needs $q\ge k_p$).** Stated in Cor. 2, with the factorisation that shows it and the remark that the bound is false without it. Sec. 5.7 notes that the ReLU configuration satisfies it ($q=8\ge k_p=6$).
- **m2 (Exp. D metric; "≤0.3%" vs 0.34%).** Sec. 5.4 now reports the ratio errors (21%, 8.9%, 11%, 4.3%, 1.8%, 0.26%, 0.22%, 0.34%), the same metric as the summary table, and explains that the previously quoted numbers were errors of the unified loss. The summary table lists 0.26% / 0.22% / 0.34% instead of "≤0.3%". The preamble says "unless a sentence says otherwise".
- **m3 (superseded Exp. G result cited for Cor. 3).** Replaced by a direct test (`exp_headint.py`, new paragraph in Sec. 5.6): five configurations, error at most 1.2% for $n\ge10^4$ and up to 8.1% at $n=10^3$; the $q=k_p$ configuration gives exactly 1, as predicted.
- **m4 (dangling "Sec. 5.5 gives the conversion").** Rewritten: $\kappa$ and $M/\kappa$ convert same-size to same-total-capacity inside the power-law model only, and no conversion to equal compute is given.
- **m5 (Exp. C does not restate $d$, $q$).** Added ($d=64$, $q=16$, $k_s=k_p=4$), also for the fine sweep.
- **m6 (random-design theory from the last repetition).** Stated, with the reason it is harmless.
- **m7 ($\alpha=1.5$ slope on 4 points).** Leave-one-out slopes computed (`analysis_r3.json`): 0.72–0.79 for $\alpha=1.5$, 0.549–0.573 for $\alpha=2$. The text now says the $\alpha=1.5$ fit spans less than a decade of $n$ and does not test the exponent.
- **m8 (specialist rank capped at $q$).** Stated in Remark 2.
- **m9 ($\phi\approx0$ asserted, not measured).** **Not resolved by measurement.** We did not run a CCA or probing estimate (the image and audio streams are unpaired, and trained models were not saved). Instead the assertion is removed and Sec. 5.8, the status table and Limitations say that $\phi$ was not measured and that the Cor. 3 link is not established.

---

## Summary of what remains open

1. No test above 442k parameters, and no video or action data (W1).
2. No quantitative account of the ReLU results: $\Gamma$ fails, and the competition below $k_s$ is explained in sign only (W2).
3. The converged, learning-rate-tuned control covers one transformer cell with 3 seeds (W3).
4. $\phi$, $\alpha$ and the irreducible loss are unmeasured for the transformer data, so P2, P3 and P5 are untested there (m9).
5. Han et al. [2026] was read at abstract level only (W4).
6. The 10–24% error of the fixed-rank df heuristic at $r=24$ is reported but unexplained (W5).

We accept the assessment that the significance is limited. The revision does not change that; it makes the paper say so consistently, and it replaces two claims that the evidence did not support.
