# References: Single-Machine Weighted Tardiness Scheduling with Release Times

Verification: the shared web-search budget ran out before this project was searched. Every entry below was confirmed in this session by fetching its arXiv record (title, authors, date). The classic papers on this problem (dynasearch, exact solvers, OR-Library instances) are not on arXiv; they are under "Unverified leads" and still need checking. This file is incomplete.

## Theory and established methods

- **Kramer, Subramanian (2015).** A unified heuristic and an annotated bibliography for a large class of earliness-tardiness scheduling problems. arXiv:1509.02384. https://arxiv.org/abs/1509.02384
  - Local-search metaheuristic covering single-machine weighted tardiness with release dates and idle time, with fast move evaluation, tested on hundreds of instances; its annotated bibliography is the best index to the literature and to public instance sets for this exact problem.
- **Kramer, Dell'Amico, Iori (2018).** Enhanced arc-flow formulations to minimize weighted completion time on identical parallel machines. arXiv:1808.10661. https://arxiv.org/abs/1808.10661
  - Arc-flow (time-indexed) exact formulations that scale further than plain time-indexed MILP; a route to exact optima for the 40-job instances to confirm they are solved.

## Recent advances and breakthroughs (2023-2026)

- **Bouška, Šůcha, Novák, Hanzálek (2023).** Deep learning-driven scheduling algorithm for a single machine problem minimizing the total tardiness. European Journal of Operational Research 308(3), 990-1006. https://arxiv.org/abs/2402.14847
  - A neural estimator of the objective guides Lawler-style decomposition in a single pass and generalises from 75-100 jobs to 800; an idea for a fast learned evaluator inside a one-second budget.
- **Liu, Tong, Yuan, Lin et al. (2024).** Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model. arXiv:2401.02051. https://arxiv.org/abs/2401.02051
  - LLM-evolved heuristic code; fits a track where 43 scored proposals gave no accepted improvement from manual parameter changes.
- **Ye, Wang, Cao, Berto et al. (2024).** ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution. arXiv:2402.01145. https://arxiv.org/abs/2402.01145
  - Evolves heuristic components (for example the priority index or the perturbation step) with reflection between generations.
- **Novikov, Vũ, Eisenberger, Dupont et al. (2025).** AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv:2506.13131. https://arxiv.org/abs/2506.13131
  - Evolves whole programmes against an automatic scorer; applicable because this task has a fixed scorer and a short run time.

## Breakthroughs in related fields that could transfer

- **Parmentier, T'Kindt (2021).** Learning to solve the single machine scheduling problem with release times and sum of completion times. arXiv:2101.01082. https://arxiv.org/abs/2101.01082
  - Maps a hard release-time instance to an easy one solved exactly, then transfers the schedule back; competitive on large instances and directly about release times.
- **Bouška, Novák, Šůcha, Módos et al. (2020).** Data-driven Algorithm for Scheduling with Total Tardiness. arXiv:2005.05579. https://arxiv.org/abs/2005.05579
  - Earlier version of the learned objective estimator for tardiness decomposition.
- **Chen, Tian (2018).** Learning to Perform Local Rewriting for Combinatorial Optimization. arXiv:1810.00337. https://arxiv.org/abs/1810.00337
  - Learns which part of a solution to rewrite and how, with a job-scheduling case; a learned replacement for random perturbation in iterated local search.
- **da Costa, Rhuggenaath, Zhang, Akcay (2020).** Learning 2-opt Heuristics for the Traveling Salesman Problem via Deep Reinforcement Learning. arXiv:2004.01608. https://arxiv.org/abs/2004.01608
  - Learned move selection in a sequencing local search; the same move types (swap, insert, 2-opt) are used on job sequences.
- **Zhang, Song, Cao, Zhang et al. (2020).** Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning. arXiv:2010.12367. https://arxiv.org/abs/2010.12367
  - Learned priority dispatching rules that generalise across instance sizes; a possible better seed than ATC.
- **Kwon, Choo, Kim, Yoon et al. (2020).** POMO: Policy Optimization with Multiple Optima for Reinforcement Learning. arXiv:2010.16011. https://arxiv.org/abs/2010.16011
  - Many diverse constructive starts in one batched pass; a way to seed multi-start search cheaply.

## Published results to beat

- Bouška et al., arXiv:2402.14847 (abstract): average optimality gap 0.26% on instances with up to 800 jobs for unweighted total tardiness without release times; a different and easier problem, given for scale only.
- This repository (README): iterated local search −12.5% against ATC with k = 2; 40-job instances appear solved. HFM incumbent gain: 1.211 on private instances.

## Unverified leads

- Congram, Potts, van de Velde (2002), "An iterated dynasearch algorithm for the single-machine total weighted tardiness scheduling problem", INFORMS Journal on Computing: not confirmed in session.
- Tanaka, Fujikuma, Araki (2009) and Tanaka, Fujikuma (2012), exact successive sublimation dynamic programming for single-machine problems, including idle time and release dates, with a public solver (SiPS): not confirmed in session.
- OR-Library weighted tardiness instances (wt40, wt50, wt100; Beasley) with best-known values: not confirmed in session.
- Akturk, Ozdemir (2000, 2001), dominance rules and instance generator for 1|r_j|sum w_j T_j: not confirmed in session.
- Vepsalainen, Morton (1987), apparent tardiness cost (ATC) rule, and Morton, Rachamadugu X-RM rule for release dates: not confirmed in session.
- Pessoa, Uchoa, Poggi de Aragão, Rodrigues (2010), exact branch-cut-and-price over arc-time-indexed formulations: not confirmed in session.
- Ding, Lü et al., breakout dynasearch for single-machine weighted tardiness: not confirmed in session.

## Second pass (confirmed by web search, 25 November 2026)

- **Congram, Potts, van de Velde (2002). An Iterated Dynasearch Algorithm for the Single-Machine Total Weighted Tardiness Scheduling Problem. INFORMS Journal on Computing 14(1).** https://pubsonline.informs.org/doi/fpi/10.1287/ijoc.14.1.52.7712
  - Searches an exponentially large neighbourhood of swaps in polynomial time by dynamic programming; the method to test against this project's compiled local search at equal time on 100-job instances.
- **Lourenco, Martin, Stutzle. Iterated Local Search (arXiv math/0102188).** https://arxiv.org/pdf/math/0102188
  - The reference description of the search scheme used in this project, including the backtracking restart rule.
- **Geiger (2009). The Single Machine Total Weighted Tardiness Problem - Is it (for Metaheuristics) a Solved Problem? (arXiv 0907.2990).** https://arxiv.org/pdf/0907.2990
  - Asks whether standard benchmarks are solved by metaheuristics, which matches this project's finding that 40-job instances are very likely solved.
- **Ding et al. (2016). Breakout dynasearch for the single-machine total weighted tardiness problem.** https://research.polyu.edu.hk/en/publications/breakout-dynasearch-for-the-single-machine-total-weighted-tardine/
  - Combines dynasearch with adaptive perturbation; a stronger candidate than plain iterated dynasearch.
- **Wei Jing (2009). Master's thesis on the weighted tardiness problem with release dates, McMaster University.** https://macsphere.mcmaster.ca/bitstream/11375/9504/1/fulltext.pdf
  - One of the few sources found on the release-date variant that this project addresses; contents not read.

