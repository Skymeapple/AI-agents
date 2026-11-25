# References: Wireless 400-Cell Synthetic MILP Case Solver

Verification: the shared web-search budget ran out before this project was searched. Every entry below was confirmed in this session by fetching its arXiv record (title, authors, date). Items that could not be confirmed are under "Unverified leads". The HFM instance format is unpublished, so relevance is judged against the README description only (one 400-cell MILP, Python with NumPy, SciPy and HiGHS, no network).

## Theory and established methods

- **Huangfu, Hall (2015).** Parallelizing the dual revised simplex method. arXiv:1503.01889. https://arxiv.org/abs/1503.01889
  - The simplex implementation behind HiGHS, the only MILP solver available in the task sandbox; explains what warm starts and repeated LP solves cost.
- **Ye, Rong, Chen, Al-Shalash et al. (2012).** User Association for Load Balancing in Heterogeneous Cellular Networks. arXiv:1205.2833. https://arxiv.org/abs/1205.2833
  - Classic cell-association formulation: the network-wide association problem is NP-hard and is handled through relaxations; the likely structure of a 400-cell case.
- **Bengio, Lodi, Prouvost (2018).** Machine Learning for Combinatorial Optimization: a Methodological Tour d'Horizon. arXiv:1811.06128. https://arxiv.org/abs/1811.06128
  - Framework for where learning can sit inside a MILP solver (branching, heuristics, configuration).
- **Song, Lanka, Yue, Dilkina (2020).** A General Large Neighborhood Search Framework for Solving Integer Linear Programs. arXiv:2004.00422. https://arxiv.org/abs/2004.00422
  - Large-neighbourhood search that repeatedly frees a subset of variables and re-solves with an off-the-shelf solver; the loop itself needs no training and fits a one-instance, no-network setting with HiGHS as the sub-solver.
- **Applegate, Díaz, Hinder, Lu et al. (2021).** Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient. arXiv:2106.04756. https://arxiv.org/abs/2106.04756
  - PDLP: a first-order LP method built from matrix-vector products, so it can be written in NumPy if the LP relaxation is too slow in simplex.

## Recent advances and breakthroughs (2023-2026)

- **Kempke, Koch (2025).** Fix-and-Propagate Heuristics Using Low-Precision First-Order LP Solutions for Large-Scale Mixed-Integer Linear Optimization. arXiv:2503.10344. https://arxiv.org/abs/2503.10344
  - Shows low-accuracy LP solutions are good enough to drive a fix-and-propagate primal heuristic on MIPLIB 2017; a cheap way to get a first feasible solution to warm-start HiGHS.
- **Han, Yang, Chen, Zhou et al. (2023).** A GNN-Guided Predict-and-Search Framework for Mixed-Integer Linear Programming. arXiv:2302.05636. https://arxiv.org/abs/2302.05636
  - Predicts variable values, then searches a trust region around the prediction with a solver; the search step works with any predictor, including a greedy heuristic.
- **Huang, Ferber, Tian, Dilkina et al. (2023).** Searching Large Neighborhoods for Integer Linear Programs with Contrastive Learning. arXiv:2302.01578. https://arxiv.org/abs/2302.01578
  - Learned choice of which variables to free in each large-neighbourhood step; describes the expert (local branching) it imitates, which can be run directly.
- **Cai, Huang, Dilkina (2024).** Learning Backdoors for Mixed Integer Linear Programs with Contrastive Learning. arXiv:2401.10467. https://arxiv.org/abs/2401.10467
  - Finds small sets of variables to branch on first; the idea of prioritising a few key binaries transfers to cell on/off or assignment variables.
- **Paulus, Krause (2023).** Learning To Dive In Branch And Bound. arXiv:2301.09943. https://arxiv.org/abs/2301.09943
  - Diving heuristics tailored to an instance family; diving on the LP relaxation can be coded on top of SciPy's LP interface.
- **Scavuzzo, Aardal, Lodi, Yorke-Smith (2024).** Machine Learning Augmented Branch and Bound for Mixed Integer Linear Programming. arXiv:2402.05501. https://arxiv.org/abs/2402.05501
  - Current survey of learned branching, cut selection, primal heuristics and solver configuration.
- **Lu, Yang (2023).** cuPDLP.jl: A GPU Implementation of Restarted Primal-Dual Hybrid Gradient for Linear Programming in Julia. arXiv:2311.12180. https://arxiv.org/abs/2311.12180
  - Details the restart and step-size rules that make PDLP competitive; needed for a from-scratch NumPy version.
- **Bolusani, Besançon, Bestuzheva, Chmiela et al. (2024).** The SCIP Optimization Suite 9.0. arXiv:2402.17702. https://arxiv.org/abs/2402.17702
  - Describes the heuristics and presolve of the leading open-source MILP solver; a reference for methods HiGHS lacks and for offline comparison runs.

## Datasets and benchmarks

- **Huang, Huang, Ferber, Dilkina (2024).** Distributional MIPLIB: a Multi-Domain Library for Advancing ML-Guided MILP Methods. arXiv:2406.06954. https://arxiv.org/abs/2406.06954
  - Curated MILP instance distributions by domain and hardness with standard test sets; stand-in problems for practice until HFM publishes its format.
- **Gasse, Cappart, Charfreitag, Charlin et al. (2022).** The Machine Learning for Combinatorial Optimization Competition (ML4CO): Results and Insights. arXiv:2203.02433. https://arxiv.org/abs/2203.02433
  - Public instance sets and a scoring scheme with primal, dual and configuration tracks; shows how a composite MILP reward is typically built.

## Open-source code and tools

- **Prouvost, Dumouchelle, Scavuzzo, Gasse et al. (2020).** Ecole: A Gym-like Library for Machine Learning in Combinatorial Optimization Solvers. arXiv:2011.06069. https://arxiv.org/abs/2011.06069
  - Instance generators (set cover, facility location and others) and a solver-in-the-loop environment for offline experiments.

## Breakthroughs in related fields that could transfer

- **Lee, Yu, Li (2019).** Learning to Branch: Accelerating Resource Allocation in Wireless Networks. arXiv:1903.01819. https://arxiv.org/abs/1903.01819
  - Learned pruning in branch and bound for mixed-integer wireless resource allocation; evidence that these problems have exploitable, repeated structure.
- **Shen, Shi, Zhang, Letaief (2018).** LORM: Learning to Optimize for Resource Management in Wireless Networks with Few Training Samples. arXiv:1812.07998. https://arxiv.org/abs/1812.07998
  - Imitation of branch-and-bound decisions for wireless MINLPs with few samples.
- **Shen, Shi, Zhang, Letaief (2020).** Graph Neural Networks for Scalable Radio Resource Management: Architecture Design and Theoretical Analysis. arXiv:2007.07632. https://arxiv.org/abs/2007.07632
  - Shows wireless allocation problems are permutation-equivariant on the interference graph; supports decomposing the 400 cells by interference cluster.
- **Novikov, Vũ, Eisenberger, Dupont et al. (2025).** AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv:2506.13131. https://arxiv.org/abs/2506.13131
  - LLM-driven evolution of whole programmes against an automatic score; the deliverable here is a single self-contained programme scored by a fixed reward, which is the setting that method targets.
- **Nair, Bartunov, Gimeno, von Glehn et al. (2020).** Solving Mixed Integer Programs Using Neural Networks. arXiv:2012.13349. https://arxiv.org/abs/2012.13349
  - Neural diving and neural branching; the "partial assignment then sub-MIP" step is usable with a hand-made assignment rule.

## Published results to beat

- None. HFM has published neither a baseline nor a best score for this track (README).

## Unverified leads

- MIPLIB 2017 (Gleixner et al., Mathematical Programming Computation 2021) and its instance tags for wireless or network design problems: not confirmed in session.
- CALMA and COST 259 frequency-assignment benchmarks (FAP web, Zuse Institute Berlin) and SNDlib network design instances: not confirmed in session.
- Fischetti, Lodi (2003) local branching; Danna, Rothberg, Le Pape (2005) RINS; Fischetti, Glover, Lodi (2005) feasibility pump: not confirmed in session.
- HiGHS documentation for `scipy.optimize.milp` options (time limit, MIP gap, no warm-start argument in SciPy's interface): needs checking against SciPy 1.16.1, because the README's warm-start plan depends on it.
