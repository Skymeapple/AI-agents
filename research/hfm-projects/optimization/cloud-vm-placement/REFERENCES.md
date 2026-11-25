# References: Online Cloud Resource Allocator for Azure Trace Replay

Verification: every entry was confirmed in this session, either by a web search result or by fetching the arXiv record. Items that could not be confirmed are under "Unverified leads".

## Theory and established methods

- **Hadary, Marshall, Menache, Pan, Greeff, Dion, Dorminey, Joshi, Chen, Russinovich, Moscibroda (2020).** Protean: VM Allocation Service at Scale. OSDI 2020, 845-861. https://www.usenix.org/conference/osdi20/presentation/hadary
  - Describes Azure's production allocator and its rule-based packing; the packing trace used here was released with this paper.
- **Cortez, Bonde, Muzio, Russinovich, Fontoura, Bianchini (2017).** Resource Central: Understanding and Predicting Workloads for Improved Resource Management in Large Cloud Platforms. SOSP 2017, 153-167. https://doi.org/10.1145/3132747.3132772
  - Characterises Azure VM workloads and describes a system that learns VM behaviour (including lifetime) offline and serves predictions online to resource managers; the starting point for a lifetime predictor.
- **Buchbinder, Fairstein, Mellou, Menache, Naor (2021).** Online Virtual Machine Allocation with Lifetime and Load Predictions. SIGMETRICS 2021 (arXiv:2011.06250, "Online Virtual Machine Allocation with Predictions"). https://arxiv.org/abs/2011.06250
  - Models placement as dynamic bin packing minimising total machine usage time, and studies what predictions of lifetime and of future load are worth; gives algorithms with guarantees to compare with.
- **Barbalho et al. (2023).** Virtual Machine Allocation with Lifetime Predictions. MLSys 2023 (Outstanding Paper). https://proceedings.mlsys.org/paper_files/paper/2023/hash/48eb2c79643df5cb7d125945238bd7d0-Abstract-mlsys2023.html
  - Lifetime Alignment: lifetime ranges split into classes, and VMs with similar predicted lifetimes are placed together; deployed in Azure. This is the rule the README's "lifetime classes" experiment approximates.
- **Mitzenmacher, Vassilvitskii (2020).** Algorithms with Predictions. arXiv:2006.09123. https://arxiv.org/abs/2006.09123
  - Framework for algorithms that gain from good predictions and degrade gracefully with bad ones; the right lens for the README's noise-sensitivity table.
- **Angelopoulos, Kamali, Shadkami (2021).** Online Bin Packing with Predictions. arXiv:2102.03311. https://arxiv.org/abs/2102.03311
  - Bin packing with predicted item frequencies and a tunable trust parameter; a design pattern for mixing best fit with a prediction-led rule.

## Recent advances and breakthroughs (2023-2026)

- **Ling, Worah, Wang, Kong et al. (2024, revised 2025).** LAVA: Lifetime-Aware VM Allocation with Learned Distributions and Adaptation to Mispredictions. MLSys 2025; arXiv:2412.09840. https://arxiv.org/abs/2412.09840
  - Predicts a lifetime distribution, re-predicts remaining lifetime as a VM survives, and moves a host up a lifetime class when it outlives its deadline; fills gaps only with VMs at least 10x shorter-lived. This directly targets the README's failure mode, where a one-shot median predictor erased the gain.
- **Google Research (blog).** Solving virtual machine puzzles: How AI is optimizing cloud computing. https://research.google/blog/solving-virtual-machine-puzzles-how-ai-is-optimizing-cloud-computing/
  - Plain-language description of the LAVA family (NILAS, LAVA, LARS) and how re-prediction is used in production.

## Datasets and benchmarks

- **Microsoft Azure.** AzurePublicDataset (GitHub). https://www.github.com/Azure/AzurePublicDataset
  - Holds AzureTracesForPacking2020 (VM requests with priority and lifetime, used here) plus the 2017 and 2019 VM workload traces (the 2019 trace covers about 2.6 million VMs and 1.9 billion utilisation readings); the 2017 and 2019 traces give more lifetime training data.
- **Google.** cluster-data, ClusterData2019 (Borg trace v3). https://github.com/google/cluster-data/blob/master/ClusterData2019.md
  - Eight Borg cells over May 2019, about 2.4 TiB compressed, available through BigQuery under CC-BY; a second provider for checking that a lifetime model generalises.
- **Alibaba.** clusterdata, cluster-trace-v2018. https://github.com/alibaba/clusterdata/blob/master/cluster-trace-v2018/trace_2018.md
  - About 4,000 machines over 8 days with online services and batch jobs; smaller, and useful for out-of-distribution tests.

## Open-source code and tools

- **Bergsma et al. (2021).** TraceGen: survival-style lifetime model inside an RNN generator of cloud workloads (title not captured). SOSP 2021. https://tidel.mie.utoronto.ca/pubs/tracegen_camera_20210926.pdf
  - Validated on production VM workloads of two providers; a source of the hazard-based lifetime model and of synthetic traces for stress tests.

## Breakthroughs in related fields that could transfer

- **Ansari, Stella, Turkmen, Zhang et al. (2024).** Chronos: Learning the Language of Time Series. arXiv:2403.07815. https://arxiv.org/abs/2403.07815
  - Pretrained probabilistic forecaster; a quick way to forecast future arrival load, the second prediction that Buchbinder et al. show is useful.
- **Gibbs, Candès (2021).** Adaptive Conformal Inference Under Distribution Shift. arXiv:2106.00170. https://arxiv.org/abs/2106.00170
  - Online calibration of prediction intervals when the data drift; gives a calibrated lifetime upper bound per VM without assuming a stationary tenant mix.

## Published results to beat

- LAVA, arXiv:2412.09840 (abstract and text as returned by search): in Google production, stranded compute down about 3% and stranded memory down about 2%; empty-host availability up 2.3 to 9.2 percentage points; VM disruptions down 4.5%.
- Barbalho et al. (MLSys 2023 slides, citing Hadary et al. 2020): a 1% packing-efficiency gain is worth hundreds of millions of dollars; LAVA reports that Lifetime Alignment is best with perfect predictions and degrades quickly as accuracy falls.
- This repository (README): true lifetimes give −13.8% server-days against best fit; a predictor within a factor of about 3 gives −8.2%.

## Unverified leads

- Li, Tang, Cai, "Dynamic bin packing for on-demand cloud resource allocation" (IEEE TPDS) and Azar, Vainstein, "Tight bounds for clairvoyant dynamic bin packing" (SPAA 2017): not confirmed in session.
- Huawei Cloud VM scheduling dataset and the "VMAgent" simulator: believed to exist, not confirmed in session.
- Random survival forests and DeepHit-style survival models as lifetime predictors: no specific source confirmed; LAVA reports gradient-boosted trees worked best in its tests.
