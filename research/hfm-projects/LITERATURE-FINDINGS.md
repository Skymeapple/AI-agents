# What the literature pass changed

Every project folder now has a `REFERENCES.md` with theory, recent work, datasets, code and published numbers. This page lists the findings that change what to try next, and says where the collection is still thin.

## How the references were checked

- Six collectors worked by field. Each entry had to be confirmed in that session, either by a web search result or by fetching the paper's own arXiv or Zenodo record.
- A shared limit on web searches ran out part-way. After that, entries were confirmed only by fetching records directly, which favours arXiv papers over journal articles and classic textbooks.
- Anything that could not be confirmed is listed under "Unverified leads" in each file and should not be cited until checked.

## Findings that change the plan

| Project | Finding | Source |
| --- | --- | --- |
| 24-hour weather | Stored forecasts of other model families are public on the task's grid, including for 2022, HFM's held-out year. Blending is a CPU experiment (done here for 2020; 2022 queued). The ArchesWeather paper's own table shows GraphCast and Pangu ahead of it on wind and pressure at 24 h. | https://weatherbench2.readthedocs.io/en/latest/data-guide.html, https://arxiv.org/abs/2405.14527 |
| Lean theorem proving | A harness around a frozen general model reports 95.9% on miniF2F-test with sub-goal decomposition and repair, against HFM's 0.566. | https://arxiv.org/abs/2507.15225 |
| Agentic RL recipes | HFM's best idea (procedural terminal tasks with tests) is already published, and its transfer to Terminal-Bench 2.0 was small for 7B models. | https://arxiv.org/abs/2601.16443 |
| Looped pretraining | At matched training compute, looping loses to dense models unless the loop itself is improved; what to loop depends on whether parameters or compute are the limit; more loops is not monotonic. | https://arxiv.org/abs/2604.21106, https://arxiv.org/abs/2610.40325 |
| Pretraining (all) | Final loss can be estimated from about a third of a run, and loss on one dataset predicts loss on another, so short runs can rank ideas. | https://arxiv.org/abs/2410.11840, https://arxiv.org/abs/2411.12925 |
| Cloud VM placement | Re-predicting remaining lifetime as a VM survives (Google's LAVA) addresses exactly the failure seen here with a one-shot predictor. | https://arxiv.org/abs/2412.09840 |
| Bus holding | Holding policies can be learned offline from logged vehicle positions; closed-form optimal linear holding rules also exist. No bulk download of past HSL positions was found. | https://arxiv.org/abs/2610.14632, https://arxiv.org/abs/2610.18253 |
| Arterial signals, ride-hail | Simulators partly exist already: SUMO models built from Toronto counts, and open ride-hail dispatch environments. | https://arxiv.org/abs/2508.10733, https://arxiv.org/abs/2609.30628 |
| Bike and rail dispatch | A policy can be trained to imitate hindsight-optimal solutions, which both simulators here can already produce. | https://arxiv.org/abs/2302.03963 |
| Lung histology | Pre-extracted slide features for TCGA and CPTAC exist for a second encoder, which would allow CPU training. Phikon-v2 was pretrained on unlabelled NLST slides, the held-out cohort. | https://github.com/mahmoodlab/UNI, https://arxiv.org/abs/2409.09173 |
| Lung cell annotation | The full-atlas scANVI embedding is a 2.4 GB public file, so the full 35-study version can be run on CPU. | https://zenodo.org/records/7599104 |
| Crystal generation | Ion exchange of known compounds is reported to beat generative models at producing novel stable materials; screening with universal potentials helps every method. | https://arxiv.org/abs/2501.02144 |
| Regulatory DNA | A benchmark finds DNA language models give no compelling gain over supervised baselines on regulatory DNA, which fits the multi-task CNN result here. | https://arxiv.org/abs/2412.05430 |
| Rayleigh–Bénard forecast | Deterministic emulators can be fine-tuned into ensemble models with a CRPS loss, with reported 20–54% lower rollout CRPS. | https://arxiv.org/abs/2603.01949 |
| Cold battery | A third cell dataset with the same temperature plan exists (Samsung 30T). | https://data.mendeley.com/datasets/9xyvy2njj3/2 |
| Order-book projects | Deep order-book models drop significantly on unseen data; execution learning needs a simulator, not historical replay. FI-2010 is a free ten-level book for five Nordic stocks. | https://arxiv.org/abs/2308.01915, https://arxiv.org/abs/2511.15262, https://arxiv.org/abs/1705.03233 |
| Crypto projects | Binance's archive is now licensed CC BY-NC-SA 4.0: research use only, share-alike. | https://github.com/binance/binance-public-data/blob/master/TERMS_AND_CONDITIONS.md |
| Portfolio, A-share | Qlib's official dataset is disabled; the community source has no stated licence. Qlib documents one-minute A-share data, which may reopen the A-share project. | https://github.com/microsoft/qlib |

## Corrections made to project notes

- **FabNER:** an earlier note said encoder models reach the high 80s. The best published figure found is 85.4, so headroom is about 5 points, not 8.
- **Wind farms:** the public records cover 2016–2024, not only the one year used.
- **Steel:** the suggestion that one cheap load reversal would close the gap was already withdrawn after the experiment.

## Where the collection is thin

A second search pass on 25 November added confirmed entries to the steel, wave-energy, wind-farm, rail and scheduling files. The most useful additions: a 2025 paper doing this project's wave task with ERA5 as the reference (https://www.mdpi.com/2077-1312/13/6/1194), a field campaign reporting the same 9-degree wind-direction offset found here (https://wes.copernicus.org/preprints/wes-2021-32/wes-2021-32-manuscript-version3.pdf), and the rolling-horizon and dynasearch methods that the rail and scheduling projects should test next.

Before that pass, these files rested on few confirmed entries:
- physics: `steel-cyclic-from-tension`, `steel-history-transfer`, `wind-farm-wakes`, and `wave-energy-resource` (no confirmed entries, leads only);
- optimization: `rail-dispatch`, and `weighted-tardiness` in infrastructure;
- classic journal papers in the finance, unit-commitment and ride-hail files are mostly under "Unverified leads".
