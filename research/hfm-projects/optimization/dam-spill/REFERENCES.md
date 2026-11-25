# References: Columbia and Snake River Dam Spill Optimization for Juvenile Salmon and Steelhead Migration

Verification: every entry was confirmed by a web search result in this session. Items that could not be confirmed are under "Unverified leads". The search budget ran out before recent (2023-2026) survival studies could be searched, so that area is thin.

## Theory and established methods

- **Zabel, Faulkner, Smith et al. (2008).** Comprehensive passage (COMPASS) model: a model of downstream migration and survival of juvenile salmonids through a hydropower system. Hydrobiologia 609(1), 289-300. https://katalog.lib.cas.cz/KNAV/EdsRecord/asn,32672564
  - The passage model named in the README: daily fish passage per reach and dam, with four submodels (reservoir survival, dam passage, travel time, hydrology), fitted to PIT-tag data.
- **Columbia Basin Research.** COMPASS Comprehensive Passage Model (project page and manuals). https://www.cbr.washington.edu/analysis/compass
  - Entry point to the model; the 2008 manual (https://www.cbr.washington.edu/sites/default/files/manuals/COMPASS_manual_april_2008.pdf) and the 2019 review draft, version 2.0 (https://www.cbr.washington.edu/shiny/SURVDEMO/COMPASS_Manual_2019_Review_Draft_full.pdf) give the equations and fitted coefficients needed to build a reduced passage model.
- **Politano, Castro, Hadjerioua (2017).** Modeling Total Dissolved Gas for Optimal Operation of Multireservoir Systems. Journal of Hydraulic Engineering 143(6), 04017007. https://iro.uiowa.edu/esploro/outputs/journalArticle/Modeling-Total-Dissolved-Gas-for-Optimal/9984627304502771
  - Physically based dissolved-gas model built for use inside spill optimisation of a cascade; calibrated at Rock Island Dam.
- **Witt et al. (year not captured).** Development and Implementation of an Optimization Model for Hydropower and Total Dissolved Gas in the Mid-Columbia River System. Journal of Water Resources Planning and Management 143(10). https://www.ornl.gov/publication/development-and-implementation-optimization-model-hydropower-and-total-dissolved-gas
  - The closest prior work to this task: a reduced-order tailrace gas equation at seven mid-Columbia dams embedded as a prioritised goal in a system optimiser; finds that shifting spill between periods lowers supersaturation while meeting generation.
- **Stewart, Witt, Hadjerioua (2015).** Total Dissolved Gas Prediction and Optimization in RiverWare. Oak Ridge National Laboratory report. https://www.osti.gov/biblio/1222572
  - Technical report behind the paper above; has the linearised gas equations in a form that can be reused as constraints.
- **CADSWES, University of Colorado.** Total Dissolved Gas Research. https://colorado.edu/cadswes/research/total-dissolved-gas-research
  - Explains how the Politano model was linearised so a cascade optimiser converges quickly.
- **US Army Corps of Engineers ERDC.** SYSTDG workbook. https://coastalhydraulicslaboratoryfact.tpub.com/9_SYSTDG/9_SYSTDG0001.htm
  - The Corps' spreadsheet model of gas production from spill at the Columbia and Snake mainstem dams; empirical equations per dam, the standard the gas-cap operations are planned with.
- **Columbia Basin Research (2022).** Methods for modelling total dissolved gas (memo for the dam conditions tool). https://www.cbr.washington.edu/shiny/DAM_CONDITIONS/Methods.TDG.2022.July28.pdf
  - Notes SYSTDG has calibrated gas-generation equations for eight dams and that gas at some dams reaches an asymptote regardless of spill volume; that changes where a gas cap actually binds.

## Recent advances and breakthroughs (2023-2026)

- **Oregon DEQ (2024).** Order approving modification of the total dissolved gas standard for the lower Columbia River. https://oregon.gov/deq/wq/Documents/ColRiver-TDGModOrder2024.pdf
  - Sets the current legal gas limit of 125% from April 1 to June 15 at the lower Columbia dams; this is the hard constraint value for a joint optimisation.
- **ISAB (2023, 2024, 2025).** Reviews of the draft Comparative Survival Study annual reports. https://www.nwcouncil.org/media/filer_public/58/82/588263f5-6f53-4b2a-9a8e-3693b0c18399/ISAB_2025-3_ReviewOfDraftCSS2025AnnualRpt_22Oct.pdf
  - Independent review of the Fish Passage Center's models linking smolt-to-adult returns to water transit time and the powerhouse-passage index (PITPH); the 2024 report added a logistic regression of returns on those two variables, an alternative objective to COMPASS in-river survival.

## Datasets and benchmarks

- **Columbia Basin Research.** Columbia River DART (Data Access in Real Time). https://www.cbr.washington.edu/dart
  - Daily and hourly flow, spill, dissolved gas, temperature and fish passage for all eight projects, with records back to 1878 for some series; the data source for the replay.
- **Columbia Basin Research.** DART River Environment and Hourly Water Quality metadata. https://www.cbr.washington.edu/node/2101 and https://www.cbr.washington.edu/node/2100
  - Field definitions: spill percent is computed from outflow and spill; gas percent is a daily mean of hourly readings between 90% and 200% and is noisy. Needed before fitting a gas model.
- **Columbia Basin Research.** Data inventory per project (example: Bonneville). https://www.cbr.washington.edu/dart/cs/php/rpt/meta_inventory.php?proj=BON
  - Lists which series and years exist at each dam; check all eight before fixing the model's inputs.
- **Columbia Basin Research.** Columbia Basin Performance Measures: River Conditions and Fish Exposure at Hydroelectric Projects. https://www.cbr.washington.edu/perform/
  - Joins observed fish passage timing with gas, spill and temperature exposure; a ready table of what fish actually experienced in past seasons.

## Open-source code and tools

- **Beer / Columbia Basin Research (2022).** Dam conditions tool (spill and gas trade-off visualisation and forecast). https://www.cbr.washington.edu/shiny/DAM_CONDITIONS/description.pdf
  - Working implementation of the per-dam spill-to-gas relationships for the Columbia and Snake; a check for any reduced gas model built here.

## Breakthroughs in related fields that could transfer

- **Bertsekas (2019).** Multiagent Rollout Algorithms and Reinforcement Learning. arXiv:1910.00120. https://arxiv.org/abs/1910.00120
  - Dam-by-dam rollout with the other dams held at a base policy improves on that base policy by construction; a cheap step from dam-by-dam decisions towards joint optimisation.

## Published results to beat

- No comparable published score was found for the HFM metric. Reference points: the 125% gas limit (Oregon DEQ 2024 order) and the Council's 2% to 6% smolt-to-adult return objective, which Snake River wild spring/summer Chinook and steelhead "continue to fall well short of" (ISAB 2019 review, as returned by search). HFM incumbent: 0.7199 (README).

## Unverified leads

- Fish Passage Center Comparative Survival Study annual reports (fpc.org): reviews confirmed, the reports themselves not opened.
- 2019-2021 Flexible Spill Agreement and later spill operation agreements: not found in session.
- US Army Corps Northwestern Division Dataquery (hourly dam operations) and PTAGIS PIT-tag database: search budget ran out before confirmation.
- Washington Ecology and Oregon DEQ adaptive management evaluation of the 115% forebay gas requirement (https://apps.ecology.wa.gov/publications/documents/0910002.pdf): record seen, title not captured.
- NOAA Fisheries annual juvenile survival estimates through the Snake and Columbia dams (Widener, Faulkner et al.): not confirmed in session.
