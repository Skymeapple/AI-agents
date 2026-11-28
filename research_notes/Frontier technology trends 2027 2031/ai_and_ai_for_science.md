# Frontier AI and AI for Science: State as of September 2026 and Outlook for 2027-2031

Method note (for report writer): Research ran 2026-09-24. The network egress proxy blocked full-page fetches of hai.stanford.edu, metr.org, techcrunch.com and cnbc.com, so most figures below come from search-result snippets of the cited pages rather than full-text reads. Primary sources (SEC filings, IEA, METR, Epoch, Stanford HAI, Official Journal via law firms) are marked [primary]. Aggregator or blog sources are marked [secondary] and should be treated with more caution. Labels: MEASURED = observed data; PROJECTION = a model or forecast by the named source; OPINION = commentary or a claim that has not been verified.

## 1. Capability jumps in 2025-2026 and the measured trend rate

### Takeaway
Measured capability growth sped up rather than plateaued. METR's 50% task-completion time horizon doubled about every 4.3 months after 2023, against about 7 months over 2019-2025. The frontier moved from about 2 hours (GPT-5, Aug 2025) to about 12 hours (Claude Opus 4.6, Feb 2026). By May 2026 the top models exceeded METR's reliable measurement ceiling of about 16 hours. Coding and math benchmarks are near saturation: SWE-bench Verified is near 100%, and an AI scored 42/42 at IMO 2026. Harder, contamination-resistant benchmarks such as SWE-bench Pro remain around 50-60%.

### Cited Findings
- **METR doubling time (MEASURED, Jan 29 2026):** METR's Time Horizon 1.1 update estimates the post-2023 doubling time at 130.8 days (about 4.3 months). The longer 2019-2025 trend was about 7 months. — [METR, Time Horizon 1.1](https://metr.org/blog/2026-1-29-time-horizon-1-1/) [primary, via snippet]; [AI 2027 Tracker](https://ai2027-tracker.com/predictions/metr-doubling/) [secondary]
- **Faster sub-trend (analysis, 2026):** From 2023 onward the doubling is about 129 days. From 2024 onward it is about 89 days (about 3 months), which works out to roughly 10x per year. — [Read the OOM / LessWrong, "METR Time Horizons: Now 10x/Year"](https://www.lesswrong.com/posts/EYb2K9acKfyG2bome/metr-time-horizons-now-10x-year) [secondary; independent analysis of METR data]
- **Model-level horizons (MEASURED by METR):** GPT-5 agent about 2 h 17 min (Aug 2025). Claude Opus 4.5 about 4 h 49 min, with a 95% CI of 1 h 49 min to 20 h 25 min (late 2025). Claude Opus 4.6 about 718 min, roughly 12 h (added Feb 2026). GPT-5.2 was added Feb 4 2026 and GPT-5.3-Codex on Feb 20 2026. "Claude Mythos Preview (early)" was added May 8 2026 with METR's note that "measurements above 16 hrs are unreliable with our current task suite." — [METR time-horizons page](https://metr.org/time-horizons/) [primary, via snippet]; [LessWrong on Opus 4.5](https://www.lesswrong.com/posts/q5ejXr4CRuPxkgzJD/claude-opus-4-5-achieves-50-time-horizon-of-around-4-hrs-49); [EA Forum estimate for Opus 4.6](https://forum.effectivealtruism.org/posts/vxEWdn7ni4QEwEdE8/estimating-metr-time-horizons-for-claude-opus-4-6-and-gpt-5-1)
- **METR caveats (Jan 22 2026):** METR published a note on the limits of the time-horizon metric: it is task-suite dependent, it is a 50% success rate, and the confidence intervals are wide. — [METR, Clarifying limitations of time horizon](https://metr.org/notes/2026-01-22-time-horizon-limitations/)
- **SWE-bench (AI Index 2026, published Apr 2026):** "SWE-bench coding scores jumping from 60 to nearly 100 percent in a single year." Frontier models "meet or exceed human baselines on PhD-level science questions, multimodal reasoning, and competition mathematics." — [Stanford HAI, 2026 AI Index](https://hai.stanford.edu/ai-index/2026-ai-index-report); [IEEE Spectrum summary](https://spectrum.ieee.org/state-of-ai-index-2026)
- **SWE-bench Pro (MEASURED, Sep 14 2026):** Scale's SWE-bench Pro has 1,865 tasks across 41 repositories and is designed to resist contamination. Meta Muse Spark 1.1 leads with 61.5% on the public set and 51.5% on the commercial set. GPT-5.4 (xHigh) scores 59.1% and Claude Opus 4.6 scores 47.1% on the public set. — [Morph SWE-bench Pro leaderboard](https://www.morphllm.com/swe-bench-pro) [secondary]; [Scale leaderboard](https://labs.scale.com/leaderboard/swe_bench_pro) [primary]
- **IMO (MEASURED):** At IMO 2025, Google DeepMind (Gemini Deep Think) and OpenAI each scored 35/42, which is gold-medal level. — [Google DeepMind blog](https://deepmind.google/blog/advanced-version-of-gemini-with-deep-think-officially-achieves-gold-medal-standard-at-the-international-mathematical-olympiad/). At IMO 2026 (July 2026), RedNote's "dots-note-3.0" reportedly scored a perfect 42/42 and was reported as the first AI to do so. — [SCMP](https://www.scmp.com/tech/article/3361482/worlds-first-ai-model-earn-perfect-score-maths-olympiad-comes-chinas-rednote) [secondary, press report; official IMO certification status not verified]
- **Cyber capability (Apr 7 2026):** Anthropic's Claude Mythos Preview "autonomously discovered thousands of previously unknown vulnerabilities across every major operating system and web browser." Access is restricted to about 40 organizations under Project Glasswing. — [Cloud Security Alliance research note](https://labs.cloudsecurityalliance.org/research/ai-vuln-discovery-containment-claude-mythos-v1-0-csa-styled/) [secondary, based on Anthropic system card]; [Tanium](https://www.tanium.com/blog/claude-mythos-security-risks)
- **Industry share (AI Index 2026):** More than 90% of notable frontier models released in 2025 came from private companies. — [Stanford HAI takeaways](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)

### Inferences
- Take the frontier at about 12-16+ hours in early to mid 2026 and a doubling time of 4-7 months. Extrapolating gives roughly a 1-week horizon (about 40 working hours) sometime between mid-2027 and 2028. At the 7-month rate, a 1-month horizon (about 167 h) arrives around 2028-2029. METR's own suite cannot currently measure above about 16 h, so the next leading indicator is METR releasing a longer-task suite.
- Benchmarks are saturating: SWE-bench Verified and IMO. The informative signals are moving to SWE-bench Pro, METR horizons, OSWorld-type computer-use benchmarks, and real-world uplift studies.

### Gaps
- No 2026 OSWorld or computer-use benchmark top score was found in the search results.
- The METR number for Claude Mythos or the newest GPT-5.x models was not retrievable because metr.org was blocked. Only "above 16 h unreliable" was confirmed.
- Official IMO certification of the RedNote 42/42 claim was not verified.

## 2. Adoption and economics: lab revenue, agents, inference cost, capex, power

### Takeaway
Revenue at frontier labs grew several-fold within 2026. By July 2026 Anthropic's run-rate was reported at more than $65B, up from about $9B at end-2025. Nvidia data center revenue reached $89B in one quarter. The four largest US hyperscalers guided about $725-785B of 2026 capex, up about 77% year over year. Enterprise agent deployment at scale is still a minority: 23% of firms are scaling an agent somewhere, and no more than 10% in any single function. Inference price for a fixed capability falls a median of about 50x per year (about 200x per year since 2024). Data-centre electricity rose 17% in 2025 and is projected to reach about 945 TWh by 2030.

### Cited Findings
**Lab revenue (MEASURED as reported, 2026)**
- Anthropic's annualized run-rate was more than $65B at end-July 2026, up from $47B in May 2026 and about $9B at end-2025. Bloomberg (Sep 18 2026) reported it on pace to top $100B by year-end. — [TechCrunch, Aug 17 2026](https://techcrunch.com/2026/08/17/anthropics-annualized-revenue-surges-to-65b/); [Axios, Aug 17 2026](https://www.axios.com/2026/08/17/anthropic-revenue-run-rate-ipo-openai)
- OpenAI's run-rate was reported at about $40B by Aug 2026, up from about $25B in Feb 2026 and about $13B of revenue in 2025. — [ValueAdd VC](https://valueaddvc.com/blog/openai-revenue-2026-20b-arr-4b-month-path-to-profitability) [secondary; lower-quality source]. Combined ARR above $115B. — [Crypto Briefing](https://cryptobriefing.com/anthropic-openai-revenue-growth-2026/) [secondary]
- Epoch AI (earlier 2026) projected that Anthropic could pass OpenAI in annualized revenue by mid-2026. Per the reports above, that happened. — [Epoch AI data insight](https://epoch.ai/data-insights/anthropic-openai-revenue)

**Chips (MEASURED, SEC filing)**
- Nvidia Q2 FY2027 (quarter ended Jul 26 2026, reported Aug 26 2026): revenue $96.2B (+106% YoY). Data Center revenue $89.0B (+117% YoY, +18% QoQ), driven by the Blackwell Ultra ramp. Hyperscale revenue more than doubled YoY. AI Clouds/Industrial/Enterprise revenue rose 138% YoY. — [Nvidia 8-K Q2 FY27 press release, SEC](https://www.sec.gov/Archives/edgar/data/0001045810/000104581026000073/q2fy27pr.htm) [primary]

**Hyperscaler capex (Q2 2026 earnings, late Jul 2026)**
- 2026 full-year guidance: Amazon about $220B, raised from about $200B partly because of higher memory costs. Alphabet $195-205B. Meta $130-145B, with the floor raised. Microsoft about $175B, restated after an accounting change. Combined $725-785B, about +77% YoY. — [UncoverAlpha Q2 analysis](https://www.uncoveralpha.com/p/amazon-google-microsoft-meta-q2-earnings) [secondary]; [TMT Finance, "2026 hyperscaler capex tops US$700bn"](https://www.tmtfinance.com/intel/2026-hyperscaler-capex-tops-us700bn-analysis); [CNBC, Jul 28 2026](https://www.cnbc.com/2026/07/28/hyperscalers-face-higher-capex-scrutiny-after-alphabet-report-panned.html)
- Free cash flow strain in Q2 2026: Alphabet reported negative FCF of about -$5.9B, reportedly its first since the 2004 IPO. Meta's FCF was about $784M. Microsoft's FCF was $19.6B with capex up about 70% YoY. — [UncoverAlpha](https://www.uncoveralpha.com/p/amazon-google-microsoft-meta-q2-earnings) [secondary; verify against 10-Qs]
- The IEA says big-tech capex exceeded $400B in 2025 and is "expected to jump by another 75% in 2026." — [IEA news, 2026](https://www.iea.org/news/data-centre-electricity-use-surged-in-2025-even-with-tightening-bottlenecks-driving-a-scramble-for-solutions) [primary]

**Enterprise adoption**
- McKinsey State of AI 2026 (Aug 2026): 44% of organizations say AI is scaling across the enterprise (38% a year earlier). 23% are scaling an agentic AI system somewhere. No more than 10% are scaling agents in any single function. 40% of companies with $1B+ revenue are scaling agents (up from 27%). 47% are scaling chatbots. — [McKinsey, The state of AI in 2026 (PDF)](https://www.mckinsey.com/~/media/mckinsey/business%20functions/quantumblack/our%20insights/the%20state%20of%20ai/the-state-of-ai-in-2026-on-the-road-to-roi.pdf) [primary]; [Forbes, Mar 22 2026](https://www.forbes.com/sites/josipamajic/2026/03/22/10-of-enterprise-functions-use-ai-agents-mckinsey-finds/)
- AI Index 2026: organizational AI adoption reached 88% in 2025. Generative AI reached 53% population-level adoption within 3 years, faster than the PC or the internet. — [Stanford HAI 2026 AI Index](https://hai.stanford.edu/ai-index/2026-ai-index-report); [Stark Insider summary](https://www.starkinsider.com/2026/04/stanford-2026-ai-index-report.html)

**Inference cost (MEASURED trend)**
- Epoch AI: the price to reach a fixed capability milestone falls 9x-900x per year depending on the task, with a median of about 50x per year. Counting only models released since Jan 2024, the median is about 200x per year. GPT-4-level performance on PhD-level science questions got 40x cheaper per year. — [Epoch AI, LLM inference price trends](https://epoch.ai/data-insights/llm-inference-price-trends) [primary]
- Epoch "The plunging price of thought": since 2023, price has fallen about 47% per quarter, or about 13x per year, by that measure. — [Epoch AI](https://epoch.ai/publications/the-plunging-price-of-thought)
- The AI Index 2025 had measured a 280x drop in about 18 months for GPT-3.5-level quality. — [VoxBooster compilation](https://voxbooster.com/blog/ai-inference-cost-statistics-2026/) [secondary]

**Power (IEA)**
- MEASURED: global data-centre electricity demand grew 17% in 2025. Consumption by AI-focused data centres grew about 50% in 2025. — [IEA news, 2026](https://www.iea.org/news/data-centre-electricity-use-surged-in-2025-even-with-tightening-bottlenecks-driving-a-scramble-for-solutions) [primary]
- PROJECTION: data-centre use more than doubles to about 945 TWh by 2030, slightly more than Japan's total consumption today. The US accounts for nearly half of its own electricity-demand growth to 2030 from data centres. — [IEA, Energy and AI exec summary](https://www.iea.org/reports/energy-and-ai/executive-summary); [IEA, Key Questions on Energy and AI](https://www.iea.org/reports/key-questions-on-energy-and-ai/executive-summary) [primary]
- AI data-centre power capacity was reported at 29.6 GW (secondary compilation citing AI Index data; date and scope unclear). — [VoxBooster](https://voxbooster.com/blog/ai-inference-cost-statistics-2026/) [secondary, treat cautiously]

### Inferences
- Revenue is now growing faster than capex. Frontier-lab ARR rose about 5-7x in 2026 while hyperscaler capex rose about 1.8x. Even so, Alphabet and Meta's thin FCF shows capex is running ahead of internally generated cash. The ratio of AI revenue to capex is a key leading indicator of whether 2027 capex rises again or plateaus.
- Price per token at fixed capability is collapsing (about 50-200x per year), but total inference spend is rising because frontier usage (agents, long horizons) consumes far more tokens. This is a Jevons dynamic.

### Gaps
- OpenAI's 2026 revenue comes only from secondary sources. No primary confirmation was found.
- Individual hyperscaler Q2 capex quarterly numbers came from secondary analyses. Primary 10-Q figures were not fetched.
- The 2026 AI Index figures for corporate investment, training compute and incidents were not retrievable because the page was blocked.

## 3. AI for science: validated results versus hype

### Takeaway
Validated, operational results so far are concentrated in two areas. First, weather: ML models are operational at ECMWF since 2025 and at NOAA since January 2026. Second, formal mathematics: Lean-verified proofs of open Erdős problems in 2026. AI-designed drugs have reached Phase III (Insilico's rentosertib, Sep 2026), but no AI-discovered drug has yet shown Phase III efficacy. Isomorphic's first human trials had slipped to end-2026. Autonomous labs and materials discovery are still mostly at the demonstration stage.

### Cited Findings
- **Weather (VALIDATED, operational):** ECMWF began running AIFS operationally in Feb 2025, with the probabilistic ensemble following in Jul 2025. NOAA followed in Jan 2026. AIFS upgrades alongside IFS Cycle 50r1 in 2026 add new output variables such as snow. — [CACM, "AI Weather Forecasting Goes Operational"](https://cacm.acm.org/news/ai-weather-forecasting-goes-operational/); [ECMWF, "Farewell to the external AI models" (2026)](https://www.ecmwf.int/en/about/media-centre/aifs-blog/2026/farewell-external-ai-models) [primary]
- **Formal math (VALIDATED by Lean proof checker):** In May 2026 Google DeepMind's AlphaProof Nexus autonomously solved 9 of 353 open Erdős problems it attempted and proved 44 of 492 open OEIS conjectures, at "a few hundred dollars" per problem. Experts validated that the Lean statements faithfully captured the originals. Aristotle (Harmonic) and AxiomProver have also formalized solutions to open Erdős problems. — [WinBuzzer, May 26 2026](https://winbuzzer.com/2026/05/26/google-deepmind-says-alphaproof-nexus-is-still-not-agi-xcxwbn/); [arXiv 2605.22763, "Advancing Mathematics Research with AI-Driven Formal Proof Search"](https://arxiv.org/html/2605.22763v1) [primary]
- **Drug discovery, Phase III (Sep 2026):** Insilico started the GENESIS-IPF-3 Phase III trial of rentosertib (NCT07687459), a TNIK inhibitor for IPF. It plans 320 patients at 47 centers in China over 52 weeks. In the Phase IIa study, the 60 mg arm improved mean FVC by +98.4 mL at 12 weeks. Primary readout is unlikely before late 2027. — [Insilico press release](https://insilico.com/news/xmjsn4l091-insilico-initiates-phase-iii-clinical-tr) [primary]; [News-Medical, Sep 10 2026](https://www.news-medical.net/news/20260910/Generative-AI-driven-drug-Rentosertib-enters-Phase-III-trial-for-idiopathic-pulmonary-fibrosis.aspx)
- **Isomorphic Labs:** As of Jan 2026 the CEO expected first-in-human trials by end-2026, a slip from 2025. The IsoDDE technical report (Feb 10 2026) reports 50% accuracy on hard protein-ligand cases (under 20% similarity to training data), against 23.3% for AlphaFold 3. The company has 17 programs. IsoDDE is proprietary and not publicly available. — [IntuitionLabs](https://intuitionlabs.ai/articles/isomorphic-labs-alphafold-ai-drug-discovery-trials) [secondary]; [Dubach analysis](https://philippdubach.com/posts/ai-can-now-design-drugs-in-seconds-we-still-cant-tell-you-if-they-work./); [Wikipedia](https://en.wikipedia.org/wiki/Isomorphic_Labs)
- **Autonomous labs and materials:** Community papers in 2026 describe a "two-year roadmap" toward trustworthy autonomous science, which implies it is not yet mature. — [arXiv 2607.12113](https://arxiv.org/pdf/2607.12113); [arXiv 2512.01080, "Building Trustworthy AI for Materials Discovery"](https://arxiv.org/pdf/2512.01080)
- **AI in science (AI Index 2026):** The AI Index 2026 includes dedicated science and medicine chapters. — [Stanford HAI](https://hai.stanford.edu/ai-index/2026-ai-index-report) (specific numbers not retrieved)

### Inferences
- Validation hierarchy as of Sep 2026:
  - (1) Operational: weather.
  - (2) Machine-verified: formal math proofs of real open problems, though these are modest ones (9/353 Erdős).
  - (3) Clinical-stage but unproven: AI-designed drugs.
  - (4) Mostly demonstrations: autonomous labs and materials.
- A forecastable milestone for 2027-2028 is the first Phase III readout of an AI-discovered drug. Rentosertib is the earliest candidate (late 2027 at the earliest), so an approval before 2029-2030 is unlikely.

### Gaps
- No 2026 data was found on the total count of AI-discovered molecules in clinical trials.
- No validated 2026 autonomous-lab or materials result was found (for example, synthesized novel materials with confirmed properties).
- Details on AlphaFold 4 or other 2026 structure-model releases beyond IsoDDE were not found.
- MIT Technology Review 10 Breakthroughs 2026, WEF Top 10 Emerging Technologies 2026 and McKinsey Tech Trends 2026 were not retrieved.

## 4. Constraints and risks: power, chips, data, regulation, safety incidents

### Takeaway
The binding constraints are power and grid interconnection, plus memory and chip supply: Amazon raised capex because of memory costs. Regulation is loosening or being delayed. The EU deferred high-risk AI Act obligations to Dec 2027 and Aug 2028. US federal preemption of state laws remains unresolved as of Sep 2026. The most significant 2026 safety event was the cyber-offense capability of Claude Mythos Preview (Apr 2026), which prompted emergency meetings between the US Treasury, the Federal Reserve and bank CEOs.

### Cited Findings
- **EU AI Act (law, 2026):** Council and Parliament provisionally agreed on May 7 2026. The Digital Omnibus on AI, Regulation (EU) 2026/1744, was published in the Official Journal on Jul 24 2026 and entered into force on Jul 27 2026. High-risk obligations for Annex III systems are deferred to Dec 2 2027. Annex I (AI embedded in regulated products) is deferred to Aug 2 2028. — [Gibson Dunn](https://www.gibsondunn.com/eu-ai-act-omnibus-agreement-postponed-high-risk-deadlines-and-other-key-changes/); [Pinsent Masons](https://www.pinsentmasons.com/out-law/news/rules-high-risk-ai-delayed-under-eu-omnibus-deal); [Praxikon](https://www.praxikon.com/en/posts/digital-omnibus-high-risk-postponement-december-2027)
- **US policy:** EO 14365 (Dec 11 2025) created a DOJ AI Litigation Task Force, operating from Jan 10 2026, and conditioned $42B of BEAD broadband funds on states dropping "onerous" AI rules. As of mid-2026 the Task Force had not sued over major state laws, and no federal preemption bill has passed. In July 2025 the Senate stripped a 10-year state moratorium by a 99-1 vote. California SB 53 is in force, and Colorado is rewriting its AI Act. — [Paul Hastings](https://www.paulhastings.com/insights/client-alerts/president-trump-signs-executive-order-challenging-state-ai-laws); [Ropes & Gray, Mar 2026](https://www.ropesgray.com/en/insights/alerts/2026/03/the-white-house-legislative-recommendations-national-policy-framework-for-artificial-intelligence-an); [TechPolicy.Press](https://www.techpolicy.press/where-state-ai-legislation-stands-half-way-into-2026/); [CASRAI](https://casrai.org/news/federal-ai-moratorium-state-preemption-fight-2026)
- **Safety and security incident (Apr 2026):** The Mythos system card describes complex exploitation and "autonomous sandbox escape capabilities." Treasury and the Fed convened bank CEOs the day after the announcement, and the IMF flagged financial-stability risk. Analysts estimate equivalent capability reaches the broader market, possibly including adversaries, within 6-24 months (OPINION). — [CSA research note](https://labs.cloudsecurityalliance.org/research/ai-vuln-discovery-containment-claude-mythos-v1-0-csa-styled/); [Data Protection Report, May 2026](https://www.dataprotectionreport.com/2026/05/when-ai-becomes-the-cyber-attacker-mythos-and-what-comes-next/)
- **Governance lag:** The AI Index 2026 finds most companies do not publish detailed safety evaluations, bias audits or transparency reports. — [Stanford HAI](https://hai.stanford.edu/ai-index/2026-ai-index-report)
- **Power as a bottleneck:** The IEA describes "tightening bottlenecks driving a scramble for solutions" (2026). — [IEA news](https://www.iea.org/news/data-centre-electricity-use-surged-in-2025-even-with-tightening-bottlenecks-driving-a-scramble-for-solutions). Epoch projects that the largest single training runs need 4-16 GW by 2030, with training power growing 2.2-2.9x per year. — [Epoch AI, power demands of frontier training](https://epoch.ai/blog/power-demands-of-frontier-ai-training) [primary; PROJECTION]
- **Memory and chips:** Amazon attributed its capex raise (about $200B to about $220B) partly to higher memory costs. — [UncoverAlpha](https://www.uncoveralpha.com/p/amazon-google-microsoft-meta-q2-earnings) [secondary]

### Inferences
- The regulatory trajectory is toward delay and lighter touch in both the EU and the US. The main hard dates in 2027-2031 are EU Annex III (Dec 2 2027) and Annex I (Aug 2 2028).
- Cyber-offense capability is the most concrete near-term risk vector. Diffusion of Mythos-class capability to open-weight models within 6-24 months is a watch item.

### Gaps
- No data was found on training-data exhaustion in 2026 or on an updated Epoch estimate.
- There are no AI Incident Database or AI Index 2026 incident counts.
- China export-control status for 2026 was not researched because of the tool budget.

## 5. Forecasts for 2027-2031 (Metaculus, Epoch, METR, AI Index) plus forecastable milestones

### Takeaway
Forecasters have pulled timelines forward sharply. Metaculus puts "weakly general AI" around 2027 and full AGI around the early 2030s. Epoch projects that training runs of about 2e29 FLOP are feasible by 2030, with more than 200 models above 1e26 FLOP. The IEA projects about 945 TWh of data-centre demand by 2030. The milestone list below uses these trend rates. Probabilities are my own evidence-based estimates (labelled INFERENCE) and are not taken from any source.

### Cited Findings
- **Metaculus:** The "weakly general AI" question showed a community estimate of about Sep 17 2027 (snapshot date unclear). As of Feb 2026, the AGI question was about 25% by 2029 and 50% by 2033. Other summaries say "weak AGI before end-2026, strong AGI start of 2031." These snapshots conflict. — [Metaculus Q3479](https://www.metaculus.com/questions/3479/date-weakly-general-ai-is-publicly-known/); [Metaculus Q17124](https://www.metaculus.com/questions/17124/conditional-date-of-artificial-general-intelligence/); [AGI Timelines Dashboard](https://agi.goodheartlabs.com/) [secondary aggregator]; [Vera Calloway](https://www.veracalloway.com/blog/ai-culture/agi-timeline/) [secondary]. The report writer should check live values.
- **Epoch AI:** Frontier training compute is growing about 4-5x per year. About 2e29 FLOP runs are feasible by 2030. More than 200 models above 1e26 FLOP by 2030, up from about 10 by 2026. — [Epoch, Can AI scaling continue through 2030?](https://epoch.ai/blog/can-ai-scaling-continue-through-2030); [Epoch, model counts](https://epoch.ai/publications/model-counts-compute-thresholds); [Epoch, What will AI look like in 2030?](https://epoch.ai/publications/what-will-ai-look-like-in-2030) [PROJECTION]
- **METR trend outlook:** Some analysts expect the 3.5-4 month doubling to slow back toward 7 months by end-2026 (OPINION). — [Read the OOM](https://readtheoom.substack.com/p/metr-time-horizons-now-10xyear). Manifold runs a market on the 2026 doubling time. — [Manifold](https://manifold.markets/Bayesian/what-will-be-the-metr-time-horizon)
- **Prediction markets:** Polymarket listed an "AI wins IMO gold in 2026" market, which has now resolved in practice given the 2026 results. — [Polymarket](https://polymarket.com/event/ai-wins-imo-gold-medal-in-2026)
- **IEA:** About 945 TWh of data-centre electricity by 2030 (PROJECTION). — [IEA](https://www.iea.org/reports/energy-and-ai/executive-summary)
- **Epoch:** The largest training runs need 4-16 GW by 2030 (PROJECTION). — [Epoch](https://epoch.ai/blog/power-demands-of-frontier-ai-training)

### Inferences: forecastable milestones with probabilities (INFERENCE, author estimates)
These are grounded in the trend rates above. Treat them as calibrated judgement, not sourced data.
1. **METR 50% horizon of at least 40 h (1 work-week) on a published METR measurement by end-2027: about 70%.** Basis: about 12-16 h in H1 2026, 4-7 month doubling. The main risk is the measurement ceiling or suite, not capability.
2. **METR 50% horizon of at least 1 work-month (about 167 h) by end-2029: about 55%.** At a 7-month doubling this lands around 2029. At a 4-month doubling it lands in 2028.
3. **SWE-bench Pro public set above 80% by end-2027: about 65%.** The top score was 61.5% in Sep 2026, and the Verified variant went from 60% to about 100% in 1 year.
4. **At least one frontier lab with annualized revenue of $100B or more by end-2026: about 75%.** Bloomberg reported Anthropic "on pace" in Sep 2026. **A lab above $250B by end-2028: about 40%.**
5. **Top-4 hyperscaler combined capex above $1T in calendar 2027: about 55%.** It was $725-785B in 2026, and growth would need to be about 30% or more. The FCF strain at Alphabet and Meta is the main downside.
6. **IEA-reported global data-centre electricity of at least 900 TWh in any year by 2030: about 55%.** The IEA base case is about 945 TWh in 2030, and grid bottlenecks are a downside.
7. **First positive Phase III readout for an AI-discovered or AI-designed drug by end-2028: about 40%.** Rentosertib's readout comes late 2027 at the earliest. The IPF Phase III base rate is low. **First regulatory approval (FDA, EMA or NMPA) of such a drug by end-2030: about 35%.**
8. **Isomorphic Labs doses its first human patient by end-2026: about 50%.** It has already slipped once.
9. **An AI system autonomously produces a Lean-verified proof of a problem widely considered a major open conjecture (beyond Erdős-list-level) by end-2030: about 30%.** An AI-assisted result published in Annals, Inventiones or an equivalent top journal by end-2028: about 50%.
10. **EU Annex III high-risk obligations actually apply on Dec 2 2027 without further delay: about 65%.**
11. **A US federal statute preempting state AI laws enacted by end-2027: about 25%.** There was a 99-1 Senate defeat and no bill had passed as of Sep 2026.
12. **A publicly confirmed major cyber incident attributed to AI-autonomous exploitation, causing more than $1B in damage, by end-2028: about 35%.**

### Leading indicators to watch
- New METR measurements, and whether METR releases a longer-horizon task suite (currently capped at about 16 h).
- Scores on SWE-bench Pro, OSWorld and Terminal-Bench.
- Monthly ARR disclosures from Anthropic and OpenAI, and Anthropic IPO filings (Axios says "pre-IPO"), which would provide audited revenue.
- Nvidia Data Center revenue each quarter ($89B in Q2 FY27). Hyperscaler Q3 2026 capex guidance (late Oct 2026) and 2027 guidance (Jan-Feb 2027).
- IEA mid-2027 data-centre update. US interconnection queues.
- Epoch price-per-capability trend: does it stay at about 50-200x per year?
- Rentosertib Phase III enrollment. Isomorphic's first-in-human announcement.
- AlphaProof Nexus, Aristotle and AxiomProver solve counts on the Erdős and OEIS lists.
- EU implementing standards ahead of Dec 2027. DOJ AI Litigation Task Force filings.
- Open-weight models matching Mythos-class cyber capability.

### Gaps
- Live Metaculus values could not be fetched. The snapshots found conflict (weak AGI around end-2026 vs Sep 2027).
- The AI Index 2026 forward-looking content, the MIT TR 10 Breakthroughs 2026 list, the WEF Top 10 Emerging Tech 2026 list, and the McKinsey Tech Trends 2026 report were not retrieved because of blocked pages and tool budget.
- METR's own forecasts for 2027-2031 were not retrieved.
