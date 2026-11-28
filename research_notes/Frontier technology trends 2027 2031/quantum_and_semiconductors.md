# Quantum Computing and Advanced Semiconductors/Photonics: State as of September 2026 and Outlook 2027-2031

Method note (for the report writer): compiled 2026-09-24 from web searches. Several primary domains (wsts.org, thequantuminsider.com) blocked fetches from this environment, so some figures come from search-result summaries of those pages and could not be read in full. Figures marked **[MEASURED]** are published experimental or financial results. Figures marked **[ROADMAP/CLAIM]** are vendor targets or forecasts. Figures marked **[ESTIMATE]** are theoretical resource estimates or analyst projections. The probabilities under "Inferences" are this researcher's judgments from the cited evidence. They are not sourced numbers.

## Q1. Quantum error correction: where the hardware stands (2024-2026)

### Takeaway
From 2024 to 2026 the field moved from a single below-threshold surface-code memory (Google Willow, Dec 2024) to dozens of logical qubits at about 2:1 physical-to-logical overhead on trapped-ion and neutral-atom hardware (Quantinuum Helios 48-50 logical, QuEra/Harvard up to 96 logical). Real-time qLDPC decoding was also shown (IBM, <480 ns). However, most of these "logical qubits" are error-detecting or small-distance codes. None is yet a deep fault-tolerant computation of about 100 logical qubits at 1e-6 or lower logical error rates.

### Cited Findings
- **Google Willow (Dec 2024, Nature) [MEASURED]:** a 105-physical-qubit superconducting chip. Surface-code distance 3 to 5 to 7 showed the logical error rate falling by about 2.14x per distance step (Lambda about 2.14). Google describes this as the first demonstration of below-threshold exponential suppression. — [Google Quantum AI QEC milestone](https://quantumai.google/qecmilestone); [APS Physics](https://physics.aps.org/articles/v17/176)
- **Google "Quantum Echoes" (Oct 22, 2025, Nature) [MEASURED/CLAIM]:** an out-of-time-order-correlator measurement on Willow, reported as about 13,000x faster than the best classical estimate. Google calls it "verifiable quantum advantage" because the result can be reproduced on another quantum device. — [Google Quantum AI (Wikipedia summary)](https://en.wikipedia.org/wiki/Google_Quantum_AI); [thequantumbull](https://thequantumbull.com/google-quantum-ai-achieves-verifiable-quantum-advantage/)
- **Google adds neutral atoms (Mar 24, 2026) [ROADMAP]:** Google added neutral-atom hardware as a second modality alongside superconducting qubits. — [Tech-Insider](https://tech-insider.org/google-quantum-computing-neutral-atom-willow-2026/) (secondary source; not verified against a Google primary)
- **QuEra/Harvard/MIT (2025, four Nature papers) [MEASURED]:** up to 96 logical qubits on 448 atoms, continuous operation of 3,000 atoms, below-threshold error correction with error falling as size grows, and magic-state distillation at the logical level. — [PostQuantum QuEra profile](https://postquantum.com/quantum-computing-companies/quera/); [TQI Jul 2025 magic-state distillation](https://thequantuminsider.com/2025/07/15/quera-harvard-and-mit-researchers-demonstrate-logical-level-magic-state-distillation-on-a-neutral-atom-quantum-computer/)
- **QuEra (Apr 2026) [MEASURED, per secondary]:** reported a 2:1 physical-to-logical ratio using qLDPC codes. — [PostQuantum](https://postquantum.com/quantum-computing-companies/quera/); [arXiv 2608.05010 neutral-atom review](https://arxiv.org/pdf/2608.05010)
- **Quantinuum Helios (launched Nov 2025) [MEASURED]:** 98 physical barium-ion qubits; 48 logical qubits, later filings say 50, at about 2:1 overhead, versus 12 on H2. Quantinuum calls this the first 2:1 overhead in a commercial system. — [Quantum Computing Report](https://quantumcomputingreport.com/quantinuum-launches-helios-quantum-computer-with-industry-leading-fidelity-and-singapore-partnership/); [NextBigFuture May 2026](https://www.nextbigfuture.com/2026/05/quantinuum-helios-with-98-physical-qubits-and-50-logical-qubits.html). Caveat: a 2:1 overhead implies high-rate error-detecting or low-distance codes, which are not comparable to Google's distance-7 surface code.
- **Microsoft + Quantinuum (Jun 2026) [MEASURED]:** 12 "highly reliable" logical qubits, plus an end-to-end chemistry workflow (logical qubits + AI + HPC) reaching chemical accuracy for a catalytic intermediate. — [TQI Jun 13 2026](https://thequantuminsider.com/2026/06/13/microsoft-and-quantinuum-report-on-major-gains-in-quantum-error-correction/) (search summary; page fetch blocked)
- **Record logical-error suppression (Mar 2026) [MEASURED]:** reported in the headline "Researchers Demonstrate Record Suppression of Quantum Logical Errors". The group and the numbers could not be verified because the fetch was blocked. — [TQI Mar 4 2026](https://thequantuminsider.com/2026/03/04/researchers-demonstrate-record-suppression-of-quantum-logical-errors/)
- **Atom Computing (with Microsoft) [MEASURED]:** 24 logical qubits entangled from 112 physical atoms. Bernstein-Vazirani was run on 28 logical qubits. In 2026 it ran multi-round continuous toric-code QEC, with the logical error rate falling as size increased. — [TQI search summary](https://thequantuminsider.com/2026/03/04/researchers-demonstrate-record-suppression-of-quantum-logical-errors/); [arXiv 2608.05010](https://arxiv.org/pdf/2608.05010)
- **IBM Loon and Nighthawk (Nov 12, 2025) [MEASURED + ROADMAP]:** Nighthawk has 120 qubits in a square lattice. Loon demonstrates the components needed for qLDPC. Real-time qLDPC decoding in under 480 ns, using Relay-BP on an AMD FPGA, was demonstrated about a year ahead of schedule and is about 10x faster than prior approaches. — [IBM newsroom](https://newsroom.ibm.com/2025-11-12-ibm-delivers-new-quantum-processors,-software,-and-algorithm-breakthroughs-on-path-to-advantage-and-fault-tolerance); [Next Platform](https://www.nextplatform.com/2025/11/12/ibm-lets-fly-nighthawk-and-loon-qpus-on-the-way-to-quantum-advantage/)
- **Microsoft Majorana (topological) [CONTESTED CLAIM]:** Majorana 1 (Feb 2025) drew scientific debate. Majorana 2 (Microsoft Build, Jun 2026) reports qubit lifetimes over 20 s, more than 1,000x longer than earlier devices. It switches the superconductor from aluminium to lead and more than doubles the topological gap. On Sep 22, 2026 Microsoft opened a 15,000 sq ft Maryland center where DARPA will test the system independently under US2QC/QBI. QBI aims to decide by 2033 whether each approach can reach utility scale. Microsoft is targeting a scalable machine by 2029. — [TQI Jun 2 2026](https://thequantuminsider.com/2026/06/02/microsoft-reports-advances-in-majorana-2-following-debate-over-last-years-topological-claims/); [TQI Sep 22 2026](https://thequantuminsider.com/2026/09/22/microsoft-gives-darpa-access-to-majorana-system-opens-maryland-quantum-research-center/); [Forbes/Moor Jul 16 2026](https://www.forbes.com/sites/moorinsights/2026/07/16/microsoft-doubles-down-on-topological-qubits-with-majorana-2-chip/)
- **PsiQuantum (photonic) [FUNDING/ROADMAP]:** raised a $1B Series E (Sep 2025) at a $7B valuation, led by BlackRock with Temasek, Baillie Gifford and NVentures. Australia and Queensland committed AUD 940M (about USD 620M) to the Brisbane site (Apr 2024). Illinois committed $500M to the IQMP park in Chicago, including $200M for a cryoplant. Chicago is now expected online before Brisbane, and Brisbane is reported as "running very late". No public logical-qubit demonstration was found. — [PsiQuantum](https://www.psiquantum.com/news-import/psiquantum-1b-fundraise); [Startup Daily](https://www.startupdaily.net/topic/global-tech/psiquantums-brisbane-build-is-already-running-very-late/)

### Inferences
- The leading metric has shifted from "number of physical qubits" to logical qubits × logical error rate × logical-gate depth. Vendors quote different codes (distance-7 surface code vs. [[k:2k]]-style detection codes), so headline logical-qubit counts are not comparable across companies. The report should warn about this.
- Neutral atoms and trapped ions lead on logical-qubit count. Superconducting systems lead on cycle speed and on clean below-threshold scaling evidence. Google's move into neutral atoms (Mar 2026) signals that the modality race is still open.

### Gaps
- Could not retrieve the exact logical error rates per cycle for the 2026 records (TQI pages blocked). Google's post-Willow 2026 QEC results, such as a distance-9+ experiment or a logical-gate milestone, were not found.
- No verified information on whether IBM Kookaburra (the 2026 roadmap item) has been delivered as of Sep 2026.
- No independent confirmation that Majorana 2 qubits are topologically protected. DARPA testing has only just begun.

## Q2. Vendor roadmaps to fault tolerance, and quantum advantage claims

### Takeaway
Multiple vendors now converge on 2028-2030 for their first fault-tolerant machines with about 100-200 logical qubits: IBM Starling in 2029 (200 logical qubits, 1e8 gates), Quantinuum Apollo in 2029, and Microsoft "by 2029". DARPA's QBI uses 2033 as its utility-scale test horizon. IBM's "quantum advantage by end-2026" is the key near-term test, and it is being adjudicated through an open community tracker.

### Cited Findings
- **IBM roadmap (Jun 10, 2025) [ROADMAP]:** Loon (2025), then Kookaburra (2026), the first qLDPC memory plus logical processing unit module. Next is Cockatoo (2027), which entangles modules. Starling (2029, Poughkeepsie) targets 200 logical qubits and 1e8 gates, with magic-state injection across modules demonstrated in 2028. Blue Jay (2033+) targets about 2,000 logical qubits. IBM says qLDPC codes cut physical overhead by up to about 90% versus surface codes. — [IBM newsroom](https://newsroom.ibm.com/2025-06-10-IBM-Sets-the-Course-to-Build-Worlds-First-Large-Scale,-Fault-Tolerant-Quantum-Computer-at-New-IBM-Quantum-Data-Center); [IBM blog](https://www.ibm.com/quantum/blog/large-scale-ftqc); [IEEE Spectrum](https://spectrum.ieee.org/ibm-quantum-error-correction-starling)
- **IBM advantage target [CLAIM]:** "quantum advantage by end of 2026", with claims judged through an open advantage tracker (IBM, Algorithmiq, Flatiron Institute, BlueQubit). — [IBM newsroom Nov 2025](https://newsroom.ibm.com/2025-11-12-ibm-delivers-new-quantum-processors,-software,-and-algorithm-breakthroughs-on-path-to-advantage-and-fault-tolerance)
- **Quantinuum roadmap [ROADMAP]:** Helios (2025) has 48-50 logical qubits. Sol (2027) targets about 100 logical qubits approaching 99.999% logical fidelity. Apollo (2029) targets hundreds of logical qubits and up to "ten nines" logical fidelity. — [Quantinuum S-1 (SEC)](https://www.sec.gov/Archives/edgar/data/0002110105/000162828026032836/quantinuum-sx1.htm); [quantumway summary](https://www.quantumway.org/post/quantinuum-qnt-quantum-computing-leader-helios-sol-apollo-roadmap-1-68b-ipo-800-logical)
- **QuEra roadmap [ROADMAP]:** has published a roadmap to about 100 logical qubits. — [QuEra](https://www.quera.com/press-releases/quera-computing-releases-a-groundbreaking-roadmap-for-advanced-error-corrected-quantum-computers-pioneering-the-next-frontier-in-quantum-innovation-0); [Quantum Zeitgeist](https://quantumzeitgeist.com/quera-computing-roadmap-100-logical-qubits/)
- **Verified advantage status [MEASURED/CLAIM]:** Google Quantum Echoes (Oct 2025, 13,000x) is the strongest peer-reviewed "verifiable" claim. Earlier random-circuit-sampling claims remain non-useful benchmarks. — [Google Quantum AI](https://en.wikipedia.org/wiki/Google_Quantum_AI)
- **DARPA QBI** evaluates whether approaches can yield utility-scale, commercially useful machines by 2033. — [TQI Sep 22 2026](https://thequantuminsider.com/2026/09/22/microsoft-gives-darpa-access-to-majorana-system-opens-maryland-quantum-research-center/)

### Inferences (forecastable milestones; probabilities are the researcher's judgment)
- A community-accepted, practically useful quantum advantage result (not a sampling benchmark), for example on IBM's tracker, by end-2027: **about 50%**. By end-2029: **about 75%**.
- IBM delivers Starling (≥100 logical qubits running ≥1e7 logical ops) by end-2029: **about 40%**; by end-2031: **about 65%**. IBM has hit most dated roadmap items since 2020, which supports this, but a scale-up of this size has no precedent.
- Some vendor demonstrates ≥100 logical qubits with ≤1e-4 logical error per operation by end-2028: **about 55%**. Quantinuum Sol and QuEra are the most likely.
- Independent DARPA validation of a topologically protected Majorana qubit by end-2028: **about 25%**.
- Leading indicators to watch:
  - Google Lambda and distance-9/11 results
  - IBM Kookaburra/Cockatoo delivery dates
  - Quantinuum Sol shipping in 2027
  - which companies advance to QBI Stage C
  - entries on the advantage tracker that survive classical-simulation rebuttals
  - PsiQuantum's first logical-qubit data from Chicago

### Gaps
- No primary Google roadmap update for 2026 was found (the milestone 3 long-lived logical qubit target date is unknown).
- Atom Computing and Pasqal 2027-2030 dated targets were not collected.

## Q3. Cryptographic threat: RSA-2048 resource estimates and PQC deadlines

### Takeaway
Theoretical estimates of the hardware needed to break RSA-2048 fell about 20x in 2025 and roughly another 10x in early 2026: from about 20M physical qubits (2019) to under 1M (Gidney, May 2025), then to about 100k (qLDPC "Pinnacle", Feb 2026) and about 10k neutral atoms with long runtimes (Mar 2026). Expert surveys shifted accordingly: GRI's 2025 report, published Mar 2026, puts a CRQC within 10 years at 28-49%. US federal policy treats 2030 as the deprecation date and 2035 as the disallow date. NSS systems follow the stricter CNSA 2.0 schedule.

### Cited Findings
- **Gidney (May 2025 preprint) [ESTIMATE]:** RSA-2048 with under 1M noisy physical qubits (surface code) in less than a week. This is a 20x cut from the 2019 estimate of 20M. — [Quantum Computing Report](https://quantumcomputingreport.com/shor-qldpc-codes-and-the-compression-of-rsa-2048-resource-estimates-part-i/); [PostQuantum](https://postquantum.com/quantum-research/quantum-breakthrough-rsa-2048/)
- **"Pinnacle Architecture" (Feb 2026) [ESTIMATE]:** about 100,000 physical qubits using qLDPC codes, "magic engines" and modular processing units. It requires non-local connectivity. — [TQI Feb 13 2026](https://thequantuminsider.com/2026/02/13/new-architecture-could-cut-quantum-hardware-needed-to-break-rsa-2048-by-tenfold-study-finds/); [Scott Aaronson blog](https://scottaaronson.blog/?p=9564)
- **Cain et al. (Mar 2026) [ESTIMATE]:** cryptographically relevant Shor runs with as few as about 10,000 reconfigurable neutral atoms, at much longer runtimes. Both 2026 estimates are unbuilt-architecture preprints and are not directly comparable to Gidney's baseline. — [Quantum Computing Report](https://quantumcomputingreport.com/shor-qldpc-codes-and-the-compression-of-rsa-2048-resource-estimates-part-i/)
- Background on the frequently quoted "4,099 logical qubits" figure: [PostQuantum](https://postquantum.com/post-quantum/4099-qubits-rsa/). Elliptic-curve (ECDLP) targets may fall earlier than RSA: [arXiv 2508.14011](https://arxiv.org/pdf/2508.14011)
- **GRI Quantum Threat Timeline Report 2025 (dated Mar 9, 2026; 26 experts) [SURVEY]:** CRQC within 10 years is "quite possible" at 28-49%, and within 15 years "likely" at 51-70%. The optimistic 10-year average rose from 34% (2024) to 49% (2025), the largest jump in the survey's seven-year history. — [GRI](https://globalriskinstitute.org/publication/quantum-threat-timeline-report-2025b/); [PostQuantum review](https://postquantum.com/security-pqc/quantum-threat-timeline-report-2025/)
- **NIST [POLICY]:** FIPS 203/204/205 (ML-KEM, ML-DSA, SLH-DSA) were finalized Aug 2024. NIST IR 8547 (ipd Nov 2024) deprecates 112-bit-security RSA/ECC (e.g. RSA-2048, P-256) after 2030 and disallows all RSA/ECC/DH after 2035. — [NIST IR 8547 ipd](https://nvlpubs.nist.gov/nistpubs/ir/2024/NIST.IR.8547.ipd.pdf); [Entrust](https://www.entrust.com/blog/2024/12/nists-urgent-call-deprecating-traditional-crypto-by-2030)
- **US federal [POLICY, per secondary source]:** EO 14412 treats 2030 as the compliance deadline for high-value and high-impact systems. OMB M-26-15 aligns agencies to IR 8547 with full migration by 2035. — [Quantum Security Defence](https://quantumsecuritydefence.com/insights/nist-ir-8547-transition-timeline/) (not verified against the primary EO/OMB text)
- **NSA CNSA 2.0 (Sep 2022) [POLICY]:** CNSA 2.0 is preferred from 2025-2026 across most NSS categories and required by 2030-2033. The full NSS transition is due by 2035. New NSS acquisitions must be CNSA 2.0-compliant from Jan 1, 2027. — [PostQuantum on IR 8547/CNSA](https://postquantum.com/security-pqc/nist-ir-8547-ipd/)

### Inferences
- A CRQC (RSA-2048 broken in ≤1 month) by end-2031: **about 5-10%**. This matches the lower half of GRI's 10-year range, adjusted for the gap between about 100 logical qubits in 2029 roadmaps and the about 1,000+ high-fidelity logical qubits even the best 2026 estimates need.
- A further ≥2x cut in published RSA-2048 physical-qubit estimates by end-2028: **about 60%**, given the 2025-2026 trend.
- Leading indicators:
  - new Gidney-style estimates
  - logical-gate error rates reaching 1e-6
  - demonstrations of factoring small numbers with error-corrected Shor
  - GRI's 2026 survey shift
  - adoption of hybrid ML-KEM in TLS
  - agency PQC inventory reports under OMB M-26-15

### Gaps
- NIST IR 8547 final version status (whether it moved beyond the initial public draft) was not confirmed.
- The exact CNSA 2.0 per-category dates were not re-verified against the NSA primary PDF.

## Q4. Quantum funding and public markets

### Takeaway
2025-2026 was quantum computing's capital-markets breakout. McKinsey counts $12.6B invested in 2025 (6.3x 2024) and more than $1B of vendor revenue. A 2026 IPO/SPAC wave (Quantinuum, Infleqtion, Xanadu, Horizon Quantum, IQM) created a much larger listed peer group. IonQ guides to about $260-270M of 2026 revenue.

### Cited Findings
- **McKinsey Quantum Technology Monitor 2026 [MEASURED + ESTIMATE]:** quantum computing companies earned more than $1B revenue in 2025, rising to as much as $4.4B by 2028. Investment in 2025 was $12.6B (6.3x y/y), 44% from capital markets and 34% from private funds. Economic value by 2035 is estimated at $1.3-2.7T. More than 300 companies work with vendors, and 33% of analyzed companies spend over $10M/yr. — [McKinsey](https://www.mckinsey.com/capabilities/mckinsey-technology/our-insights/mckinsey-quantum-technology-monitor-2026-a-commercial-tipping-point)
- **Quantinuum listing (Jun 2026) [MEASURED]:** SEC S-1, S-1/A and 424B4 filings exist, which indicates a traditional IPO. It raised about $1.68B, per secondary sources. The earlier private round valued the company at about $10B, and analysts expected about $20B at IPO. — [SEC 424B4](https://www.sec.gov/Archives/edgar/data/0002110105/000162828026041003/quantinuum-424b4.htm); [quantumway](https://www.quantumway.org/post/quantinuum-qnt-quantum-computing-leader-helios-sol-apollo-roadmap-1-68b-ipo-800-logical). Sources conflict: one aggregator describes a "SPAC merger", which contradicts the SEC 424B4 prospectus ([entangledfuture](https://entangledfuture.com/guides/quantum-computing-spac-ipo-guide-2026/)).
- **2026 listings [MEASURED]:** Quantinuum (QNT), Xanadu (XNDU), Horizon Quantum (HQ) and Infleqtion (INFQ) listed between Feb and Jun 2026. Infleqtion's SPAC with Churchill Capital X raised $550M+ and began trading on Nasdaq in Feb 2026. IQM also IPO'd. — [Motley Fool Jul 27 2026](https://www.fool.com/investing/2026/07/27/whats-next-for-these-4-quantum-computing-ipo-stock/); [DCD Q2 2026 quantum earnings](https://www.datacenterdynamics.com/en/news/quantum-earnings-q2-2026-d-wave-ionq-and-rigetti-publish-financial-results-alongside-newly-ipo-ed-iqm-quantum-computers/)
- **IonQ [MEASURED/GUIDANCE]:** about $130M revenue in 2025. 2026 guidance was raised to $260-270M after Q1 2026 (from $225-245M). — [IonQ Q1 2026 release](https://investors.ionq.com/news/news-details/2026/IonQ-Announces-First-Quarter-2026-Financial-Results/default.aspx); [heygotrade](https://www.heygotrade.com/en/blog/quantum-computing-stocks-2026/)
- **PsiQuantum:** $1B at a $7B valuation (Sep 2025). Its latest disclosed round was a Jan 2026 venture round. — [PsiQuantum](https://www.psiquantum.com/news-import/psiquantum-1b-fundraise)

### Inferences
- Pure-play quantum revenue is still under about $2B/yr in 2026, compared with public valuations in the tens of billions. Valuations therefore depend on the 2027-2029 roadmap milestones, which makes those milestones important for the sector's share prices.
- Quantum vendor revenue will likely reach at least $4B in 2028, the top of McKinsey's range: **about 40%**.

### Gaps
- Exact Quantinuum IPO price and market cap, and full 2026 year-to-date venture totals, were not retrieved.

## Q5. Leading-edge logic: 2nm/A16/18A, High-NA EUV, backside power

### Takeaway
2nm-class nodes are in high-volume production at all three leading foundries: TSMC N2 since Q4 2025, Intel 18A in Panther Lake from Jan 2026, and Samsung SF2 in 2026. Backside power arrives at TSMC with A16 in Q4 2026 (Intel already has PowerVia on 18A), and at Samsung with SF2Z in 2027. High-NA EUV is installed and accepted at Intel for 14A (risk production 2027), and is in qualification at Samsung and imec.

### Cited Findings
- **TSMC Q2 2026 (reported Jul 16, 2026) [MEASURED]:** revenue $40.2B, gross margin 67.7%, net income NT$706.6B (+77.4% y/y), EPS NT$27.25. FY2026 USD revenue growth guidance was raised to above 40%. 2026 capex is $60-64B, mostly for N2/N3 and CoWoS. TSMC announced a further $100B for Arizona. N2 entered HVM in Q4 2025 and is ramping at two sites. — [TSMC IR Q2 2026](https://investor.tsmc.com/english/quarterly-results/2026/q2); [Troy Technical](https://troy-technical.com/2026/07/19/tsmc-achieves-record-q2-revenue-of-40-2b-with-67-7-gross-margin-cowos-fully-booked-through-2026-and-maps-14-reticle-packaging/); [BigGo](https://finance.biggo.com/news/5784aaf1-fcbc-4f76-8856-491b9e7175f6)
- **TSMC A16 [ROADMAP + VALIDATION]:** Super Power Rail backside power was validated (VLSI, Jun 2026). Versus N2P it gives +8-10% speed at iso-power, or -15-20% power at iso-speed, and +8-10% density. Volume production is planned for Q4 2026. It is expected to be the first node at scale in Arizona's third fab. — [ETNews Aug 19 2026](https://en.etnews.com/20260819200004); [SemiWiki](https://semiwiki.com/forum/threads/tsmc-a16-backside-power-at-2026-jun-vlsi.25444/)
- **Intel 18A [MEASURED, per reports]:** Panther Lake launched at CES Jan 2026 as the first 18A product (RibbonFET + PowerVia backside power). 18A yields rose about 7% per month (Nov 2025), and Panther Lake reached about 80% yield per a TechPowerUp report. 14A has High-NA, with risk production in 2027 and HVM expected in 2028. — [Intel newsroom](https://newsroom.intel.com/client-computing/intel-unveils-panther-lake-architecture-first-ai-pc-platform-built-on-18a); [TrendForce Nov 2025](https://www.trendforce.com/news/2025/11/21/news-intels-two-front-push-18a-yields-reportedly-rise-7-monthly-14a-enters-definition-stage/); [TechPowerUp](https://www.techpowerup.com/352724/intel-panther-lake-hits-80-yield-on-18a-node); [WinBuzzer Mar 2026](https://winbuzzer.com/2026/03/17/intels-18a-14a-roadmap-2026-foundry-panther-lake-xcxwbn/)
- **High-NA EUV [MEASURED]:** ASML shipped its first EXE:5200 in Jul 2025. Intel installed the EXE:5200B and completed acceptance testing for 14A. The tool runs about 175 wafers/hr, about 60% more than the EXE:5000. Samsung received its first EXE:5200B in late 2025 and a second is due H1 2026. imec is targeting Q4 2026 qualification of its EXE:5200 for sub-2nm work. — [TrendForce Jul 2025](https://www.trendforce.com/news/2025/07/17/news-asml-confirms-first-high-na-euv-exe5200-shipment-reportedly-prepping-for-intels-14a-in-2027/); [Tom's Hardware](https://www.tomshardware.com/tech-industry/semiconductors/intel-installs-industrys-first-commercial-high-na-euv-lithography-tool-asml-twinscan-exe-5200b-sets-the-stage-for-14a); [TrendForce Mar 2026](https://www.trendforce.com/news/2026/03/19/news-imec-secures-asmls-most-advanced-exe5200-high-na-euv-for-sub-2nm-4q26-qualification-target/)
- **Samsung [MEASURED/ROADMAP]:** SF2 is in mass production and SF2P follows in 2026. SF2P+ is headed to the Taylor, Texas fab, where Tesla AI5 will be built on SF2T under a $16.5B deal and Nvidia is reportedly evaluating capacity. SF2Z (backside power) is planned for 2027. — [Tom's Hardware](https://www.tomshardware.com/tech-industry/samsungs-fab-roadmap-examined); [cyberraiden comparison](https://cyberraiden.wordpress.com/2026/03/11/comparing-the-leading-2nm-nodes-in-2026-tsmc-n2-intel-18a-and-samsung-sf2-density-performance-yields-and-ecosystem/)

### Inferences
- TSMC A14 (about 2028) and Intel 14A (HVM about 2028) are the next milestones. Whether TSMC adopts High-NA at A14 or later is a key watch item: TSMC has so far said it can defer High-NA.
- Intel 14A announces at least one major external (non-US-government) foundry customer by end-2027: **about 55%**.
- TSMC A16 in HVM with at least one major AI/HPC customer by mid-2027: **about 85%**.

### Gaps
- TSMC's official N2 revenue share for Q2 2026 and its A14 High-NA decision were not retrieved.
- No reliable independent yield data for Samsung SF2 was found.

## Q6. Advanced packaging, HBM4, co-packaged optics and silicon photonics, chiplets

### Takeaway
Packaging and memory remain the AI bottleneck. TSMC CoWoS capacity has grown about 80% a year (about 35k wafers/month end-2024, about 75k end-2025, about 125-130k target end-2026) and is still fully booked through 2026. HBM4 entered mass production at SK hynix and Samsung in Feb 2026. Co-packaged optics moved into volume in mid-2026, with Nvidia Spectrum-X Photonics shipping from Jul 2026 and Broadcom Bailly 51.2T. Optical-engine yield is now the constraint.

### Cited Findings
- **CoWoS [MEASURED/TARGET]:** about 35k wafers/month (end-2024), about 75k (end-2025), and a target of about 125-130k (end-2026). It is fully booked through 2026. TSMC's roadmap goes to 14-reticle packages. — [Troy Technical, TSMC Q2 2026](https://troy-technical.com/2026/07/19/tsmc-achieves-record-q2-revenue-of-40-2b-with-67-7-gross-margin-cowos-fully-booked-through-2026-and-maps-14-reticle-packaging/); [TechTimes](https://www.techtimes.com/articles/320142/20260711/tsmc-q2-earnings-july-16-three-cowos-signals-that-test-ais-spending-ceiling.htm) (the capacity figures come from secondary analyst estimates, not TSMC disclosures)
- **HBM4 [MEASURED]:** mass production began in Feb 2026 at Samsung (Pyeongtaek) and SK hynix (M16 Icheon, M15X Cheongju). SK hynix uses a TSMC 12nm logic base die and reportedly supplies about 2/3 of Nvidia's HBM4. Samsung aims for about 250k wafers/month of HBM capacity by end-2026, up from 170k (about +47%). 12-high stacks are ramping and 16-high is being pushed. — [Digitimes Dec 2025](https://www.digitimes.com/news/a20251226PD223/samsung-sk-hynix-production-hbm4-2026.html); [TrendForce Jan 28 2026](https://www.trendforce.com/news/2026/01/28/news-sk-hynix-reportedly-to-supply-about-two-thirds-of-nvidia-hbm4-samsung-targets-early-delivery/); [TrendForce Jan 9 2026](https://www.trendforce.com/news/2026/01/09/news-nvidia-demand-fuels-hbm4-race-12-layer-ramps-16-layer-push-by-sk-hynix-samsung-and-micron/)
- **SK hynix US listing (2026):** SEC F-1 and 424B4 filings exist, indicating a US ADR offering in 2026. — [SEC 424B4](https://www.sec.gov/Archives/edgar/data/0002120882/000119312526299963/d32785d424b4.htm) (details not reviewed)
- **Co-packaged optics [MEASURED]:** TrendForce (Jul 27, 2026) says Nvidia and Broadcom have begun the volume ramp of CPO switches. Nvidia's Spectrum-X CPO switch, built on TSMC COUPE with up to 400 Tb/s and 200G micro-ring modulators, began shipping to select partners in Jul 2026, with capacity expanding in H2 2026. Broadcom's Bailly 51.2T CPO switch is in limited shipments. Optical engines, silicon photonics chips and packaging capacity are the bottlenecks. — [TrendForce press center](https://www.trendforce.com/presscenter/news/20260727-13151.html); [NVIDIA newsroom](https://nvidianews.nvidia.com/news/nvidia-spectrum-x-co-packaged-optics-networking-switches-ai-factories); [SemiAnalysis](https://newsletter.semianalysis.com/p/co-packaged-optics-cpo-book-scaling)

### Inferences
- At about 80% a year, CoWoS-class capacity would reach about 200k+ wafers/month by end-2027 if the trend holds. Watch TSMC's CoPoS (panel) and SoIC ramps, and OSAT (ASE/Amkor) CoWoS-like capacity.
- CPO in scale-up (GPU-to-GPU) links, not just scale-out switches, in a shipping flagship AI system by end-2028: **about 50%**.
- HBM4E in volume by end-2027: **about 70%**. Custom base dies from logic foundries become standard.

### Gaps
- No verified 2026 figures on chiplet standards (UCIe 3.0 adoption) or on market-size data for silicon photonics and CPO.

## Q7. Export controls and China's domestic progress

### Takeaway
US policy loosened in part in Jan 2026: H200/MI325X-class exports to China moved to case-by-case licensing with a 25% US revenue cut and a 50% volume cap. The equipment controls that keep SMIC on DUV multi-patterning remain in place. SMIC reportedly produces 5nm-class (N+3) chips at 30-40% yield. Huawei's 2026 Ascend 950 parts are, by Huawei's own roadmap, below the 910C in total processing performance (TPP).

### Cited Findings
- **BIS final rule (Jan 15, 2026) [POLICY]:** license review for H200, MI325X and similar chips (TPP under 21,000 and DRAM bandwidth under 6,500 GB/s) moved from presumption of denial to case-by-case. Conditions are a 25% payment to the US, a 50% volume cap, third-party testing in the US, KYC, and proof that US supply is not reduced. This followed Trump's Dec 8, 2025 announcement. — [Federal Register 2026-00789](https://www.federalregister.gov/documents/2026/01/15/2026-00789/revision-to-license-review-policy-for-advanced-computing-commodities); [BIS](https://www.bis.gov/press-release/department-commerce-revises-license-review-policy-semiconductors-exported-china); [ArentFox Schiff](https://www.afslaw.com/perspectives/national-security-counsel/bis-relaxes-export-license-review-h200-chips-china-and-macau)
- **SMIC [ESTIMATE, secondary]:** N+3 (5nm-class) is in volume on DUV multi-patterning, with yield estimated at about 30-40% versus TSMC's 80%+. — [FinancialContent/TokenRing Jan 2026](https://markets.financialcontent.com/stocks/article/tokenring-2026-1-28-silicon-sovereignty-how-huawei-and-smic-are-neutralizing-us-export-controls-in-2026) (low-quality aggregator; treat with caution). CFR argues Huawei cannot catch Nvidia and that SMIC is effectively capped around 7nm. — [CFR](https://www.cfr.org/articles/chinas-ai-chip-deficit-why-huawei-cant-catch-nvidia-and-us-export-controls-should-remain)
- **Huawei Ascend 950PR/950DT (2026) [ROADMAP]:** lower TPP than the Ascend 910C, per Huawei's own roadmap. — [CFR](https://www.cfr.org/articles/chinas-ai-chip-deficit-why-huawei-cant-catch-nvidia-and-us-export-controls-should-remain)
- **Taiwan** added Huawei and SMIC to its export control entity list, so Taiwanese firms now need permits to sell to them. — [Seeking Alpha](https://seekingalpha.com/pr/20136791-taiwan-adds-china-s-huawei-and-smic-to-export-control-list)
- Continued export-control breaches show China still struggles to produce high-end chips. — [MERICS](https://merics.org/en/comment/export-control-breach-reveals-chinas-continuing-struggle-produce-high-end-semiconductors)

### Inferences
- China demonstrates a domestic EUV prototype exposing production wafers by end-2028: **about 25%**. SMIC in HVM at a 3nm-class node with yield above 50% by end-2028: **about 20%**.
- US chip-export policy is likely to keep swinging, including legislation such as the proposed AI OVERWATCH Act ([Introl](https://introl.com/blog/bis-h200-china-export-policy-ai-overwatch-act-2026)). A leading indicator is actual H200 license approvals and volumes.

### Gaps
- No primary data on actual H200 shipments to China in 2026, SMIC's 2026 capex, or China's HBM (CXMT) progress.

## Q8. Semiconductor revenue 2025/2026, fabs, and 2027-2031 projections

### Takeaway
The industry had a record 2025 ($791.7B, +25.6%, per SIA). WSTS's Spring 2026 forecast (Jun 2026) makes the largest upward revision on record: 2026 at about $1.51T (+90%), driven by memory (about +250% to over $800B) on AI and HBM pricing, and 2027 at about $1.9T (+27%). The market passes $1T about four years earlier than the pre-2024 consensus of about 2030.

### Cited Findings
- **SIA (Feb 2026) [MEASURED]:** 2025 global sales were $791.7B (+25.6% versus $630.5B in 2024). Logic was $301.9B (+39.9%) and memory $223.1B (+34.8%). SIA projected about $1T for 2026 at that time. Q1 2026 sales rose 25% versus Q4 2025. — [SIA](https://www.semiconductors.org/global-annual-semiconductor-sales-increase-25-6-to-791-7-billion-in-2025/); [SIA Q1 2026](https://www.semiconductors.org/global-semiconductor-sales-increase-25-from-q4-2025-to-q1-2026/)
- **WSTS Spring 2026 (Jun 2026) [FORECAST]:** 2026 at $1.51T (+90%), with memory up about 250% to over $800B and logic up 37%. By region, Americas +112%, APAC +87%, Europe +58%, Japan +28%. 2027 is forecast at about $1.9T (+27%). The Autumn 2025 forecast had put 2026 at about $975B ("approaches $1T"). — [WSTS release](https://www.wsts.org/76/103/Global-Semiconductor-Market-Surges-Beyond-15T-2026); [Digitimes Jun 5 2026](https://www.digitimes.com/news/a20260605VL208/semiconductor-industry-wsts-growth-forecast-2026.html); [WSTS Nov 2025 PDF](https://www.wsts.org/esraCMS/extension/media/f/WST/7310/WSTS_FC-Release-2025_11.pdf)
- **US fabs [MEASURED/ROADMAP]:** TSMC Arizona's total commitment is now about $165B, plus the $100B added in Jul 2026. A16 is slated for the third Arizona fab. Samsung's Taylor fab is ramping SF2P+ for Tesla. — [Tech-Insider](https://tech-insider.org/tsmc-arizona-165-billion-expansion-gigafab-2026/); [Tom's Hardware](https://www.tomshardware.com/tech-industry/samsungs-fab-roadmap-examined)

### Inferences
- The 2026 figure is inflated by memory prices: memory at over $800B is about 3.6x 2025. A memory-price correction in 2027-2028 is the main downside risk to WSTS's $1.9T for 2027. There is roughly a **35%** chance that 2027 actual revenue falls below 2026 in nominal terms, given memory-cycle history. That chance is higher if AI capex decelerates.
- 2030 revenue above $1.5T (consensus is now well ahead of the old $1T-by-2030 target): **about 70%**, conditional on AI capex continuing.
- Leading indicators:
  - hyperscaler capex guidance
  - DRAM and HBM contract prices (TrendForce)
  - monthly SIA 3MMA sales
  - TSMC monthly revenue
  - CoWoS utilization
  - WSTS Autumn 2026 revision (due about Nov/Dec 2026)

### Gaps
- SEMI fab equipment spending and World Fab Forecast figures for 2026-2027, imec's 2031+ roadmap specifics (CFET, 2D channels), and CHIPS Act disbursement status for 2026 were not retrieved within the tool budget.
- The WSTS page itself could not be fetched, so figures come from Digitimes and the search snippet.
