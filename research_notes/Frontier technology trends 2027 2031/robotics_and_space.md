# Robotics / Physical AI, Autonomous Vehicles, Drones, and Space: State as of September 2026 and Implications for 2027-2031

Research date: 2026-09-24. Budget-limited pass (~20 tool calls). Several sources are secondary aggregators; they are marked [secondary] and should be treated with more caution than primary company or agency sources. Wikipedia and smartanalyticsglobal.com could not be fetched (egress blocked), so figures that came from them are search-snippet level only. Probabilities in the Inferences sections are my own judgments from the cited base rates. They are not taken from any source.

## Humanoid robots, robot foundation models, and industrial robots

### Takeaway
Humanoids moved from demos to low-volume commercial shipments. Around 13,000-16,000 units shipped in 2025, more than 80% of them Chinese (AgiBot, Unitree, UBTech), and about 19,100 shipped in 1H 2026 (+272% YoY). US players (Tesla, Figure) remain at pilot or internal-use scale, with limited productive work verified. Traditional industrial robots are a far larger market (542,000 installed in 2024) growing only at low single digits. Capital is flowing in very fast: a record $27.6B for robotics/physical AI in 2025 and $5-40B valuations. Measured productivity evidence has not kept pace with that funding.

### Cited Findings
**Shipments (measured, but counts vary by source and methodology)**
- 2025: AgiBot shipped 5,168 humanoids and ranked #1 globally. Unitree was #2 with 4,200 (32% share) and UBTech #3 with about 1,000. Chinese firms dominated global shipments. — [SCMP, Jan 2026](https://www.scmp.com/tech/tech-trends/article/3339346/chinese-firms-outpace-us-rivals-2025-humanoid-robot-shipments-agibot-takes-lead); [Bloomberg, 2026-01-08](https://www.bloomberg.com/news/articles/2026-01-08/chinese-firms-dominated-global-humanoid-robot-shipments-in-2025)
- Conflict: Unitree itself said its 2025 humanoid shipments to end customers "exceeded 5,500 units", which would make it #1. — [Unitree clarification](https://shop.unitree.com/blogs/news/clarification-regarding-unitrees-2025-sales-data); [Gasgoo](https://autonews.gasgoo.com/articles/news/annual-champion-changes-hands-unitree-announces-over-5500-humanoid-robot-shipments-2014711162140991489)
- Global humanoid installations in 2025 were roughly 16,000 units, with China above 80%. — [Gasgoo/search summary](https://autonews.gasgoo.com/articles/news/xiaozhi-weekly-news-unitree-humanoid-robot-shipments-exceed-5500-units-2015824041443430400) [secondary; the underlying estimator is unclear]
- 1H 2026: global humanoid shipments were 19,100 units (+272% YoY), with AgiBot overtaking Unitree for #1. — [Smart Analytics Global (SAG)](https://smartanalyticsglobal.com/global-humanoid-robot-shipments-2026-agibot-unitree/) (title only; page fetch blocked)
- Tesla Optimus: an estimated 1,000-1,200 units at Fremont and Giga Texas as of mid-2026, with zero external sales. On the Q4 2025 call, Musk said the units are "primarily for learning and data collection rather than performing productive tasks". Tasks cited include sorting 4680 cells, kitting, and moving parts. — [IIoT World](https://www.iiot-world.com/smart-manufacturing/tesla-optimus-manufacturing-2026/) [secondary; the unit estimate is attributed to "Axis Intelligence Research", not Tesla]; Tesla filings: [Q1 2026 10-Q](https://www.sec.gov/Archives/edgar/data/0001318605/000162828026026673/tsla-20260331.htm)
- Figure AI: Figure 03 is built at the BotQ facility (stated capacity 12,000 units/yr). At BMW Spartanburg, Figure logged 1,250+ hours and placed 90,000+ parts at over 99% accuracy. — [search summary of IIoT World / trackers](https://www.iiot-world.com/smart-manufacturing/tesla-optimus-manufacturing-2026/) [company-reported, secondary]
- Agility, 1X (NEO) and UBTech Walker S unit prices: I found no reliable 2026 figures in this pass (see Gaps).

**Forecasts (projections, not measurements)**
- Goldman Sachs raised its 2026 humanoid shipment estimate from 51k to 75k units and its 2030 estimate from 256k to 890k. It raised 2035 to about 6.5M units (from about 1.4M), a market of roughly $138B (from $38B). — [Yahoo Finance on GS Physical AI report](https://finance.yahoo.com/technology/ai/articles/goldman-sachs-just-supercharged-humanoid-135551371.html)
- Morgan Stanley projects a humanoid market above $5T by 2050, with about 1B units in use (90% industrial/commercial) and about 13M in service by 2035. — [Morgan Stanley](https://www.morganstanley.com/insights/articles/humanoid-robot-market-5-trillion-by-2050)
- Note: annualized, the 1H 2026 run rate (about 38k) is roughly half of Goldman's raised 2026 estimate of 75k.

**Robot foundation models**
- Google DeepMind Gemini Robotics (a VLA built on Gemini 2.0) launched 2025-03-12. — [arXiv 2503.20020](https://arxiv.org/abs/2503.20020). Gemini Robotics ER 2 was announced 2026-07-30 with real-time video understanding, progress tracking and multi-robot collaboration, and is available through the Gemini API. — [Wikipedia: Gemini Robotics](https://en.wikipedia.org/wiki/Gemini_Robotics) (snippet only)
- NVIDIA GR00T N2 was announced at GTC 2026. It is a "world-action model" for 30+ DoF full-body humanoids that predicts both the next action and the next observation. — [RoboCloud Hub](https://robocloud-dashboard.vercel.app/learn/blog/gr00t-world-action-models-2026) [secondary]; [MarkTechPost, 2026-04-28](https://www.marktechpost.com/2026/04/28/top-10-physical-ai-models-powering-real-world-robots-in-2026/)
- Physical Intelligence: pi0 is built on PaliGemma (~3B), followed by pi0.5. — [Rohit Bandaru](https://rohitbandaru.github.io/blog/Foundation-Models-for-Robotics-VLA/)

**Funding and valuations**
- Robotics/physical AI startups raised a record $27.6B in 2025. — [futureinvestments.news](https://futureinvestments.news/p/when-intelligence-gets-a-body-the-investment-case-for-physical-ai) [secondary]
- Physical Intelligence: last confirmed valuation was $5.6B (a $600M Series B, Nov 2025). In March 2026 it was reported to be in talks to raise $1B at more than $11B. — [Teahose](https://www.teahose.com/guides/physical-intelligence-valuation); [PYMNTS](https://www.pymnts.com/news/artificial-intelligence/2026/physical-intelligence-seeks-1-billion-as-robotics-interest-grows/)
- Reported valuations: Figure AI $39B, Skild AI more than $14B, Boston Dynamics an implied $21-28B. — [TSG Invest](https://tsginvest.com/physical-intelligence/) [secondary]

**Industrial robots (IFR World Robotics 2025, measured)**
- 542,000 industrial robots were installed in 2024, more than double the figure 10 years earlier and the fourth straight year above 500k. Asia took 74% of installations, Europe 16% and the Americas 9%. China accounted for 54%; Japan installed 44,500 (-4%). The operational stock was 4.664M (+9%). IFR's outlook was low single-digit growth in 2025, then mid single digits. — [IFR press release, Sep 2025](https://ifr.org/ifr-press-releases/news/global-robot-demand-in-factories-doubles-over-10-years)
- IFR's World Robotics 2026 (2025 installation data) is typically released in late September. It was not found in this pass.

### Inferences
- Humanoid volume roughly doubles to triples each year from a small base: about 13-16k in 2025, a likely 35-60k in 2026. By 2027-2028, humanoids could reach 5-10% of annual industrial-robot unit installations. They remain small in dollar terms.
- Milestone: global humanoid shipments of at least 100k in calendar 2027 (any source such as SAG or IDC). Probability about 35%. A tripling from about 40k is needed; China subsidies push the number up, while the lack of proven productive use-cases pulls it down.
- Milestone: Tesla sells Optimus externally (commercial customer deliveries) by end-2027. Probability about 35%, given Tesla's history of missed timelines and Musk's "data collection" framing.
- Milestone: a humanoid fleet of 1,000 or more units at a single third-party customer site doing productive work, publicly verified, by end-2028. Probability about 40%.
- Leading indicators: the SAG/IDC/TrendForce quarterly shipment trackers; the average selling price trend (Unitree's low-cost models pull the ASP down); the IFR World Robotics 2026 release; hours and interventions disclosed from BMW/Figure-type pilots; foundation-model benchmarks on unseen tasks; down rounds versus up rounds for PI, Figure and Skild.

### Gaps
- No reliable 2025-2026 shipment or price data for Agility Digit, 1X NEO (consumer preorder deliveries), or Apptronik.
- Unit prices were not captured for Unitree G1/H2, UBTech Walker S2, or Figure 03.
- Discrepancies between shipment trackers (Unitree 4,200 vs 5,500+) are unresolved.
- The IFR 2026 report (2025 data) was not yet found. No confirmed release date or details for pi0.6 or later Physical Intelligence models.

## Autonomous vehicles (robotaxis, safety, trucking)

### Takeaway
Waymo is the clear Western scale leader. It reached about 500k paid rides/week across about 10 US cities by March 2026, was still around 500k in July, and targets 1M/week by end-2026. It has strong peer-reviewed-style safety data. Baidu Apollo Go is at comparable weekly volume (about 300-350k) in China and abroad. Tesla has moved into limited unsupervised operation (about 200 unsupervised-designated vehicles) with the Cybercab starting in September 2026, but it is roughly two orders of magnitude behind Waymo on driverless miles. Aurora runs driverless trucks at small scale (about 440k driverless miles).

### Cited Findings
**Waymo (primary and press)**
- About 400k paid rides/week across 6 cities in early 2026, with a stated target of more than 1M paid rides/week by end-2026 (co-CEO Tekedra Mawakana). — [Yahoo Finance](https://finance.yahoo.com/news/waymo-eyes-1-million-paid-152515066.html); [Automotive World](https://www.automotiveworld.com/news/waymos-metric-for-2026-success-one-million-weekly-rides/)
- About 500k paid rides/week in 10 US cities (March 2026), doubling in under a year. It first crossed 250k/week in April 2025. — [TechCrunch, 2026-03-27](https://techcrunch.com/2026/03/27/waymo-skyrocketing-ridership-in-one-chart/); [InsideEVs](https://insideevs.com/news/791245/waymo-rides-per-week-2026-doubled/)
- 2026-07-08: fully driverless service announced for four more US cities, with about 500k rides/week at that time. — [SpaceDaily](https://spacedaily.com/m-on-july-8-2026-waymo-announced-fully-driverless-taxis-for-four-more-american-cities-its-fleet-now-giving-about-half-a-million-rides-a-week-and-aiming-for-a-million-a-week-before-the-year-is-out/) [secondary]
- Safety (Waymo Safety Impact hub, through March 2026): 220.6M rider-only miles. Compared with human benchmarks, crashes with serious or fatal injury were 94% lower, airbag-deployment crashes 82% lower, and any-injury crashes 82% lower. That equals an estimated 47 fewer serious-injury crashes, 305 fewer airbag crashes and 707 fewer injury crashes. Waymo drives more than 4M miles/week. — [Waymo Safety Impact](https://waymo.com/safety/impact/); [Waymo blog, 2026-03-19](https://waymo.com/blog/shorts/waymo-safety-impact-update-170m/); [June 2026 update](https://waymo.com/blog/shorts/safetydata-june26/)
- Peer-reviewed predecessor: a crash-type comparison at 56.7M miles. — [Traffic Injury Prevention, 2025](https://www.tandfonline.com/doi/full/10.1080/15389588.2025.2499887)

**Tesla**
- Robotaxi launched in Austin on 2025-06-22 with safety monitors. — [TechCrunch](https://techcrunch.com/2025/06/22/tesla-launches-robotaxi-rides-in-austin-with-big-promises-and-unanswered-questions)
- The unsupervised fleet was reported ramping in April 2026, and at about 25 unsupervised vehicles at one point. — [Electrek, 2026-04-30](https://electrek.co/2026/04/30/tesla-robotaxi-unsupervised-finally-signs-ramping-up/); [eletric-vehicles.com](https://eletric-vehicles.com/tesla/tesla-ramps-unsupervised-robotaxi-fleet-to-25-vehicles/)
- As of September 2026: 420 autonomous vehicles registered in Texas (2026-09-03), 45 Cybercabs authorized for driverless operation, about 200 US vehicles designated "unsupervised", and more than 1M cumulative unsupervised miles. Expansion to Miami, Orlando and Tampa came in July 2026, and the Cybercab rollout began in September 2026. — [Wikipedia: Tesla Robotaxi](https://en.wikipedia.org/wiki/Tesla_Robotaxi) (snippet); [Quartz](https://qz.com/tesla-cybercab-driverless-robotaxi-austin-090326); [Engadget](https://www.engadget.com/2240859/tesla-robotaxi-fleet-might-finally-be-driving-around-austin-unsupervised/)
- Skepticism: reports have challenged claims that the Austin service is truly unsupervised. — [Seeking Alpha](https://seekingalpha.com/news/4542023-teslas-claims-of-unsupervised-austin-robotaxi-challenged---report)

**China / international robotaxis**
- Baidu Apollo Go: weekly orders peaked above 300k in Q4 2025, with more than 20M cumulative rides by February 2026 and expansion to South Korea. — [CnEVPost, 2026-02-27](https://cnevpost.com/2026/02/27/baidu-apollo-go-robotaxi-300000-weekly-rides-expands-to-south-korea/). A peak of about 350k weekly driverless rides in March 2026 across 27+ cities. — [sqmagazine](https://sqmagazine.co.uk/robotaxi-statistics/) [secondary]
- Conflicting claim: "Apollo Go cumulative service volume surpassed 100 million rides" in 1H 2026. — [bydtoday](https://bydtoday.com/china-robotaxi-commercial-scale-2026/) [secondary; this is inconsistent with 20M in Feb 2026 at about 300-350k/week, so treat as unverified or a different metric]
- Pony.ai: fleet above 1,700 and Q1 2026 robotaxi revenue of RMB 59.1M. Operates in Shenzhen, Guangzhou, Hangzhou, Changsha, Croatia, Doha, Dubai and Singapore, and targets 20+ cities by end-2026. — [sqmagazine / Zacks](https://it.tradingview.com/news/zacks%3A810087e30094b%3A0-china-s-av-push-bidu-pony-wrd-lead-the-robotaxi-revolution) [secondary]
- WeRide: about 1,000 robotaxis in China. Launched fully driverless paid service in Dubai via Uber on 2026-03-31. Also operates in Abu Dhabi and Riyadh, with a commitment of 1,200+ Middle East vehicles by about 2027. — [sqmagazine](https://sqmagazine.co.uk/robotaxi-statistics/) [secondary]

**Autonomous trucking (Aurora)**
- Nearly 440,000 cumulative driverless miles through 2026-06-30, with 100% on-time performance and no collisions attributed to the Aurora Driver (company-reported). Q2 2026 net loss was $270M on revenue of $2M. Plan: exit 2026 with about 200 driverless trucks as Aurora Driver 2 launches. — [TradingView/Aurora Q2](https://www.tradingview.com/news/tradingview:a2ad9186cb004:0-aurora-innovation-posts-q2-net-loss-of-270m-revenue-2m-outlines-driverless-truck-scale-up/); [Seeking Alpha](https://seekingalpha.com/news/4621122-aurora-outlines-plan-to-exit-2026-with-200-driverless-trucks-as-aurora-driver-2-launches); [FreightWaves](https://www.freightwaves.com/news/aurora-q2-earnings-driverless-truck-rates); [Aurora 8-K](https://www.sec.gov/Archives/edgar/data/0001828108/000182810826000082/businesswire-aurora2026ana.htm)

### Inferences
- Waymo went from 250k (Apr 2025) to 500k (Mar 2026): about 2x per year. Hitting 1M/week by 2026-12-31 needs another doubling in about 9 months, and volume appeared flat around 500k from March to July. Probability Waymo publicly announces 1M or more paid rides/week by end-2026: about 30%. By end-2027: about 75%.
- Waymo at 2M or more paid rides/week by end-2028: about 50%. The constraints are vehicle supply (Zeekr/Hyundai Ioniq 5 lines) and permitting in new states.
- Tesla at 1,000 or more truly driverless (no employee aboard) vehicles in commercial service by end-2027: about 50%. Tesla publishing Waymo-style third-party-comparable safety data by end-2027: about 25%.
- Apollo Go at 1M or more weekly rides by end-2028: about 45%.
- Aurora at 1,000 or more driverless trucks by end-2028: about 30%. Its 2026 target is 200.
- Leading indicators: Waymo/Alphabet quarterly ride disclosures; CPUC quarterly AV data (passenger trips, VMT); CA DMV collision reports; the NHTSA Standing General Order crash database; Texas registration counts for Tesla; Baidu quarterly Apollo Go ride counts; Aurora quarterly driverless truck count and miles.

### Gaps
- CPUC/CA DMV quarterly data for 2026 was not retrieved in this pass.
- The exact Waymo city list as of September 2026 was not verified (about 10 in March plus 4 announced in July; some may be announced but not yet live, and London/Tokyo status is unknown).
- No independent safety comparison exists for Tesla or Chinese robotaxis.
- Zoox, May Mobility, Kodiak and Chinese trucking were not covered.

## Drones and defense autonomy

### Takeaway
Ukraine is the global volume leader and test lab. It produces millions of low-cost FPV drones a year, and cheap autonomous interceptors are displacing missiles against Shaheds. The US is trying to buy Ukrainian designs and scale domestic output through the Drone Dominance Program, but acknowledges it is years behind.

### Cited Findings
- Ukraine is projected to produce 6-7M small FPV attack drones in 2026, about 500k/month, according to Travis Metz, deputy director of the Pentagon's Drone Dominance Program. Zelensky has claimed Ukraine aims for up to 10M. — [US News/Reuters, 2026-07-27](https://www.usnews.com/news/top-news/articles/2026-07-27/pentagon-says-us-industry-still-years-from-matching-ukraines-wartime-output)
- The Pentagon says US industry is still years from matching Ukraine's wartime output. — [same](https://www.usnews.com/news/top-news/articles/2026-07-27/pentagon-says-us-industry-still-years-from-matching-ukraines-wartime-output)
- Interceptors: SkyFall P1-SUN costs about $1,000/unit and Wild Hornets Sting about $2,500. Production is scaling toward about 2,000/day. In January 2026, Ukraine downed a record 1,704 Shaheds, about 70% of them by interceptor drones. — [Military Times, 2026-03-11](https://www.militarytimes.com/news/pentagon-congress/2026/03/11/these-are-ukraines-1000-interceptor-drones-the-pentagon-wants-to-buy/); [DroneXL, 2026-03-22](https://dronexl.co/2026/03/22/ukraines-interceptor-5-drones-air-defense/)
- The Pentagon and Gulf states are turning to Ukrainian interceptors as Patriot stocks are strained. — [Quwa](https://quwa.org/middle-east-military-news/pentagon-and-gulf-states-turn-to-ukraines-2500-drone-interceptors-as-patriot-stocks-get-strained/)
- The US Drone Dominance Program reportedly aims to acquire more than 200,000 drones by 2027. — [search summary; Kyiv Post](https://www.kyivpost.com/post/85225) [secondary]

### Inferences
- The cost-exchange ratio ($1-2.5k interceptor vs. a Shahed that likely costs tens of thousands) is driving rapid doctrinal change. Expect NATO and Gulf procurement of interceptor drones at 100k+ scale by 2028. Probability about 60%.
- Autonomy (terminal guidance lock-on that is robust to EW) is becoming standard. Fully autonomous target selection remains politically contested.
- Leading indicators: DoD Drone Dominance contract awards; Ukrainian monthly production statements; US production of Ukrainian designs under license; Shahed interception ratios.

### Gaps
- Status of the US Replicator program, Anduril/Shield AI production volumes and CCA (Collaborative Combat Aircraft) milestones were not researched in this pass.
- China's military drone production data is unavailable.

## Space: Starship, launch cadence, Starlink/Kuiper, Artemis, China lunar, stations, orbital data centers

### Takeaway
Launch cadence hit records: 329 global attempts in 2025 (+25%), including 165 Falcon 9 flights. Starship V3 debuted in May 2026 (Flight 12, with the booster lost) and flew a successful Flight 13 in July 2026 that deployed 20 Starlinks. Ship-to-ship propellant transfer, the gating item for Artemis III/IV landings, has not yet been demonstrated and is targeted for late 2026. Artemis II flew crew around the Moon in April 2026. Starlink reached about 12M subscribers (June 2026) versus Amazon Leo's roughly 377 satellites. Commercial stations are slipping: Vast Haven-1 moved to 2027. Orbital data centers remain early-stage demos.

### Cited Findings
**Launch cadence (measured)**
- 329 orbital launch attempts worldwide in 2025, 25% above the 2024 record (Jonathan's Space Report). — [SpaceNews](https://spacenews.com/spacex-china-drive-new-record-for-orbital-launches-in-2025/); [Aviation Week](https://aviationweek.com/space/launch-vehicles-propulsion/spaceops-global-orbital-launch-rate-jumped-25-2025)
- SpaceX flew 165 Falcon 9 missions in 2025 (vs 134 F9 + 2 FH in 2024), about 85% of US orbital launches and nearly twice China's total. — [Space.com](https://www.space.com/space-exploration/private-spaceflight/spacex-shatters-its-rocket-launch-record-yet-again-167-orbital-flights-in-2025) (headline says 167 including other vehicles); [SpaceDaily](https://spacedaily.com/t-spacex-launched-165-falcon-9-rockets-into-orbit-in-2025-nearly-one-every-other-day-accounting-for-roughly-85-of-all-u-s-orbital-launches-and-almost-twice-as-many-orbital-launches/)
- Cost per kg: no fresh 2026 primary figure found (see Gaps).

**Starship**
- Flight 12 (2026-05-22) was the first flight of V3/Raptor 3 and the first from Pad 2. The ship lost one RVac but completed its mission and deployed modified Starlinks. The booster failed its boostback and crashed. The FAA declared a mishap on 2026-05-27. — [SpaceX](https://www.spacex.com/launches/starship-flight-12); [Space.com](https://www.space.com/space-exploration/launches-spacecraft/spacex-starship-v3-megarocket-first-test-flight)
- The FAA concluded its review on 2026-07-13. Flight 13 launched on 2026-07-24, after an earlier abort, and was the first to deploy operational satellites: 20 Starlinks deployed on a suborbital trajectory, which made contact with the constellation. — [Spaceflight Now](https://spaceflightnow.com/2026/07/23/live-coverage-spacex-ready-for-2nd-attempt-to-launch-starship-flight-13-following-post-abort-engine-work/); [SpaceX](https://www.spacex.com/launches/starship-flight-13)
- Flight 14 was listed on the FAA operations plan with a NET of 2026-09-18. Its outcome was not confirmed in this pass. — [Starship wiki (fandom)](https://starship-spacex.fandom.com/wiki/Starship_Flight_14) [secondary]
- The Propellant Transfer Demonstration (ship-to-ship) is planned for late 2026. Earlier targets (March-June 2026) have slipped. — [Wikipedia: Starship Propellant Transfer Demonstration](https://en.wikipedia.org/wiki/Starship_Propellant_Transfer_Demonstration) (snippet); [SpaceX updates](https://www.spacex.com/updates)

**Starlink / direct-to-cell / Kuiper (Amazon Leo)**
- Starlink subscribers: 10.3M as of 2026-03-31 and about 12M as of June 2026 (SpaceX statement on X, 2026-06-04). — [Axis Intelligence](https://axis-intelligence.com/starlink-statistics/) [secondary]; see also the [Starlink progress page](https://starlink.com/progress)
- Direct to Cell: the first-generation constellation of 650+ satellites was deployed over about 18 months, and more than 12M people have connected at least once. — [Starlink progress](https://starlink.com/progress)
- Amazon Leo (Kuiper): about 377+ production satellites in orbit across 12 missions (Atlas V, Falcon 9, Ariane 6) as of September 2026. — [OrbitalRadar](https://orbitalradar.com/satellites/operator/amazon-leo) [secondary]. (Amazon's FCC milestone is about 1,618 satellites, half the constellation, by July 2026. Amazon was far short of it and had sought an extension. That is from background knowledge and was not verified in this pass.)

**Artemis**
- Artemis II (Wiseman, Glover, Koch, Hansen) splashed down on 2026-04-10 at 8:07 pm EDT after a roughly 10-day lunar flyby. — [NASA](https://www.nasa.gov/image-article/artemis-ii-splashes-down/)
- NASA says Artemis III is to fly in 2027, with lunar surface missions beginning in 2028. This reflects a restructured Artemis III, not a landing on it. — [NASA initial Artemis II assessments](https://www.nasa.gov/missions/nasa-on-track-for-future-missions-with-initial-artemis-ii-assessments/)

**China lunar**
- Chang'e-7 (south-pole orbiter, lander, hopper and rover on a Long March 5) was planned for 2026. One search snippet claims a delay to 2027 due to Typhoon weather. That is unverified and the attribution looks garbled. — [NSSDCA](https://nssdc.gsfc.nasa.gov/nmc/spacecraft/display.action?id=CHANG-E-7); [SpaceLaunchSchedule](https://www.spacelaunchschedule.com/launch/long-march-5-change-7/)
- China's crewed lunar landing target is "before 2030" (Mengzhou/Lanyue/Long March 10), per background knowledge. Its 2026 test status was not verified in this pass.

**Commercial stations (ISS replacement)**
- Vast Haven-1 slipped from May 2026 to NET Q1 2027 on Falcon 9. It is a single module with a planned 3-year life and 4 missions of 4 crew for 2 weeks each. Vast is bidding for NASA's Commercial LEO Destinations (CLD) Phase 2. — [Payload Space](https://payloadspace.com/vast-delays-haven-1-launch-to-2027/); [Space.com](https://www.space.com/space-exploration/private-spaceflight/vast-space-now-aims-for-2026-launch-of-haven-1-space-station-after-key-milestone-photos)
- ISS retirement is planned for about 2030 (US Deorbit Vehicle by SpaceX), per background knowledge. Not re-verified here.

**Orbital data centers / space-based compute**
- Starcloud launched Starcloud-1 with an NVIDIA H100 in November 2025. It raised a $170M Series A at a $1.1B valuation (March 2026), filed an 88,000-satellite constellation plan (March 2026), and plans its second satellite for October 2026 with 100x the power generation. — [TechCrunch, 2026-03-30](https://techcrunch.com/2026/03/30/starcloud-raises-170-million-series-ato-build-data-centers-in-space); [GeekWire](https://www.geekwire.com/2026/orbital-ai-seattle-area-startup-starcloud-hits-1-1b-valuation-to-build-space-based-data-centers/); [KeepTrack](https://keeptrack.space/space-brief/space-brief-2026-03-16)
- Not researched in this pass: Google Project Suncatcher, SpaceX/xAI orbital compute claims, and space-based solar power demos.

### Inferences
- Milestone: first ship-to-ship Starship propellant transfer by 2026-12-31. Probability about 20%, given only one successful V3 flight by July and repeated slips. By end-2027: about 65%.
- Milestone: Starship catches and reuses a ship (upper stage) by end-2027. Probability about 45%.
- Milestone: crewed lunar landing (NASA, any lander) by end-2028. Probability about 35%. By end-2030: about 65%. HLS depends on 10+ refilling flights.
- Milestone: Chinese crewed lunar landing by end-2030. Probability about 55%. China has a strong record of meeting schedules, but Long March 10 and Lanyue are still in testing.
- Global orbital launch attempts: at least 400 in 2027 (about 60%) and at least 500 in 2029 (about 50%), with growth driven by Falcon 9, Chinese megaconstellations (Guowang, Qianfan) and Starship.
- Starlink reaching 20M or more subscribers by end-2027: about 65%. From about 12M in June 2026 it was adding about 0.5-0.6M/month (10.3M on Mar 31 to about 12M in early June).
- A commercial station (Haven-1) crewed in orbit by end-2027: about 45%. At least one CLD station operational before ISS deorbit (about 2030-31): about 50%.
- An orbital data center at 1 MW or more of compute power in orbit by end-2030: about 20%.
- Leading indicators: FAA Starship licenses and mishap closures; SpaceX flight-rate statements; Jonathan's Space Report monthly counts; Artemis III hardware milestones (SLS core stage, HLS propellant tests); CNSA Long March 10 test flights; NASA CLD Phase 2 awards; Amazon Leo satellite count and FCC extension; Starcloud-2 performance.

### Gaps
- Starship Flight 14 outcome (NET 2026-09-18) was not confirmed.
- No 2026 year-to-date global launch count was found. McDowell's site was not directly fetched.
- Cost per kg: no current primary figure. Falcon 9 list price and Starship target costs were not verified this pass.
- Chang'e-7 status is uncertain (conflicting or garbled delay snippet).
- China crewed lunar hardware test status was not verified.
- Axiom, Starlab and Orbital Reef timelines were not researched.
- Space-based solar power demos and other orbital compute proposals were not covered.
- The MIT Technology Review 2026 "10 Breakthrough Technologies" list was not checked.
