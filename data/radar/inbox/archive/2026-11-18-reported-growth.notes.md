# Reported growth notes, 2026-11-18

Collector: agent:web-research. This pass covers Nov 16–18 (the radar did not run on Nov 17). Tools: Tavily search and Tavily extract worked (keyed). WebSearch (extended) also worked but found nothing new. WebFetch was not tried. Extract read these pages: Baidu's Q3 2026 release (ir.baidu.com), the Globe and Mail copy of Baidu's Q3 call transcript, Tencent's IR results page and newsroom, General Atlantic's copy of the Model ML release, the Numa release (PR Newswire), CodeRabbit's blog post and newsroom release, HeyGen's June ARR blog post, and the StreetInsider copy of HeyGen's Avatar 6 release. Every recorded figure comes from a page I read, not from a snippet. The file has 7 observations for 4 apps and passes validation. I checked every candidate against observations.jsonl and earlier notes: DuMate, Model ML and CodeRabbit have no earlier rows. HeyGen has only the June 25 ARR and growth rows, and those are not repeated here.

## Recorded
- Baidu DuMate: on the Q3 call (Nov 16), Robin Li said: "In September, its MAU increased by more than 30-fold quarter-over-quarter." Recorded as growth_pct +2900 over 91 days, dated Sept 30 (only the month is stated, so the day is set to month end, following earlier files). This is MAU growth. Page read. https://www.theglobeandmail.com/investing/markets/stocks/BIDU-Q/pressreleases/8862532/baidu-bidu-q3-2026-earnings-call-transcript
- Model ML: Series B release (Nov 17): "In the third quarter of 2026, Model ML's annual recurring revenue grew 12x year on year and daily active users increased 26x." Recorded as two growth_pct rows over 365 days, both dated Sept 30 (end of Q3): ARR +1100 and DAU +2500. Page read. https://www.generalatlantic.com/media-article/model-ml-raises-250-million-at-1-5-billion-valuation-18-months-after-launching-from-stealth
- CodeRabbit: blog post and Business Wire release (Nov 12). Three rows:
  - "$100 million in annual recurring revenue": revenue_annualized_usd 100M, with no basis set.
  - "400,000 monthly active users" (blog): users 400,000, basis mau.
  - "120,000 daily active users" (newsroom copy of the release): users 120,000, basis dau.
  
  Pages read. The Nov 12 release was missed by the Nov 12–16 passes. "$10 million to $100 million … in just over a year" (also "tenfold") is not recorded as growth, because the window is not exact and the figure compares two levels.
- HeyGen: Avatar 6 release (Nov 17): "HeyGen now reaches over 40 million users across 196 countries". Recorded as users 40M with no basis (the release says "users" and "people"), dated to the Nov 17 announcement. Page read (StreetInsider copy of the PR Newswire release).

## Baidu Q3 2026 (Nov 16)
- The release (page read) gives quarterly recognized revenue for each AI line: AI Cloud Infra RMB 8.2B (+94% YoY), AI Applications RMB 2.6B (0% YoY), AI-native Marketing Services RMB 2.4B (−15% YoY), and Core AI-powered Business RMB 13.2B (+37%). None of these is recorded. They are quarterly recognized revenue, not ARR. AI Cloud Infra is infrastructure. The other lines are blends of several products, which matches how the Q2 Baidu, NICE, Five9 and Digi Power X lines were handled.
- ERNIE Assistant: the only figure is "daily average conversation rounds grew 126% year-over-year" in September. That is a usage count, so it is not recorded. Neither the release nor the call states a user level or user growth for ERNIE Assistant this quarter.
- Baidu App had 604M MAU in September (Quantisnow summary of the release). The app is not a separately stated AI product.
- Qianfan MaaS "external customer token usage revenue … more than 12-fold" (TipRanks call summary, snippet) is token revenue on an infrastructure platform, so it is not recorded.

## Tencent Q3 2026 (Nov 18)
- Not out when I collected. Tencent's IR results page (read) still lists Q2 2026 as the latest result. Longbridge lists the call for Nov 18, 12:00 UTC. Searches in English and Chinese found only previews. A later run should check the Q3 release and call for Yuanbao users.
- Yuanbao figures seen are QuestMobile or Baike third-party estimates (57.35M MAU in Q1, 49.84M in June), or the Feb 18 Spring Festival figures (DAU over 50M, MAU 114M), which are outside the 90-day window. None are recorded.

## Seen but not recorded
- Numa (Nov 16 Series C, PR Newswire, page read): "It caps a year in which Numa tripled in size, tripled its annual recurring revenue". The same release says Numa acquired SocketTime in August, so the tripling may include acquired revenue. The window ("a year") also has no stated end. https://www.prnewswire.com/news-releases/numa-triples-arr-raises-45m-series-c-to-run-the-entire-dealership-302940271.html
- Corma ($125M Series A, Nov 16): ARR "estimated to be in the tens of millions" (Globes) is an estimate with no exact number. "Server capacity 24-fold" is compute, not a tracked metric. Dreamteam ($53M, Nov 16): "more than 100 companies" use it, which is a customer count. LanceDB, Daytona, Arena, Firecrawl and Traversal: no ARR, user or growth figure found.
- HeyGen "$200 million in ARR in June, doubling revenue in eight months" (Nov 17 release): this re-quotes the June 25 figures, which are already recorded. The eight-month window would not be accepted in any case.
- AlphaSense "$600M ARR" (cited in Nov 18 Wowtale coverage of Model ML): from the June 3, 2026 release, outside the window and re-quoted.
- Research Solutions (Nov 17, Q1 FY27): "AI ARR Jumps 38%" (GuruFocus snippet). This is a segment of a research-content company that is not an AI company, and the same treatment applies as for NICE and Axon.
- Similarweb Q3 (Nov 17): AI businesses drove "roughly two-thirds of ARR added", which is a share, not a number. TechnologyOne ARR +18% and Klarna 122M users are whole-company figures from companies that are not AI companies.
- PhysicsWallah PW Talk "18 Cr+ ARR" (Investor Day deck, Nov 17, NSE PDF snippet): the figure is in rupees and a product line of an edtech company, and converting it to USD would be an estimate. "70k+ active paid subscribers" is a subscriber count of a single feature.
- Superhuman/GPTZero "$100M ARR product suite" (a job post): not a release and has no date.

## Trends
- Chinese platforms now disclose AI-agent growth as multiples with no levels. Baidu gives DuMate MAU only as "more than 30-fold QoQ" and ERNIE Assistant only as conversation rounds. Levels come from QuestMobile instead.
- Funding releases for vertical agents in finance (Model ML) and auto retail (Numa) lead with ARR multiples rather than dollar ARR.
- AI code validation has reached $100M ARR at CodeRabbit, with users disclosed on two bases in the same day's materials (400k MAU, 120k DAU).


## Review by the daily run (2026-11-18)

- Baidu DuMate and Model ML rows re-dated to their announcement dates (Nov 16, Nov 17): the sources state only "September" and "the third quarter", and the inbox rules forbid inferring a period end date.
