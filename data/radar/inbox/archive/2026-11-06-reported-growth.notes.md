# Reported growth notes, 2026-11-06

Collector: agent:web-research. Tools: WebSearch worked (index for Nov 4–6 thin; summaries sometimes contradict each other). WebFetch got ENOTFOUND (outside the session's allowed network domains) for www.benzinga.com and www.sec.gov, so I stopped fetching. No page was read; every figure comes from search-result snippets or summaries. Tavily was unavailable (needs authorization). The file has 4 observations for 4 apps. Every candidate was checked against observations.jsonl; none of the four apps had an earlier observation.

## Recorded
- E2B: Business Wire release (Nov 5): passed "$100 million in annualized run rate", "first public revenue disclosure", "almost 100% from ALG: Agentic-Led Growth". Headline says "Passes $120 Million" — conflict; recorded the $100M threshold, reviewer should check. Dated to Nov 5 announcement; no basis. https://www.financialcontent.com/article/bizwire-2026-11-5-e2b-which-gives-ai-agents-their-own-cloud-computers-passes-120-million-in-annual-recurring-revenue-with-100-inbound
- Figma MCP server: Q3 2026 call (Nov 4): "approximately 1 million weekly active users", "our fastest-growing product by growth in weekly active users". basis wau; no as-of date, dated to the Nov 4 call. https://www.fool.com/earnings/call-transcripts/2026/11/05/figma-fig-q3-2026-earnings-call-transcript/
- Cars.com Carson: Q3 2026 results (Nov 5): "crossed 1 million monthly users, nearly doubling year-over-year"; call: "surpass 1 million monthly users during Q3". basis mau; dated to Nov 5 announcement (no as-of date). App URL is the Carson launch release. https://investor.cars.com/2026-11-05-Cars-com-Reports-Third-Quarter-2026-Results
- SoundHound AI: Q3 2026 8-K (Nov 5): revenue "$62.4 million, up 48.5% year over year" (includes LivePerson acquisition). growth_pct 48.5 over 365 days, dated to Nov 5 release. https://www.sec.gov/Archives/edgar/data/0001840856/000184085626000025/soun-20261105ex991earnings.htm

## Seen but not recorded
- E2B "at $8m just a year ago" and "projected to add over $150M of net new revenue this year": two levels / forecast, no growth rate derived. "2 billion agent computers started": usage.
- Cars.com Carson "nearly/almost doubled year-over-year": imprecise, not a stated rate.
- Figma MCP "total MCP usage grew 3x quarter-over-quarter", write-back 4x QoQ, "5% of credit consumption": usage, not users. Figma agent used weekly by 65% of paid customers >$10K ARR (Oct 31): a share, not a count. https://www.sec.gov/Archives/edgar/data/0001579878/000162828026070133/q326pressrelease.htm
- Traversal (Nov 3, $120M Series B at $1B): ARR "grown nine times since the start of the year", MAU "nearly 20x", usage ~100x. Window is year-to-date (~10 months, not a quarter or year), so left out per the months rule; no ARR level disclosed. https://thenextweb.com/news/traversal-120m-series-b-1b-valuation-ai-sre ; https://www.unite.ai/traversal-raises-120m-series-b-at-1b-valuation-to-expand-ai-sre/
- Runway (Bloomberg, Nov 3): co-CEO says ARR "set to hit $450 million by the end of the year" — a forecast; latest stated level ($200M, Sep 8) is already in observations.jsonl. https://www.bloomberg.com/news/articles/2026-11-03/runway-ai-says-arr-set-to-double-to-450-million-by-end-of-year
- Axon Q3 2026 (Nov 5): "AI annual recurring revenue surpassed $100 million"; AI bookings "up over 50% year to date". Aggregate of AI products (Draft One etc.) of a non-AI company, not one product; not recorded (same treatment as DigitalOcean AI Customer ARR on Nov 4). https://filecache.investorroom.com/mr5ir_axon/645/AXON_Q3_2026_ShareholderLetter.pdf
- NICE Q3 2026 (Nov 4): "AI ARR: $392 million, up 46% year-over-year". Segment aggregate of a CX software company; not recorded, same reason. If admitted: revenue_annualized_usd 392M and growth_pct 46 / 365 days. https://www.gurufocus.com/news/9171786/nice-ltd-nice-q3-2026-earnings-call-highlights-ai-arr-jumps-46-as-revenue-tops-guidance
- Future Secure AI (AFR via Startup Daily, Nov 6): US$80M at US$1.2B; ARR "expected to approach US$100 million by the end of 2026" — forecast; other searches could not confirm the round. https://startupdaily.net/topic/funding/future-secure-ai-raises-us80-million-for-ai-co-workers
- Zillow AI mode: "more than 2.5 million people used AI mode" March–September — cumulative over a months window, not a level of a defined basis. BlackLine "returning AI users grew 25% quarter over quarter" — transcript flagged as internally inconsistent and no absolute; not recorded.
- ChatGPT 1.2B weekly users (Nov 2 dinner): already recorded (Sep 29). Gemini app "one billion users" (Alphabet Q3 call, October): outside Nov 4–6 focus, older milestone.
- Fal, Applied Compute: covered in the Nov 4 file. Serval ($150M Series C at $3B), Ocient ($100M), Larridin: no ARR/users figure found.

## Trends
- Agent infrastructure keeps posting fast revenue ramps: E2B to a $100M+ run rate from about $8M a year earlier, almost entirely self-serve "agentic-led growth".
- Incumbents report AI as segment ARR (Axon >$100M, NICE $392M +46%) or as feature-level users (Figma MCP ~1M WAU, Cars.com Carson 1M+ monthly users) rather than whole-product numbers.
- MCP is becoming a reported product line: Figma calls its MCP server its fastest-growing product by WAU.

## Dropped before ingest (radar agent)
- SoundHound AI "+48.5% year over year" revenue: whole-company growth that includes the LivePerson acquisition, so it is not organic growth of an AI product. Not recorded.
