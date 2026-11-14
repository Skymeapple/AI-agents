# Reported growth notes, 2026-11-12

Collector: agent:web-research. Tools: Tavily search and Tavily extract both worked (keyed). Extract read the Glia press release, the SiliconANGLE Daylight article, the Higgsfield CEO's X post, the CellCog Higgsfield write-up and an MLQ Rox article; it failed only on dealroom.co ("Failed to fetch url"). WebSearch worked but returned mostly older items. WebFetch got ENOTFOUND for www.glia.com (outside the session's allowed network domains), so I did not try other hosts with it. All three recorded figures come from pages read, not snippets. The file has 3 observations for 3 apps. Every candidate was checked against observations.jsonl: Higgsfield's earlier rows are $700M (Aug), $1B (Sep 25) and +1900% (365 days), with no $1.35B row; Glia and Daylight had no earlier rows.

## Recorded
- Higgsfield: CEO Alex Mashrabov's X post (Nov 11, 1:07 AM): "We crossed $1B and got faster. In October we hit $1.35B." (annualized run rate). revenue_annualized_usd 1,350,000,000, no basis. The post names only the month, so this is dated to the Nov 11 post date. Page read. https://x.com/alexmashrabov/status/2120322553600430213
- Glia: press release (Nov 9): "Glia today announced it has surpassed $100 million in annual recurring revenue (ARR)". revenue_annualized_usd 100,000,000, no basis, dated to the Nov 9 announcement date because the release gives no as-of date. Page read. https://www.glia.com/news/glia-surpasses-100m-arr-powered-by-accelerated-adoption-of-its-banking-ai-platform
- Daylight Security: SiliconANGLE (Nov 10): "raised $80 million in new funding after what it says was more than 40-fold revenue growth over the past year". growth_pct +3900, period 365, dated to the Nov 10 announcement date. Company-stated, no revenue level given. Page read. https://siliconangle.com/2026/11/10/ai-driven-managed-security-startup-daylight-raises-80m-on-40-fold-revenue-growth

## Seen but not recorded
- Higgsfield: "$2B in annualized run rate in the next 3 months" is a forecast. The $1B to $1.35B move ("+35%" in secondary posts) is a rate derived from two levels.
- Glia: "sales of Glia's AI solutions grew 93% year over year in the third quarter of 2026". The release does not say whether "sales" means revenue, ARR or bookings, and it covers only the AI solutions line. "Over 95 million U.S. consumers" are the client banks' customers, not Glia's users.
- Rox ($100M Series B at $1.3B, Fortune exclusive, Nov 11): ARR "grew more than tenfold since its first year after launch" (Fortune) / "during its first year in the market" (other outlets). The window is unclear, and the first year in market likely ended more than 90 days ago. No dollar ARR. Net revenue retention >700% is not a tracked metric.
- Path Robotics ($300M Series E, Business Wire, Nov 10): ARR "growing 17-fold over the past 18 months". The window is in months and is not a quarter or year, so it is left out.
- Wix Base44 (Q3 2026, Nov 10): ARR "close to $250 million" (secondary summary). An approximate ceiling, so not recorded. Wix Symphony ARR "in the millions of dollars" is not a number.
- DeepSeek Harness: 1.54M DAU as of Nov 4 is already recorded (Nov 9 file). A "+545%" DAU figure and "5.48M monthly active users" appear only in a fan's X post and FourWeekMBA, not in a company source I read.
- Fireworks AI $1B+ ARR: the July 2026 Series D figure re-quoted in Nov 7–10 coverage, with no new date. Meshy 12x ARR growth: July round, quoted secondhand in a podcast.
- PuppyGraph ($10M Series A, Nov 11): ARR "grew 7x over the past year". This is a graph query engine, not an AI application.
- Aura ($355.0M ARR, +24% YoY, Q3 2026): a whole-company metric of an identity-protection firm, not an AI company. Kaltura, Arlo and Exasol ARR: same reason.
- OpenAI quarterly revenue $10B (Q3): "according to a person familiar", an unnamed source. Nebuly ($21M Series A), Serval ($150M Series C, earlier) and Mercor ($750M at $20B): no ARR, users or growth figure found.
- Tencent Q3 2026 results: not out yet when searched (expected after market Nov 12). Yuanbao Search launched Nov 11 with no user figures. Baidu reports Nov 16. QuestMobile Yuanbao MAU (third-party) not recorded.

## Trends
- Run-rate disclosures are moving to founders' social posts (Higgsfield) ahead of company documents. These need extra care over dating and basis.
- Funding announcements this week (Daylight, Rox, Path Robotics, PuppyGraph) led with growth multiples rather than dollar levels, and several used windows the radar can't record (first year in market, 18 months).


## Review by the daily run (2026-11-12)

- **Glia $100M ARR dropped.** It is the whole company's ARR. Glia has sold a
  digital customer-service platform since long before its AI products, and
  the release does not separate AI revenue from the rest. The rule excludes
  whole-company metrics of a non-AI company; monday.com was handled the same
  way on Nov 10, where only its AI-products line was recorded.
