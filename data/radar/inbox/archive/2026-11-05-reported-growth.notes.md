# Reported growth notes, 2026-11-05

Collector: agent:web-research. Tools: WebSearch worked, but its index for Nov 3–5 is thin and its summaries often mix in 2025 results or contradict each other. WebFetch got ENOTFOUND (host outside the session's allowed network domains) for www.sec.gov, www.fool.com and ppc.land, so I stopped fetching as the error advised. No page was read; every figure comes from search-result snippets or summaries. Tavily was unavailable (needs authorization). The file has 1 observation for 1 app. Candidates were checked against observations.jsonl (no Shopify/Sidekick, Doubao or Airbnb entries exist).

## Recorded
- Shopify Sidekick, growth_pct +14 over 91 days: Q3 2026 earnings call (Nov 3, 2026), "From the second to the third quarter, merchant conversations rose 14%, while cost per conversation fell 28% and total Sidekick cost declined 18%." Dated to the Nov 3 announcement (snippet gives no period end date). https://www.shashi.co/2026/11/shopify-cuts-sidekicks-cost-per.html ; same figure in a summary of results incl. https://ppc.land/shopify-revenue-gains-32-to-3-7bn-as-ai-referred-orders-rise-3-3x/ and https://finance.yahoo.com/markets/stocks/articles/shopify-inc-shop-q3-2026-210823609.html . Snippet only.

## Seen but not recorded
- Shopify Sidekick "daily active merchant base grew 3.7 times year over year, while daily sessions increased 4.2 times" (one search summary of Q3 coverage): a targeted search found only the Q2 2026 figure (3.6x YoY daily active merchants, ~34M conversations in Q2; https://www.retailtouchpoints.com/news/shopify-credits-ai-for-34-revenue-growth-in-q2-2026/620805/), so 3.7x could not be confirmed as a Q3 figure and was left out. Also, keeping one growth figure per app avoids two growth_pct readings for Sidekick over different windows.
- Shopify Q3 2026 (Nov 3): "AI traffic to merchant storefronts increased 2.6 times year-over-year, while orders originating from AI searches jumped 3.3 times"; Sidekick "created around two-thirds of all Shopify flows". AI-referred traffic/orders to merchants are not an AI product's own metric (they measure third-party chatbots' referrals), so not recorded. https://ppc.land/shopify-revenue-gains-32-to-3-7bn-as-ai-referred-orders-rise-3-3x/
- Doubao (ByteDance): 419M monthly users in September, and AI-native apps 546M monthly users (+82.1% YoY), per QuestMobile reported Nov 3. Third-party estimate, not company-reported, so not recorded. https://fourweekmba.com/ai-bytedances-doubao-hits-419m-users-daily-base-6x-qwens/
- Qianwen (Alibaba): 158M monthly users in September, per QuestMobile. Third-party, not recorded. https://www.chaincatcher.com/en/article/2301251
- ChatGPT 1.2B weekly users (inspire2rise, Nov 4, attributing it to OpenAI's Adam Fry; "200 million users added in the last 60 days"): the 1.2B weekly level is already recorded (Sep 29); the 200M/60-day change is two levels, not a stated rate. https://www.inspire2rise.com/chatgpt-1-2-bn-users-milestone.html
- Fal (Nov 3): $1.3B round at $16B; "$800 million in annualized revenue" already in observations.jsonl. https://fourweekmba.com/ai-fal-hits-16b-in-1-3b-round-about-20x-its-reported-revenue/
- Uber Q3 2026 (Nov 4): gross bookings +20% YoY; ~700 driverless vehicles on Uber by year-end. Company-wide/AV fleet figures, no AI product user or revenue figure. https://www.sec.gov/Archives/edgar/data/0001543151/000154315126000039/uberq326earningspressrelea.htm
- Airbnb, AppLovin, Duolingo, Qualcomm, Arm: searches returned only Q2 2026 or 2025 results; no Q3 2026 AI-product figure found. Airbnb's Q2 figure (AI assistant resolves ~45% of inquiries) is a share, not a radar metric.
- Arena $100M annualized run rate (reached June, announced Oct 8) and Modal $1B annualized run rate (Oct 14): both already in observations.jsonl (web:lmarena.ai, web:modal.com, collected 2026-10-20). https://techcrunch.com/2026/10/08/popular-ai-leaderboard-arena-nearly-doubles-valuation-to-3-1b-valuation-in-10-months/ ; https://modal.com/blog/1-billion-annualized-run-rate-series-d

## Trends
- Incumbents report AI-feature usage as efficiency and engagement ratios (Sidekick conversations +14% QoQ with cost per conversation -28%) rather than user counts.
- AI-referred commerce keeps growing but decelerating in multiple terms at Shopify (orders ~13x in Q1, ~3x in Q2, 3.3x in Q3).

## Dropped before ingest (radar agent)
- Shopify Sidekick "+14% merchant conversations" (Q2 to Q3): a usage count (conversations), not users or revenue, consistent with how earlier runs treated Alexa interactions and token volume; and its only source is a low-authority blog. Not recorded.
