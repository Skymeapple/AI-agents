# Reported growth notes, 2026-11-13

Collector: agent:web-research. Tools: Tavily search and Tavily extract worked (keyed). Extract read the Retell release (Yahoo Finance copy), the Kindo PR Newswire release, the Unite.ai Kindo article and The AI Insider Blue J article. It failed on businesswire.com (request timed out) and dealroom.co ("Failed to fetch url"). WebSearch (extended) worked but found nothing dated Nov 12–13; it returned older roundups. WebFetch was not tried. All recorded figures come from pages read, not snippets. The file has 3 observations for 2 apps. I checked every candidate against observations.jsonl: Retell (retellai.com) and Kindo (kindo.ai) had no earlier rows.

## Recorded
- Retell: Business Wire release (Nov 12), "Key stats as of November 12, 2026": "surpassed $100 million in annualized run rate". revenue_annualized_usd 100,000,000, no basis, dated Nov 12 (stated as-of date). Page read. https://finance.yahoo.com/technology/ai/articles/retell-surpasses-100m-annualized-run-140000130.html
- Retell: same release: "Serves more than 56,000 monthly active users and 4,000 businesses". users 56,000, basis mau, dated Nov 12. Page read. Same URL.
- Kindo: PR Newswire release (Nov 12): "The company grew revenue by approximately 800% year over year in 2026". growth_pct +800, period 365. The release gives no period end, so this is dated to the Nov 12 announcement date. Page read. https://www.prnewswire.com/news-releases/kindo-secures-57-6-million-to-expand-enterprise-control-over-ai-agent-actions-and-reduce-costs-302937058.html

## Seen but not recorded
- Tencent Q3 2026: not out. TheEdge/i3investor (Nov 13) says Tencent "is scheduled to report third-quarter earnings on Nov. 18", not Nov 12. Yuanbao Search launched Nov 11 (so.qq.com) with no user figures. Coverage gives index size and "hundreds of millions of daily calls", which are usage counts. QuestMobile September MAU (Doubao 420M, Qwen 158M, DeepSeek 123M, Yuanbao 37.33M) and its AI office app figures are third-party, so not recorded.
- Retell: "$10 million to $100 million in 17 months" uses a window in months that is not a quarter or year. It is also two levels, not a stated rate. "Enterprise revenue 9x in 2026" covers one segment over a year-to-date window. Forbes gives "21,000 businesses" where the release says 4,000. Business counts are not a tracked metric. "100 million connected calls a month" is a usage count.
- Blue J ($176M at $700M, The AI Insider, Nov 12, page read): "Recurring revenue has since grown 165% to the mid-eight-figure range". The window runs from the August 2025 round to now, about 15 months, which is neither a quarter nor a year, so it is left out. "Mid-eight-figure" is not a number.
- Rox ($100M Series B, Nov 11): ARR "more than tenfold" in its first year in market. The window is unclear and was already declined in the Nov 12 file.
- Hatz AI ($24M Series A, Nov 12): "revenue grew even faster" than a tripled partner count. No figure was given.
- LangChain ($300M at $3B, WSJ Nov 12): coverage shows no ARR, users or growth figure.
- Future Secure AI ($82M, AU unicorn): ARR "to approach US$100 million by the end of 2026" is a forecast and an approximate ceiling.
- Nebius Q3 (Nov 12): ARR $6.4B (+1,062% YoY). This is GPU cloud infrastructure, not an AI application.
- Trend Micro TrendAI ARR: Trend Micro's existing security business under a new name, not a separately stated AI-product line. CoreWeave Q3 is AI infrastructure. Twist Bioscience's AI business "triple-digit growth" has no number and Twist is not an AI company.
- Adobe AI-first ARR "topped $650 million" (Futurum recap of Q3 FY26): an approximate floor re-quoted from the September quarter. The +150% YoY rate is already recorded. Adobe's ">1 billion MAU" covers all its products.
- Meta Muse "10 million downloads in 40 days" is a downloads count from a Sensor Tower/TechCrunch relay. Muse ">10 million users" is already recorded as registered accounts.
- Peec AI $20M ARR (Nov 2): already recorded. "Doubling revenue in just six months" uses a 6-month window and is left out.
- Emergent $120M run rate, 10M users, 4x in six months: run rate already recorded. The users figure is undated in a Facebook relay, and the 6-month window is left out.
- Anthropic/OpenAI Q3 revenue (The Information, Nov 10): quarterly revenue from an exclusive report, not run-rate or company-stated.

## Trends
- Bootstrapped and capital-light AI companies (Retell, at $4.6M raised) are now announcing $100M run rates on their own, without a funding round.
- This week's funding releases again led with growth multiples instead of dollar levels (Kindo, Blue J, Rox, Hatz), and several used windows the radar can't record.
