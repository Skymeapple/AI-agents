# Reported growth notes, 2026-11-07

Collector: agent:web-research. Tools: WebSearch worked, but its index for Nov 5–7 is thin and summaries often mix dates. WebFetch got ENOTFOUND (outside the session's allowed network domains) for ai2roi.substack.com and blog.mean.ceo, so I stopped fetching. No page was read; every figure comes from search-result snippets or summaries. Tavily was unavailable (needs authorization). The file has 3 observations for 3 apps. Every candidate was checked against observations.jsonl: none duplicates an existing app/date/value (Gemini had only a Sensor Tower 662M estimate for May; chatgpt.com had no paid_seats reading; ChatGPT Voice is new).

## Recorded
- Google Gemini app: Pichai on X (Aug 11, 2026): app passed "1 billion monthly active users", "fastest-growing product". basis mau; a threshold; dated to the Aug 11 announcement (inside the 90-day window). Snippets only. https://9to5google.com/2026/08/11/gemini-app-1-billion/
- ChatGPT Voice: Adam Fry, Nov 2 press dinner: "about 250 million people are reportedly using ChatGPT Voice" (Gizmodo). Another summary calls it "250 million weekly active users" for voice input (dictation + voice mode), but the source page for that wording is unclear, so no basis. Dated to Nov 2 (no as-of date). App URL chatgpt.com/features/voice/ keeps it apart from ChatGPT's 1.2B WAU. Snippets only. https://gizmodo.com/chatgpt-claims-to-have-achieved-20-of-googles-search-volume-2000834464
- ChatGPT: same dinner: "the most recent public figure, 50 million paying subscribers", said to have grown since (amount not given). basis paid_seats; stale lower bound dated to the Nov 2 statement because no as-of date is given. Snippets only. Same Gizmodo URL.

## Seen but not recorded
- ChatGPT 1.2B weekly users (Nov 2 dinner): already recorded (Sep 29). ChatGPT search volume "roughly one-fifth" of Google's: usage share, not users.
- Gemini app "one billion users" on Alphabet's Q3 call (Oct 28): no basis stated, same level as the Aug 11 MAU record. https://www.fool.com/earnings/call-transcripts/2026/10/29/alphabet-googl-q3-2026-earnings-call-transcript/
- OpenAI "$50 billion in net annualized revenue ... 77% year-over-year" (SiliconANGLE Nov 6, citing FT): 50B and 77% at 2026-09-30 already in observations.jsonl. https://siliconangle.com/2026/11/06/openai-reportedly-raises-11b-in-new-funding-at-1-25t-valuation/
- Anthropic: run rate "over $80 billion" at end of October (WSJ, unnamed source) — not company-reported. Q3 2026 revenue "approximately 1,127%" YoY — only in one secondary explainer (Indmoney) attributing it to an S-1/A of Nov 2 that I could not find; Q3 QoQ +45% and H1 +1,300% already recorded. https://www.indmoney.com/blog/us-stocks/anthropic-ipo-s1-filing-explained
- Five9 Q3 2026: "AI revenue jumps 91%" (TradingKey summary); the 8-K snippet only says AI revenue growth "stepping up significantly". Segment aggregate of a CX software company (same treatment as NICE/Axon on Nov 6) and the rate is unconfirmed. https://www.sec.gov/Archives/edgar/data/0001288847/000128884726000143/a093026ex991earningsrelease.htm
- Upwork Q3 2026 (Nov 2): MCP server WAU "increased 20-fold between the August launch and the end of the third quarter" — window ~2 months, not a quarter or year, so left out per the months rule; no level given. GSV from AI-skilled freelancers ~$450M run rate, +55% YoY — marketplace volume of a non-AI company, not AI product revenue. https://www.fool.com/earnings/call-transcripts/2026/11/02/upwork-upwk-q3-2026-earnings-call-transcript/
- Klaviyo Q3 2026 (Nov 5): Composer "10,000 daily active accounts, more than doubling in the three weeks" after a release — accounts, and "more than doubling" is not a precise rate; not confirmed beyond one summary. https://www.fool.com/earnings/call-transcripts/2026/11/05/klaviyo-kvyo-q3-2026-earnings-call-transcript/
- Dropbox Q3 2026 (Nov 6): AI-tool weekly uploads 600K to 2M+ since September — usage. Datadog "AI customer count" >850 — customers, not users. LegalZoom AI agent resolves ~half of chats — a rate.
- Sentra ($150M, Nov 4): "more than 300% year-over-year growth" with the metric undefined (2025 claim was revenue) — not recorded. https://techfundingnews.com/sentra-150m-capital-one-ventures-ai-data-security/
- Future Secure AI (Nov 6): ARR "expected to approach US$100 million by the end of 2026" — forecast (also noted Nov 6). Pliant ($80M Series C, ">$100M ARR") — B2B payments fintech, not an AI app.
- Alta ($15M run rate, 800% growth) and Alice (~$100M ARR): announced Jul 8 and ~Aug 25 — Alta outside the 90-day window; Alice "approaching" is not a stated level.
- Duolingo DAU 64.9M (+29%), Pinterest 661M MAU: whole-company metrics of non-AI companies.
- Wix Base44, HubSpot Breeze: no Q3 2026 figures indexed yet.

## Trends
- The consumer assistant race is reported in round-number thresholds: Gemini 1B MAU (Aug), ChatGPT 1.2B WAU (Nov 2), with voice as a separately counted surface (~250M).
- Nov 5–7 earnings mostly gave usage or account counts for AI features (Klaviyo, Dropbox, Upwork MCP) rather than user levels or AI revenue for a single product.
- Funding announcements this week (Sentra, Future Secure AI, OpenAI) disclosed growth claims without defined metrics or only forecasts.

## Dropped before ingest (radar agent)
- Gemini "passed 1 billion monthly active users" (Aug 11): the same claim is already recorded for 2026-08-11; adding it again with a basis would count one claim twice.
- ChatGPT "50 million paying subscribers": described as "the most recent public figure", i.e. an older number with no date of its own; dating it to Nov 2 would present stale data as new.
- ChatGPT Voice "about 250 million people": a feature's usage figure ("reportedly"), excluded for the same reason on 2026-11-04.
