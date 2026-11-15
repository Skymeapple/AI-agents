# Reported growth notes, 2026-11-15

Collector: agent:web-research. Tools: Tavily search and Tavily extract worked (keyed). Extract read the Anthropic S-1/A-2 PDF (Fidelity-hosted copy), the citybiz Nov 13 Anthropic IPO article, the Five9 Q3 2026 earnings-call transcript (TradingKey) and the Crunchbase "Week's 10 Biggest Funding Rounds" (Nov 13). WebSearch (extended) worked but surfaced only items from Nov 2–8 that earlier files had already covered. WebFetch was not tried. All three recorded figures come from pages read, not snippets. The file has 3 observations for 2 apps. I checked every candidate against observations.jsonl: Claude has no $81B row, and Level AI has no earlier rows.

## Recorded
- Claude (Anthropic): S-1/A Amendment No. 2 (filed Nov 10): "our Run-Rate Revenue was approximately $81 billion as of October 31, 2026, up $10 billion from September 30, 2026." revenue_annualized_usd 81,000,000,000, rounded_to 1B, no basis, dated Oct 31 (stated). The Nov 11 notes declined an "$81B October ARR" because the only source was a secondhand newsletter. This reading comes from the filing itself. Page read. https://www.fidelity.com/bin-public/600_Fidelity_Institutional/fidelityinstitutional/Application/AP154833/documents/clients/ANTLLV/ANTLLVred.pdf
- Level AI: Five9 Q3 call (Nov 5), on its $325M acquisition: "Level AI reported stand-alone annual recurring revenue of approximately $20 million as of October 31 and grew approximately 50% year-over-year." revenue_annualized_usd 20,000,000 (dated Oct 31, stated) and growth_pct +50 over 365 days. The growth window is not explicitly dated, so I dated it to the Nov 5 call. No basis. Page read. https://www.tradingkey.com/news/transcripts/262293018-

## Seen but not recorded
- Anthropic $71B (Sept 30) and $65B (July 31): citybiz's table gives $71B, but the filing only says "up $10 billion from September 30". Recording $71B would mean deriving it. $65B is already recorded. citybiz's "approximately 25% in three months" is the article's own calculation, not a company-stated rate.
- Level AI: Five9 expects "a similar level" of ARR growth over the next 12 months. That is a forecast. "More than one billion customer interactions annually" is a usage count.
- Five9: AI revenue +91% YoY and roughly $45M of AI revenue in Q3 (quarterly recognized revenue). Five9 is a CCaaS company that is not an AI company. The AI line is a quarterly revenue blend, not a distinct app or an ARR, so it is not recorded.
- Gemini app 1B monthly users (Motley Fool, Nov 14): the same figure is already recorded (Aug 11 and Oct 28 rows).
- Muse (Meta) 10M users in the US and Canada (PaidFreeDroid, Nov 13): already recorded (Oct 21). The "10M U.S. App Store downloads in 40 days" figure is a Sensor Tower estimate via TechCrunch, so it is third-party.
- Retell, Rox, Kindo, Databento, LangChain, Mercor, Mirendil, Imprint, Nebuly: handled in the Nov 12–14 files.
- Future Secure AI (AFR via CX Network, Nov 13): ARR "expected to approach US$100 million by end of 2026". This is a forecast and secondhand. Wonderful (Globes): "market estimates" of ARR approaching $100M. Unnamed source and a ceiling.
- Cohere $240M ARR end-2025 (investor memo via CNBC/BetaKit): secondhand and old. Cohere's 165% YoY ARR growth is already recorded (Nov 3 file).
- AgileMD "on pace for about $7M ARR by year-end" (AI Weekly, Nov 14): a forecast.
- Hyphenate "explosive revenue growth" (CPA Trendlines, Nov 14): no rate given. Griffin's recurring revenue up 3.2x YoY: a fintech bank, not an AI app, and reported by FinTech Global.
- Digi Power X AI revenue +400% QoQ (Nov 13): quarterly revenue of an AI data-center operator, not an AI application. Nebius ARR $6.4B: AI infrastructure, not an app, and secondhand (Substack review).
- Kimi ARR $100M→$200M (Mar–Apr 2026): older than 90 days and from a directory page.
- Opera iOS MAU +20% YoY, Tubi 110M MAU, PlayStation 127M MAU, Pluto TV 75M MAU: not AI products.
- Nominal ($240M), Path Robotics ($300M), Torus ($500M), Mavria ($130M): no ARR, users or growth figure found.

## Trends
- The week's one large new company-stated level came from a filing: Anthropic's S-1/A put run-rate at about $81B on Oct 31, up $10B in a month.
- Acquirers' earnings calls are disclosing target ARR (Five9 on Level AI) that the private startups never published.
- Nov 13–15 funding news (Crunchbase top 10) again led with valuations; nearly all ARR claims in the window were already captured on Nov 12–14.
