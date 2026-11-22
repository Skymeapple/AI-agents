# Reported growth notes, 2026-11-20

Collector: agent:web-research. This pass covers Nov 17–20 news, follows up on yesterday's Tencent and Kuaishou questions, and picks up two in-window items that earlier passes missed (Replit on Nov 4, Baidu Kuku on Nov 12).

Tools:
- Tavily search and Tavily extract worked (keyed). Every recorded figure was read on its page with extract, not from a snippet.
- WebSearch (standard) ran but returned only stale 2025 items for this week's query.
- Extract failed on kuku.baidu.com (a guessed URL). The Yahoo/Business Wire page came back with heavy navigation boilerplate, but the full release text was included.

Every candidate was grepped against observations.jsonl by name and host. The file has 7 observations for 6 apps and passes validation. None of the apps had an earlier row for these metrics.

## Follow-ups from yesterday
- **Tencent WorkBuddy $500M annualized gross revenue: confirmed and recorded.** 36Kr's call write-up names WorkBuddy for the figure: 刘炽平还说——"WorkBuddy过去七天的年化总收入已超过5亿美元，是七年前推出的腾讯会议的2倍以上。" It also notes "英文原话用的是gross revenue". So "WorkMate" in the Investing.com transcript is a transcription error. Recorded as revenue_annualized_usd 500M, basis gross, dated Nov 18. https://www.36kr.com/p/4076046847710087
  - Pengpai (via Sohu) and Jiemian (via Sohu) also attribute the 7-day annualized >$500M to WorkBuddy. https://www.sohu.com/a/1103353612_260616
  - Caveat from 36Kr: the 7-day window sits on the Hy4 launch and the end of the free offer (Nov 15), and gross revenue includes pass-through to third-party model vendors.
- **Kuaishou / Kling Q3 2026 (Nov 19): nothing recordable.**
  - The release says only that other-services revenue rose 1.5% to RMB6.0B, "primarily attributable to the increase in games and Kling AI revenue". https://www.prnewswire.com/news-releases/kuaishou-technology-announces-third-quarter-2026-unaudited-financial-results-302945173.html (search snippet; not opened)
  - On the call (Yahoo transcript, read): "Following Kling 4.0's launch, Kling AI's gross billings increased ... User scale and ARPU both show solid growth momentum." No number was given. https://finance.yahoo.com/quote/KUASF/earnings/KUASF-Q3-2026-earnings_call-736381.html
  - 36Kr EU confirms Kuaishou "did not disclose the quarterly revenue or ARR data of Kling AI". https://eu.36kr.com/en/p/4076736740676484
  - GuruFocus: "revenue from Kuaishou Vanchin up nearly 90% quarter-over-quarter". This is the enterprise MaaS platform, and "nearly" makes it an approximate value, so it is not recorded. https://www.gurufocus.com/news/9210741/

## Recorded
- **Doubao Work (ByteDance), users.** 8M DAU, Nov 18: "豆包官方宣布，上线仅三个月的豆包工作日活跃用户数（DAU）突破800万，企业客户年度经常性收入（ARR）突破20亿元。" http://k.sina.com.cn/article_1496814565_593793e5020028vqi.html
  - The RMB 2B ARR is not recorded: the metric is in USD, and converting would be an estimate. A reviewer may want an RMB-denominated metric.
  - Identity URL doubao.com/work was confirmed by reading the page.
- **GC AI, ARR and growth.** Business Wire, Nov 18: "passed $40 million in annual recurring revenue, growing 4x year over year." Recorded as ARR 40M and growth_pct +300 over 365 days. https://finance.yahoo.com/technology/ai/articles/gc-ai-hits-40-million-151500505.html
- **Replit, users.** Nov 4 PR Newswire boilerplate: "more than 60 million users worldwide". The users are cumulative, so no basis is recorded. https://www.prnewswire.com/news-releases/replit-opens-first-asia-office-in-tokyo-expanding-global-footprint-302929049.html
- **Daydream, users.** TechCrunch, Nov 17: the company "says more than 2 million shoppers have used Daydream ... since its June 2025 launch". Cumulative, so no basis. https://techcrunch.com/2026/11/17/daydreams-new-ai-agent-acts-as-your-personal-stylist-and-can-shop-for-you-too
- **Baidu KOOKO / 库库, users.** Baidu World, Nov 12: "库库全球月活用户已达5000万". Recorded as 50M MAU. https://wap.eastmoney.com/a/202611123929865868.html
  - Identity needs review. kooko.ai is the overseas site, while Baidu's English release calls the whole product KOOKO. If a reviewer prefers a China URL, re-key the row.

## Seen but not recorded
- **Meta Muse:** "10.24M daily active users on November 13". This is a Similarweb estimate relayed on X, not company-reported. https://x.com/AGTPinsights/status/2122680044489248836
- **Kuku/KOOKO, other figures:**
  - KOOKO overseas "累计服务1200万海外用户" is cumulative and a subset of the Kuku family.
  - Sept 21 "全球AI办公月活超4000万" has a possibly different scope. https://www.sohu.com/a/1079073932_120988576
- **Baidu DuMate (百度搭子):** "977.0万的月活 ... 1567.5%的季度环比" is from a NetEase op-ed (third-party style, not attributed to Baidu). It is not recorded; DuMate's 30x QoQ from the call is already in observations. https://www.163.com/dy/article/LC1LTCRB0511D2LM.html
- **Netlify:** "Annual Recurring Revenue Approaches $130 Million". The value is approximate and the company is a hosting platform. https://www.prnewswire.com/news-releases/netlify-unveils-new-ai-app-building-experience-as-annual-recurring-revenue-approaches-130-million-302943011.html
- **Honeycomb:** gives growth claims without a number.
- **Linq:** 1B messages is a usage count.
- **StepSecurity:** calendar-year multiples; already noted yesterday.
- **Applied Compute:** the >$100M run rate was re-stated at the Palo Alto acquisition (Nov 18). It is already recorded (Nov 2).
- **Model ML:** ARR 12x / DAU 26x is already recorded.
- **Palo Alto Prisma AIRS:** "nearly $170M" was already rejected yesterday.
- **LATO:** ">$1 million" ARR roughly 100 days after its first study. The value is small and the source is CREDX secondary coverage, which I did not open. https://credx.media/articles/lato-funding-4-7m-seed-ai-due-diligence-private-equity
- **Anthropic:** "$17 billion in third-quarter 2026 revenue" (TechCrunch) is quarterly revenue, not a run rate. The ~$81B run rate at Oct 31 is already recorded.
- **Tencent Xiaowei:** 30M DAU, still no identity URL (see yesterday).

## Trends
- **China's office-agent race went metric-first on the same night (Nov 18).** Doubao Work published DAU 8M and enterprise ARR RMB 2B three minutes before Tencent's call. Tencent answered with WorkBuddy's 10M DAU and $500M 7-day annualized gross revenue. Baidu had published Kuku's 50M MAU at Baidu World a week earlier.
  - Yunyingpai on the timing: https://www.yunyingpai.com/news/1070892.html
  - BT Finance on the two disclosures and the 7-day annualization: https://businesstimescn.com/articles/633659.html
- **Every office agent now aggregates models.** WorkBuddy, Doubao Work, Kuku and Qianwen Office all route to third-party models (Kimi, GLM, DeepSeek, MiniMax) and show per-model quota multipliers. Gross revenue therefore includes model pass-through, and Tencent calls WorkBuddy "the largest customer of almost all open-weight models in China". https://www.163.com/dy/article/LC3STKR70530UH99.html
- **Disclosure keeps shifting to short-window run rates and cumulative counts.** Examples are WorkBuddy's 7-day annualization, Replit's and Daydream's lifetime user counts, and KOOKO's "累计" users. Treat these as ceilings on momentum, not steady-state levels.
- **Kling disclosure regressed.** Kuaishou gave Kling figures in Q2 (revenue >RMB850M, >200% YoY) but in Q3 said only "gross billings increased". This came in a quarter when adjusted net profit fell 84% on Kling spending. https://www.investing.com/news/company-news/kuaishou-q3-2026-slides-profit-slides-84-as-ai-spending-surges-93CH-5038720
- **Revenue growth across AI businesses is slowing.** Stripe's AI Index shows same-business AI revenue growing at an ~80% annualized rate over the 13 weeks to Nov 8, down from 140% a few months earlier. This is an aggregate, not app-level, so it is not recorded. https://www.edwardconard.com/macro-roundup/introducing-the-stripe-ai-index


## Review by the daily run (2026-11-20)

- **Doubao Work 8M DAU dropped.** Its identity URL (doubao.com/work) canonicalises to doubao.com, which would merge it with the Doubao chatbot, the same reason Tencent Xiaowei was left out on Nov 19. The figure and source stay above for when the radar can key a product path.
