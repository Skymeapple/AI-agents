# Reported growth notes, 2026-11-19

Collector: agent:web-research. This pass covers Nov 18–19, plus Perplexity's Nov 16 ARR disclosure, which the Nov 18 pass missed.

Tools:
- Tavily search and Tavily extract worked (keyed).
- WebSearch worked in standard and extended modes. It confirmed the Humyn Labs and Perplexity leads, but its Tencent and Kuaishou answers were stale or wrong.
- WebFetch was not tried.

Extract read these pages: the PR Newswire copy of Tencent's Q3 2026 release, the Yahoo copy of the same release (only partly returned), the Investing.com Q3 call transcript, 36Kr's call write-up, Srinivas's X post, Jawlah's Perplexity article, the Humyn Labs release, Business Standard's Humyn article, the AI Funding Tracker roundup for Nov 18, and the Globes Gong article.

Every recorded figure comes from a page I read, not from a snippet. The file has 5 observations for 3 apps and passes validation. I checked every candidate against observations.jsonl:
- WorkBuddy and Humyn Labs have no earlier rows.
- Perplexity's only earlier revenue row is the August $750M row (The Information, relayed). There is no $1B row.

## Recorded
- **Tencent WorkBuddy, users.** Q3 release (Nov 18): "over 10 million DAU". Recorded as users 10M, basis dau, dated Nov 18. Page read. https://www.prnewswire.com/apac/news-releases/tencent-announces-2026-third-quarter-results-302943839.html
- **Tencent WorkBuddy, ARR.** Q3 call (Nov 18): "annualized gross revenue over the last seven days has exceeded $500 million". Recorded as revenue_annualized_usd 500M, basis gross, dated Nov 18.
  - The Investing.com transcript spells the product "WorkMate". 36Kr quotes Martin Lau naming WorkBuddy.
  - Per 36Kr, the seven-day window follows the Hy4 launch and the end of a free offer, and gross revenue includes pass-through to model vendors.
  - Page read. https://www.investing.com/news/transcripts/earnings-call-transcript-tencent-q3-2026-profit-slips-as-ai-spending-surges-93CH-5036117
- **Tencent WorkBuddy, revenue growth.** Same call: "revenue for the third quarter was over 5x the previous quarter". Recorded as growth_pct +400 over 91 days (QoQ), dated to the Nov 18 call because the source names only the quarter. This is quarterly recognized revenue, following the precedent of the Kling Q2 row. Page read.
- **Perplexity, ARR.** Aravind Srinivas on X (Nov 17): "Yesterday, I revealed that Perplexity crossed $1B ARR." Recorded as revenue_annualized_usd 1B with no basis, dated Nov 16 (the day he spoke at Semafor's event). Post read. https://x.com/AravSrinivas/status/2122734422445445629
- **Humyn Labs, ARR.** Series A release (Nov 18): "surpassed USD 50 million in annualised revenue within six months of operations". Recorded as revenue_annualized_usd 50M with no basis, dated Nov 18. Mercor and micro1 are already tracked, so a training-data vendor is in scope.
  - Caution: TechCrunch quotes the co-founder as saying it had $10M of revenue from April to October plus $20M in signed contracts.
  - Page read. https://humynlabs.ai/blog/humyn-labs-raises-usd-47m-in-series-a

## Tencent Q3 2026 (Nov 18): not recorded
- **Yuanbao:** no user figure is stated as a separate line in the release or on the call. Lau said Yuanbao is "not focused on growing user base, but focused on growing its capability", and that Tencent "reduced the spending on Yuanbao quite significantly". 36Kr's "37.33 million in September" MAU is a third-party figure (QuestMobile-type), so it is not recorded.
- **Xiaowei** (Weixin's built-in AI assistant):
  - On the call: "Since testing started in June, we have consistently expanded the user testing pool for Xiaowei, and now has achieved 30 million DAU". Lau also said: "Our DAU in terms of interaction with Xiaowei is now 30 million". Page read.
  - It is not recorded because Xiaowei has no URL of its own to serve as the identity key. Using weixin.qq.com would merge it with WeChat. The developers.weixin.qq.com "腾讯小微" page is the 2017 hardware voice assistant, a different product.
  - A reviewer could add it if a canonical URL is chosen. The values would be users 30,000,000, basis dau, dated Nov 18, sourced to the transcript.
- **Model as a Service:** revenue "more than tripled quarter-on-quarter". **GPU rental:** revenue "more than 30%" QoQ. Both are infrastructure lines, so neither is recorded.
- **WeChat:** 1.4B combined MAU is a whole-company figure from a company that is not an AI company.
- **Green Bubble short drama:** 143M MAU, from QuestMobile, and not an AI product.

## Seen but not recorded
- **Anthropic IPO** (priced Nov 18 at $83): no new run-rate figure. CNBC repeats the S-1/A's ~$81B at Oct 31, which is already recorded. "Nine-month revenue up roughly 1,200%" is CNBC's comparison of two YTD levels.
- **Palo Alto Prisma AIRS** (Nov 18 call): "nearly $170 million in ARR". The value is an approximate ceiling, and the product is a line of a security company.
- **Kuaishou/Kling Q3:** results were scheduled for Nov 19 (Beijing time), but no release had surfaced when I collected. A later run should check Kling revenue growth and users. The Q2 Kling figures (>RMB850M, >200% YoY) are already recorded.
- **Gong:** "ARR exceeded $500 million at the end of April, 55% annual growth". This is a May 12, 2026 Globes article, outside the window.
- **Linq** ($90M Series B, Nov 18): "more than a billion messages", adding 150M a month. These are usage counts.
- **Payhawk** ($280M Series C): "77 percent of customers use its AI agents daily". This is a share, and the company is a finance platform rather than an AI company.
- **LangChain** ($300M Series C): 7,000 LangSmith customers. This is a customer count.
- **StepSecurity** (Series A): "ARR grew 5x in 2024 and again in 2025; in 2026 already crossed 5x". The windows are calendar-year and year-to-date, and the company is in security.
- **Serri AI:** "doubling its ARR within three months". This compares two levels ($1M to $2M) over a window that is not exactly a quarter. "July–September quarter grew 105%" is quarterly revenue from a CXOToday relay that I did not read.
- **Other Nov 18 rounds without metrics:** BrightAI ($97M), Botsync, VeroSecurity, CloudBackend, Soket AI and Perry AI had no ARR, user or growth figure. Isomorphic Labs ($5B) and Genesis AI ($500M) also had none.

## Trends
- Tencent now discloses AI app metrics product by product. WorkBuddy got DAU, ARR and QoQ revenue in the same quarter, and Xiaowei got its first DAU. Yuanbao is explicitly deprioritized and has no disclosure.
- Run rates calculated over a short window are spreading. WorkBuddy's $500M ARR is annualized from 7 days of gross revenue, and Humyn's $50M run rate comes after 6 months of operation. Treat both as peak-window run rates.
- Perplexity announced its $1B ARR on stage and on X, not in a release, so the CEO's post is the primary source.


## Review by the daily run (2026-11-19)

- **WorkBuddy $500M annualized gross revenue dropped.** The Investing.com call transcript names the product "WorkMate"; without the release or an official transcript confirming it is WorkBuddy, the figure cannot be attributed. The 10M DAU (release) and Q3 revenue 5x QoQ (call) rows stay.
