# Reported growth notes, 2026-11-21

Collector: agent:web-research. This pass covers Nov 17–21 news. Most of the week's big disclosures (WorkBuddy, Doubao Work, GC AI, Model ML, Applied Compute, Perplexity $1B) were recorded on earlier days. Today's pass therefore found three new readings.

Tools:
- Tavily search and Tavily extract worked (keyed). Every recorded figure was read on its page with extract.
- WebSearch (standard) ran but returned only items that were already recorded (GC AI, Perplexity) or older.
- Extract of the TNW Magnific article returned only cookie and navigation boilerplate, so Media-Marketing.com and Dealroom were used instead. The CXOtoday Serri page came back without a publication date.

Every candidate was grepped against observations.jsonl by name and host. The file has 3 observations for 3 apps and passes validation.

## Recorded
- **Magnific (formerly Freepik), ARR above $300M, Nov 17.** "Magnific has surpassed $300 million in annual recurring revenue, or ARR, globally." The company says this makes it the fourth-largest European generative AI company by ARR, behind Lovable, ElevenLabs and Mistral. https://www.media-marketing.com/en/news/magnific-invests-10-million-in-expansion-among-asian-agencies-studios-and-brands
  - Dealroom: the figure is "up from $230M when it rebranded in April". The April figure has no day, so it is not recorded. https://dealroom.co/news/174618-magnific-hits-300m-arr-mulls-first-outside-funding-since-2010
  - TNW (Nov 19) says "1M subscribers". It is not recorded because the TNW body could not be read, and the magnific.com homepage line ("More than 1 million subscribers") is undated.
- **Pulley (withpulley.com), ARR growth +400% YoY, Nov 18.** Business Wire: "Following 5x year-over-year growth in annual recurring revenue". No ARR level was given. https://finance.yahoo.com/real-estate/articles/pulley-raises-64m-accelerate-every-223000419.html
  - This is a different company from Pulley the cap-table startup (pulley.com).
  - The business is service-heavy: 75+ licensed experts work alongside AI agents.
- **DeepSeek Harness, 5.48M MAU as of Nov 4.** Caijing: "截至11月4日，DeepSeek Harness日活跃用户已达154万，月活跃用户达到548万". The count covers official-API users only. The DAU was already recorded; the MAU is new. https://m.sohu.com/a/1099299267_115571
  - Not recorded: "+545% DAU vs. the Sep 29 desktop launch". Turning it into a growth_pct would mean deriving a period from two dates. The underlying DAU levels are already in observations.

## Seen but not recorded
- **Archer (hackajob), Nov 20, $10M round.** "annual recurring revenue grew fivefold in 2026" is a calendar-year multiple with an unclear window, the same reason StepSecurity was rejected. "More than 700,000 candidates signed up ... in the 12 months" is a count of sign-ups over a period, which no metric in the vocabulary covers. https://www.unite.ai/archer-raises-10m-to-expand-ai-tools-for-candidates-and-recruiters
- **Serri AI:** ">$2 million ARR, doubling ... within three months". It is not recorded for three reasons:
  - No page states a publication date: CXOtoday, Dealroom and ARR Club were all read, and none is dated.
  - It is messaging infrastructure (WhatsApp campaigns).
  - "75 million users monthly" is the reach of its brand customers, not users of Serri.

  https://cxotoday.com/media-coverage/serri-ai-crosses-2-million-arr-doubles-revenue-in-three-months-on-just-200k-funding
- **Lovable, two-year anniversary (Nov 18).** It reported "70 million projects", 1.2M new projects a week and "over 1 billion monthly visits" to Lovable-built apps. Projects is not a vocabulary metric, and the visits go to apps built on Lovable, not to lovable.dev. https://lovable.dev/blog/two-years-of-lovable and https://futurumgroup.com/insights/lovable-at-two-70m-projects-signal-ai-native-development-goes-mainstream
- **Perplexity:** "$1 billion ARR, more than quadrupling its revenue pace since January" (Semafor stage, Nov 20). The $1B was already recorded (Nov 16), and "more than quadrupling" gives no exact value. https://www.threads.com/@evolving.ai/post/Df-KVBTgLzd (social; not opened)
- **Litera Lito:** "monthly active users grew 300% quarter over quarter". Lito lives on litera.com, the parent company's domain, so it fails the identity rule. https://www.lawnext.com/2026/11/litera-names-greg-ingino-to-chief-technology-and-ai-officer.html (snippet)
- **Clutch (Palo Alto target):** "approaching $10 million in ARR" is an approximate value. https://americanbazaaronline.com/2026/11/19/palo-alto-networks-bets-4-5-billion-on-ai-startup-applied-compute-as-shares-jump-9-492677 (snippet)
- **Google Gemini:** Tasks rolled out to "Gemini's more than 1 billion monthly active users" (Inc., Nov 18). This repeats the 1B already recorded on Oct 28. https://www.inc.com/aaron-mok/google-just-made-geminis-personal-assistant-free-use-it-to-handle-these-everyday-tasks/91438908
- **Meta Muse:** "more than 10 million people in the US and Canada set up agents" in six weeks. This comes from a Bain article, not from Meta, and Muse sits on a Meta domain. https://www.bain.com/insights/ai-is-the-new-search-heres-how-retailers-win-the-answer
- **Microsoft Ignite (Nov 17):**
  - "Nearly 80% of Fortune 500 companies use Copilot Cowork."
  - "50 million agents managed in Agent 365."

  Neither is a user count, and both products sit on paths of microsoft.com. https://www.witivio.com/en/blog/microsoft-ignite-2026-announcements
- **Tencent Xiaowei (30M DAU) and Kuaishou/Kling:** both unchanged from yesterday. Xiaowei still has no identity URL, and Kling gave no Q3 number.
- **Alibaba Qwen app:** Alibaba did not disclose Qwen app users this quarter (NetEase: "本季财报里，阿里干脆没有披露千问的用户数据"). AI-related product revenue of RMB17.9B is segment-level. https://www.163.com/dy/article/LC66270V05118O92.html
- **Qianwen Office:** "用户数突破3000万" is from Sep 4, outside the 90-day window. https://timeline.sohu.com/news/cQXfhSIvGB
- **WorkBuddy:** "9月月活达到3235万" is a QuestMobile estimate, not company-reported. Eguan's 85.34M September visits are also third-party. https://timeline.sohu.com/news/cQXfhSIvGB and https://post.m.smzdm.com/p/a4q43qnl

## Trends
- **Chinese commentators push back on 7-day annualization.** Yicai notes that "月度是最短的合理年化周期，季度更稳定" and that WorkBuddy's $500M works out to $9.615M of revenue for the 7 days. A Zhihu breakdown says that week overlapped Singles' Day discounts. Tencent shares fell about 6% the next day.
  - https://www.yicai.com/news/103446257.html
  - https://post.smzdm.com/p/aqrzd88p
- **Big platforms disclose office-agent metrics but not chatbot metrics.**
  - Tencent says Yuanbao "没有达到预期" and has cut spending on it.
  - Alibaba gave no Qwen app MAU.
  - ByteDance led with Doubao Work DAU and ARR, not Doubao chat MAU.

  Disclosure is moving from consumer chat reach to paid productivity agents. https://www.21jingji.com/article/20261120/herald/88547be20042b3748359b03010d92ab7.html and https://www.163.com/dy/article/LC66270V05118O92.html
- **Bootstrapped and capital-light companies are using ARR milestones as their announcement.** Magnific ($300M, no outside round since 2010, EBITDA-positive) and Serri ($2M on $200K raised) both headline profitability alongside ARR. Magnific says most revenue goes back into compute ("We already spend hundreds of millions of dollars generating what our users ask for"). https://dealroom.co/news/174618-magnific-hits-300m-arr-mulls-first-outside-funding-since-2010
- **Agent harnesses are being measured by active users and tokens.** DeepSeek Harness (1.54M DAU, 5.48M MAU, ~92T tokens/day) is now benchmarked in the press against Claude Code (reported ~14M MAU in October, third-party) and Qianwen Office. Caijing estimates its API run-rate from token volume; that is an estimate, not company-reported, so it is not recorded. https://m.sohu.com/a/1099299267_115571
- **Desktop computer-use is now a standard office-agent feature.** WorkBuddy (Nov 20) and Gemini for macOS (Nov 19) both shipped app-control agents in the same week. https://news.mydrivers.com/1/1167/1167440.htm and https://www.geekpark.net/news/374235
