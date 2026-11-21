# Reported growth notes, 2026-10-31

Collector: agent:web-research. Tools: WebSearch worked, though many date-scoped queries came back with older or undated items. WebFetch got ENOTFOUND for hervecuviliez.substack.com and gradientvc.substack.com, which are outside the session's allowed network domains, so no further fetches were tried. Tavily was not used because it is over quota. No page was read; every figure comes from search-result summaries. The file has 1 observation. Every candidate was checked against observations.jsonl, and most of this week's announcements are already recorded.

## Recorded
- MiniMax: ARR "above $800 million" for August, from the mid-year call on Aug 26. It is dated to the call date because no period end date is given. https://www.edgen.tech/news/post/minimax-w-revenue-jumps-283-as-arr-tops-800-million ; https://www.caixinglobal.com/2026-08-27/minimaxs-revenue-jumps-280-as-business-customers-drive-growth-102478492.html

## Oct 29-31: seen but not recorded
- Already in observations.jsonl:
  - Serval (ARR doubling every quarter, Oct 29). https://www.morningstar.com/news/business-wire/20261029114967/serval-raises-150m-series-c-at-3b-valuation-led-by-sequoia
  - Gemini app 1B MAU (Oct 28 call). https://ppc.land/gemini-app-gains-350m-users-in-a-year-to-pass-1bn-alphabet-says/
  - Microsoft Copilot daily users +~50% QoQ and Foundry revenue >3x (Oct 28 call). https://finance.yahoo.com/markets/stocks/live/microsoft-earnings-live-microsoft-beats-q1-expectations-on-top-and-bottom-line-on-ai-growth-204510452.html
  - Simplismart 14x run-rate (Oct 27). https://startuptalky.com/news/simplismart-raises-14-mn-series-b-nvidia-dallas-venture-capital/
  - Daytona 25x run-rate. https://aiweekly.co/alerts/daytona-raises-60m-series-b-at-1b-for-ai-agent-cpu-cloud-25x-revenue-jump-since
  - Anthropic S-1 (Oct 26). https://finance.yahoo.com/technology/live/anthropic-ipo-live-updates-frontier-ai-lab-discloses-revenue-and-risks-in-highly-anticipated-s-1-filing-141702607.html
- Riverside (podcast/video recording): a single Dealroom snippet says the company "said on 29 October 2026" it passed $100M ARR. No second source confirmed it, and Riverside is a recording tool with AI features rather than an AI app. https://dealroom.co/news/168302-riverside-founded-by-dutch-born-keyson-brothers-crosses-100m-arr/
- Peec AI (Oct 30 GlobeNewswire, SOC 2): "more than 3,500 brands and agencies" use it. That counts customer organisations, not users, so it was not recorded as `users`. https://www.globenewswire.com/news-release/2026/10/30/3392635/0/en/peec-ai-completes-soc-2-type-ii-examination-making-its-ai-search-visibility-platform-fully-enterprise-ready.html
- Reflex ($14M Series A, revenue grew more than 10x): an aggregator credited SiliconANGLE "Oct 29", but the matching SiliconANGLE item appears to be from Oct 29, 2025, a year outside the window, and the date could not be confirmed. https://www.datagrom.com/ai-news/reflex-raises-14m-for-custom-enterprise-app-platform-71d37f2c
- Earnings that gave no metric for an AI product:
  - Pinterest Q3 (Oct 29): 661M MAU is the whole platform, with no Pinterest Assistant user count. https://www.pymnts.com/earnings/2026/pinterests-ai-push-pays-off-in-record-users-and-revenue/
  - Roblox Q3 (Oct 29): the AI figure is a creator survey ("97% ... accelerated"). https://www.sec.gov/Archives/edgar/data/0001315098/000162828026068235/ex991-q32026earningsshar.htm
  - Reddit Q3 (Oct 29): no Reddit Answers figure. https://www.sec.gov/Archives/edgar/data/0001713445/000171344526000114/earningspressreleaseq326.htm
  - T-Mobile (Oct 29): T-Life is not an AI app. https://www.fool.com/earnings/call-transcripts/2026/10/29/t-mobile-tmus-q3-2026-earnings-call-transcript/
- Other Oct 29-30 rounds stated no ARR, users or growth: Solink (>$100M), AvalonX, Collective, Wabi. Snap, Apple and Atlassian have not reported Q3 yet.

## Within 90 days: seen but not recorded
- Town (Oct 19): "paying customers grew fourfold since July" counts customers, not users, and the "seven figures" ARR has no number. https://thenextweb.com/news/town-108m-series-b-1b-valuation-ai-assistants
- Harvey (Oct 1 Q3 recap): $400M+ ARR. The same level is already recorded for Sep 9, and "added >$100M in the quarter" is a dollar increment, not a stated rate. https://okara.ai/blog/harvey-ai-revenue-growth
- Muse: 10.9M downloads is an Appfigures third-party estimate, not company-reported. The 10M accounts figure is already recorded. https://techcrunch.com/2026/10/20/metas-muse-hits-10m-u-s-downloads-in-40-days-a-milestone-reached-by-only-11-other-apps/
- Tencent WorkBuddy: the 85.34M figure is September "visits" from Analysys, a third party. The visit type is unclear, and the 20M MAU figure is undated. https://fourweekmba.com/ai-tencents-workbuddy-tops-chinas-office-agents-at-85m-visits/
- MiniMax ">400% within six months": an outlet computed this from two levels ($150M in Feb to $800M in Aug), so it is a derived rate. https://www.edgen.tech/news/post/minimax-w-revenue-jumps-283-as-arr-tops-800-million
- Replit ($1B on an arr.club ranking) is described as "nearing" a target, not a stated level. Lovable's $600M is undated. Cursor's $4B is from June, outside the window. https://www.arr.club/rankings/2026/10 ; https://ecosistemastartup.com/lovable-cruza-us600m-arr-y-consolida-el-vibe-coding/

## Trends
- The late-October news flow is dominated by earnings and IPO filings (Alphabet, Microsoft, Amazon, Anthropic S-1) rather than startup ARR announcements. Startups raising this week mostly disclose growth multiples off an undisclosed base, or nothing at all (Serval, Solink, Daytona). https://quasa.io/stories/daytona-hits-1b-but-its-25-growth-starts-from-an-undisclosed-base
- Consumer platforms (Pinterest, Reddit, Roblox) credit AI for engagement but give no user counts for their AI features. https://www.pymnts.com/earnings/2026/pinterests-ai-push-pays-off-in-record-users-and-revenue/
- Chinese model makers report ARR led by APIs and enterprise customers: about 80% of MiniMax's comes from business customers. https://www.caixinglobal.com/2026-08-27/minimaxs-revenue-jumps-280-as-business-customers-drive-growth-102478492.html
- Startups increasingly state customer or team multiples instead of ARR (Town 4x paying customers, Peec AI 3,500+ brands). https://www.arr.club/town/town-quadrupled-paying-customers-raised-108m-series-b
