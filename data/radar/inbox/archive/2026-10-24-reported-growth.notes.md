# Reported growth: qualitative trends (collected 2026-10-24)

All figures come from WebSearch result summaries. WebFetch failed with ENOTFOUND on techcrunch.com, and Tavily returned 432 (over its usage limit). No page was opened directly.

This file has 11 observations, which is below the 15-observation target. Most queries returned one of three things, all left out:
- figures dated before 2026-07-26;
- Sacra, Latka or other third-party estimates;
- figures already recorded in the 10-19 to 10-23 files, such as Podium, Legora, OpenArt, Muse, Higgsfield, Runway and Anthropic.

Also left out:
- AI data vendors: Proximal, micro1 and Nyne.
- Non-AI companies: 1Password, Ramp and Guardio.
- RingCentral's "~15% of ARR from customers using paid AI products". This is a customer-cohort figure, not AI revenue.

## Trends

- **Agent compute and infrastructure is compounding fast.**
  - Runpod reports more than $325M ARR, up from $120M in January 2026.
  - Its developer base is more than 1.2M, up from about 495k a year earlier.
  - This follows Modal's $1B run rate, recorded on 10-20.
  - Source: https://www.runpod.io/press/runpod-325m-arr-new-cfo

- **Enterprise AI workspaces and "AI coworker" agents reach $50M+ within months.**
  - Viktor, an agent inside Slack and Teams, went from $20M in June to a reported $50M run rate on Sep 18.
  - Langdock passed €50M (about $55M) ARR in September, roughly three months after $40M.
  - Sources:
    - https://www.arr.club/getviktor/viktor-arr-hit-50m-in-just-7-months
    - https://app.dealroom.co/news/note/langdock-hits-50m-arr-55m

- **Vibe-coding builders keep doubling.**
  - Base44 (owned by Wix) passed $200M ARR five months after $100M. It credits its in-house Base1 model.
  - Source: https://calcalistech.com/ctechnews/article/syw900pvdzg

- **Big incumbents now disclose AI seat and user counts rather than revenue.**
  - Microsoft 365 Copilot has more than 30M paid seats, and net adds more than doubled quarter over quarter.
  - GitHub Copilot has 50M users. The active-user window was not stated.
  - Sources:
    - https://www.uctoday.com/unified-communications/microsoft-365-copilot-passes-30-million-paid-seats-as-cloud-and-ai-growth-power-record-quarter
    - https://www.fool.com/earnings/call-transcripts/2026/08/07/microsoft-msft-q4-2026-earnings-call-transcript/

- **Niche consumer AI (privacy chat, companions) is monetising.**
  - Venice AI went from more than $70M to more than $100M ARR in about a month.
  - Tolan reports 250k+ active users and ARR in the "double-digit millions".
  - Sources:
    - https://cryptobriefing.com/venice-ai-100m-annualized-revenue/
    - https://runtimewire.com/article/tolan-raises-30m-ai-companion-daily-assistant

- **Run-rate definitions are under scrutiny.**
  - OpenAI's widely reported ~$70B run rate was a gross figure. Net of partner sales it is about $50B; that figure is already recorded.
  - Proximal's run rate jumped from $200M to $660M after it switched from quarter×4 to month×12.
  - Treat headline ARR milestones with care.
  - Sources:
    - https://www.axios.com/2026/10/08/openai-50-billion-arr-anthropic-revenue-accounting
    - https://www.arr.club/proximal/proximal-hit-55m-monthly-revenue-660m-run-rate

- **AI security tools cross $100M.**
  - Horizon3.ai (NodeZero autonomous pentesting) passed $100M ARR on Oct 20.
  - Snippets disagree on its growth rate: 120%+ versus 102% YoY.
  - Source: https://www.morningstar.com/news/business-wire/20261020386063/horizon3-surpasses-100-million-arr-marking-a-key-milestone-in-the-shift-to-continuous-production-safe-autonomous-security

## Source caveats

- Viktor ($50M) and Peec ($15M) come from ARR Club pages that relay founder announcements. No second outlet confirmed them.
- The Langdock USD value is the outlet's own conversion of €50M.
- Base44 and Langdock were stated only by month, so `observed_on` is set to the last day of the month.
