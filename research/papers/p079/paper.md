# The Language Tax in Years: Decomposing the AI Affordability Gap of Developing Countries into Income, Tokenization and Price Components

**Target venue:** ACM Conference on Fairness, Accountability, and Transparency (FAccT 2027)

---

## Abstract

Commercial language models are priced, rate-limited and context-limited per *token*. The same content written in many languages of the Global South takes more tokens than it does in English. Earlier work measured this "tokenization premium". It has not been placed in the same units as the two other forces that set whether a developing country can afford AI: per-capita income and the fast fall of inference prices. We build a minimal affordability model and use it to express all three forces in one currency: log-cost units, which convert to *years of price decline*. Under the model's assumptions, three results follow. (i) A uniform price decline never closes the relative affordability gap between countries. It only turns that gap into a fixed time lag. (ii) The lag splits additively into an income term and a language term. The language term's share of the gap does not depend on the rate of price decline. (iii) When the affordability rule is a threshold, small premiums can price out large populations. We then measure the premium for 380 UDHR translations under 11 production tokenizers released between 2020 and 2025. We combine these measurements with World Bank 2025 GNI per capita for 79 countries. Main findings: (1) In every tokenizer, poorer countries' predominant languages carry higher premiums. A tenfold income gap goes with a premium that is 1.2–1.4 times higher. The slope ranges from −0.10 to −0.16 per log-GNI, and it holds within Latin-script languages alone. (2) Newer tokenizers greatly reduced the premium, mostly for the worst-served languages. Population-weighted mean premium in low and lower-middle income countries fell from 4.34 (GPT-4's cl100k) to 1.82 (o200k) and 1.59 (Gemma 3). Yet once the initial premium is held fixed, four of five within-vendor upgrades favoured languages of richer countries. The coefficient is about −0.06 log-premium per log-GNI, with 95% CIs excluding zero. (3) For the median low or lower-middle income country, the language premium is 12.5% of the log affordability gap with the United States under o200k, down from 20.5% under cl100k. It reaches 25–38% for Lao, Amharic, Khmer and Burmese. (4) Thresholds amplify the premium. Take a monthly bundle that costs $2 in English and the 2%-of-income affordability rule. The bundle is unaffordable for 6.8% of the 7.3 billion people covered if they pay English token counts. That rises to 16.4% under o200k and 40.8% under cl100k. A lingua-franca sensitivity scenario cuts these figures to 8.4% and 11.4%. Choosing a model family with a better tokenizer for one's script is a free lever. For Lao and Amharic it is worth 0.9–1.2 log units, which is as much as the price-tier discounts vendors now offer in some markets. We give four falsifiable predictions with test protocols, including a closed-form prediction for how reasoning tokens dilute the cost premium.

## 中文摘要

商业大语言模型按"词元（token）"计费，也按词元限速、限定上下文。同一内容若用许多全球南方语言书写，所需词元多于英语。已有研究测量过这种"分词溢价"，但没有把它和另外两种力量放在同一单位里比较。这两种力量同样决定发展中国家能否负担得起AI：人均收入，以及推理价格的快速下降。本文建立一个最小可负担性模型，把三种力量统一为对数成本单位，并可换算为"价格下降的年数"。在模型假设下得到三个结论：（一）统一的价格下降永远不能缩小国家间的相对可负担性差距，只能把差距变成一个固定的时间滞后；（二）该滞后可加性地分解为收入项和语言项，语言项在差距中所占比例与价格下降速率无关；（三）当可负担性按阈值判定时，较小的溢价也可能把大量人口排除在外。随后，我们用2020–2025年发布的11个生产级分词器，测量了《世界人权宣言》380个译本的溢价，并结合79个国家2025年的世界银行人均国民总收入数据。主要发现如下。（1）在所有分词器下，较贫穷国家的主要语言溢价都更高：收入相差十倍，溢价约高1.2–1.4倍。斜率为每单位对数人均收入−0.10至−0.16，仅看拉丁字母语言时依然成立。（2）新一代分词器大幅降低了溢价，降幅主要集中在原本最差的语言上。低收入和中低收入国家的人口加权平均溢价，从GPT-4的cl100k下的4.34，降到o200k下的1.82和Gemma 3下的1.59。但在初始溢价相同的条件下，五次同厂商升级中有四次更有利于较富裕国家的语言，系数约为每单位对数收入−0.06，95%置信区间不含零。（3）对中位数的低收入或中低收入国家，语言溢价在o200k下占其与美国对数可负担性差距的12.5%，在cl100k下为20.5%；老挝语、阿姆哈拉语、高棉语和缅甸语高达25–38%。（4）阈值会放大溢价的影响。以一个英文成本为每月2美元的服务包和"不超过月收入2%"的标准为例：若按英文词元数付费，所覆盖的73亿人口中有6.8%负担不起；o200k下为16.4%，cl100k下为40.8%。在通用语敏感性情景下，这两个数字分别降到8.4%和11.4%。针对本国文字选用分词器更好的模型家族是一个零成本的杠杆：对老挝语和阿姆哈拉语，它值0.9–1.2个对数单位，与厂商目前在部分市场提供的价格档折扣相当。文末给出四个可证伪预测及其检验方案，其中包括推理词元如何稀释成本溢价的闭式预测。

---

## 1 Introduction

How developing countries avoid being left behind by AI is usually discussed in terms of compute, talent, data and electricity [Korinek & Stiglitz 2021]. These matter for *building* frontier systems. Most of the world's population will instead meet AI as *users* of models built elsewhere, paying per token through APIs or through subscriptions with token quotas. For users, a more immediate divide is affordability. Measured adoption already shows a divide: Microsoft's diffusion telemetry estimates that 24.7% of the working-age population in the Global North used generative AI in H2 2025, against 14.1% in the Global South, and the gap widened over the year [Microsoft AI Economy Institute 2026].

Three quantities set the affordability of a fixed amount of AI use in a country. First, *income*: the user's budget. Second, *price per token*: this has fallen between 9× and 900× per year at fixed capability, with a median of 50×/year [Epoch AI 2025]. Third, *tokens per unit of content*: this depends on the language and on the tokenizer. Prior work established that the third term varies greatly across languages [Petrov et al. 2023; Ahia et al. 2023; Lundin et al. 2026; Somide 2026], and that it correlates with lower development indicators [Ahia et al. 2023]. The policy literature on price and income has not quantified it, though, and the NLP literature on tokenization does not express it in units a finance ministry or a regional regulator would recognise. So we ask a narrow question:

> *How large is the tokenization premium relative to the income gap and to price dynamics? Is it shrinking, and for whom? Which levers that a developing country, a vendor or a model developer controls are comparable in size?*

**Approach.** We write a minimal affordability model. In it, all three forces act multiplicatively on the share of income spent on a fixed bundle of AI use. That makes them additive in logs and convertible to "years of price decline". We then measure the premium on a parallel corpus (380 UDHR translations) under 11 production tokenizers from five developers and four generations, and map it onto 79 countries with World Bank 2025 GNI per capita.

**Contributions.**

1. **A unit-consistent decomposition (theory).** Propositions 1–3 show the following. (a) Uniform multiplicative price declines keep cross-country burden ratios fixed and turn them into a constant *time lag*. (b) The lag splits additively into income and language terms, and the language term's share of the gap does not depend on the rate of price decline. (c) Under threshold affordability rules, which policymakers use for broadband (A4AI's "1 for 2" target [A4AI]), the language premium moves each country's affordability threshold. The population-level effect is therefore discontinuous and can be large even when log premiums are modest. Proposition 4 gives a closed form for how hidden reasoning tokens dilute the cost premium but not the context premium.
2. **A measurement across tokenizer generations (empirical).** We release token counts for 380 languages × 11 tokenizers (GPT-3-era p50k through GPT-4o/5-era o200k, Llama 3/4, Gemma 1/3, Mistral v1/Tekken, DeepSeek-V3, and the legacy Claude tokenizer). We validate them against an independent FLORES-based study and against translation variants. The validation shows where the corpus is reliable (levels; between-script contrasts) and where it is not (fine rankings within Latin script).
3. **An income-gradient and progressivity analysis (new empirical results).** The premium falls with national income under all 11 tokenizers, and the gradient survives a script control. Tokenizer upgrades mainly compress the worst premiums. Conditional on the initial premium, however, four of five upgrades favoured richer countries' languages. To our knowledge this is the first test of whether tokenizer progress is income-progressive.
4. **A lever comparison in common units (policy).** We compare the free lever of choosing a tokenizer, vendor price tiers (e.g., ChatGPT Go at ₹399 vs Plus at ₹1,999 per month [CNBC 2025; TechCrunch 2025]), and income growth, all in log-cost and price-years.
5. **Falsifiable predictions with test protocols** (§7) that need API billing access or usage logs. We did not have these and do not claim to have run such tests.

All numbers in this paper come either from code we ran (`code/`, outputs in `results/`) or from cited sources. Each is labelled.

## 2 Related Work

**Tokenization disparity.** Petrov et al. [2023] showed that the same FLORES-200 text can differ up to 15× in tokenized length across languages, and that this creates unfairness in cost, latency and context. Ahia et al. [2023] analysed OpenAI's API in 22 languages. They found speakers of many languages are "overcharged while obtaining poorer results", and that these speakers tend to live where the API is less affordable to begin with, which they call a "double unfairness". Rust et al. [2021] linked tokenizer quality to downstream monolingual performance. Lundin et al. [2026] showed that fertility predicts accuracy on AfriMMLU across 10 LLMs. Somide [2026] measured 20 African languages under 11 tokenizers on FLORES-200+. They report a median premium of 1.88× on o200k and a fall in the mean premium from 3.31× (cl100k) to 2.38× across tokenizers. Joshi et al. [2020] documented resource stratification across the world's languages. Our measurement overlaps with these studies. Our contribution is to put the premium into an economic decomposition next to income and price dynamics, to test income progressivity *across tokenizer generations* globally rather than only within Africa, and to model threshold effects.

**Price dynamics.** Epoch AI [2025] estimated that the price of reaching fixed benchmark performance fell between 9× and 900× per year (median 50×) from 2023 to 2025, with faster declines after January 2024. We use 9× and 50× as scenarios, and add a slow 2×/year scenario to represent users who want frontier-tier models, whose prices do not follow fixed-capability curves.

**Affordability standards and the digital divide.** For broadband, the Alliance for Affordable Internet's "1 for 2" target defines affordability as 1 GB of mobile data costing at most 2% of average monthly income. The UN Broadband Commission adopted it [A4AI]. ITU and A4AI report that consumers in low- and middle-income economies paid five to six times more, relative to income, for ICT services than consumers in high-income economies in 2021 [ITU 2022]. We borrow the 2% threshold as a transparent convention for AI bundles. We do not claim it is the right threshold for AI.

**AI and development.** Korinek and Stiglitz [2021] argue that labour-saving AI and winner-takes-all dynamics may reverse globalisation's gains for developing economies. Microsoft's diffusion reports measure a widening North–South usage gap and note that DeepSeek gained share in Africa partly through openness and price [Microsoft AI Economy Institute 2026]. We address one mechanism on the user side that such adoption gaps could reflect. We do not estimate its causal contribution to adoption (see §7, H4).

**Latent pivot languages.** Wendler et al. [2024] found that Llama-2's intermediate representations sit closer to English. This motivates our Proposition 4 assumption that hidden reasoning may be produced in a pivot language. We treat that assumption as testable, not established, for current reasoning models.

## 3 Model

### 3.1 Setup

Consider a bundle of AI use with fixed *content*: for example, a month of drafting, translation and question answering. Written in English, it takes *B* tokens. Let *p(t)* be the blended price per token at time *t* for a fixed capability tier. Country *c* has monthly income per capita *m_c* (GNI per capita / 12) and a predominant language *ℓ(c)*. Under tokenizer *T*, the premium of language ℓ is

  π_T(ℓ) = (tokens of text in ℓ under T) / (tokens of the parallel English text under T).

The **burden** is the share of monthly income spent on the bundle:

  b_c(t) = p(t) · B · π_c / m_c,  with π_c = π_T(ℓ(c)).

A country (or a representative user) can afford the bundle when b_c(t) ≤ θ. We use θ = 0.02, following A4AI's convention. Prices fall exponentially: p(t) = p_0 e^{−g t}. Incomes may grow: m_c(t) = m_c e^{γ_c t}.

### 3.2 Results

**Proposition 1 (relative gaps are invariant; time lags are additive).** Let all countries face the same price path and γ_c = γ_r for c and a reference country r. Then (a) the burden ratio b_c(t)/b_r(t) = (π_c/π_r)(m_r/m_c) does not depend on t. (b) Country c reaches the threshold Δ_c years after r, where

  Δ_c = [ ln(m_r/m_c) + ln(π_c/π_r) ] / (g + γ_r).

(c) The language share of the lag, s_c = ln(π_c/π_r) / [ln(m_r/m_c) + ln(π_c/π_r)], does not depend on g, γ, p_0, B or θ.

*Proof.* (a) p(t) and B cancel in the ratio. (b) Solving b_c(T_c) = θ gives T_c = ln(p_0 B π_c / θ m_c)/(g + γ_c). With γ_c = γ_r, taking the difference T_c − T_r removes p_0, B and θ. (c) follows from (b) because g + γ cancels in the ratio. ∎

Price declines, however fast, therefore cannot close the *relative* divide. They make every capability tier affordable everywhere *eventually*, and the divide becomes a constant delay. Faster declines shorten that delay in calendar time. They leave unchanged the share of it that comes from language.

**Proposition 2 (lever equivalence).** Any intervention that multiplies country c's effective price by 1/κ shortens the lag by ln κ / (g + γ). Examples include a regional price tier, a subsidy, a tokenizer that cuts π_c by a factor κ, or a choice of model family with a better tokenizer. A growth differential δ = γ_c − γ_r > 0 closes the gap at rate δ/(g+γ_r+δ). *Proof.* Substitute π_c/κ into Proposition 1(b). For differential growth, T_c = ln(p_0Bπ_c/θm_c)/(g+γ_r+δ), and subtracting T_r gives the expression. ∎

Every lever can therefore be compared in **log-cost units** (dividing by ln 9 ≈ 2.20 gives years at the Epoch lower-bound rate). Tokenizer improvements, discounts and income gains are substitutes at known exchange rates.

**Proposition 3 (threshold amplification).** Let w_c be country populations, and let S(k) = Σ_c w_c · 1[k π_c > θ m_c] / Σ_c w_c be the share of population for whom a bundle with English cost k is unaffordable. Relative to the no-premium curve S_0(k), the premium adds

  S(k) − S_0(k) = Σ_c w_c · 1[θ m_c/π_c < k ≤ θ m_c] / Σ_c w_c ≥ 0.

Each country's threshold moves left by ln π_c on the log-cost axis. The population effect at a given k is the mass of countries whose income sits within ln π_c of the threshold. It is therefore large when big countries sit just above the line, even if their ln π_c is modest. *Proof.* Directly from the indicator definition: c is priced out with the premium and not without it exactly when θm_c/π_c < k ≤ θm_c. ∎

**Proposition 4 (reasoning dilution).** Suppose a request produces V visible tokens (input plus output) and R = rV hidden reasoning tokens. Suppose too that reasoning is produced in a pivot language with premium 1 regardless of the user's language. Then the *cost* premium is

  π_eff(r) = (π + r) / (1 + r),

which tends to 1 as r grows. The *context* premium on visible tokens, and the latency of visible output, remain π. *Proof.* Cost in the local language is (πV + rV)·p; in English it is (V + rV)·p; take the ratio. ∎

Proposition 4 depends on the pivot assumption. If reasoning follows the user's language, π_eff = π. We therefore state it as a testable prediction (H3), not as a finding.

### 3.3 What the model omits

The model treats π as a pure multiplier on cost. Real deployments add per-request fees and subscription quotas. Under quotas, π_c shrinks the *effective quota*, Q/π_c, instead of raising the price. It omits quality differences: Ahia et al. [2023] and Lundin et al. [2026] document lower accuracy in high-premium languages, which would add to the gap. It also omits within-country income distribution, which A4AI notes is decisive for the poorest quintile [A4AI].

## 4 Measurement

### 4.1 Corpus

We use the NLTK `udhr2` corpus of Universal Declaration of Human Rights translations [Bird et al. 2009], downloaded from the NLTK data repository. We normalise text to Unicode NFC, collapse runs of whitespace within lines, drop empty lines, and exclude files under 5,000 UTF-8 bytes as partial translations (e.g., `pbu`, 296 bytes). That leaves **380 language/script/variant files**. The UDHR is a short (≈1,800 English words) legal-register text. Some translations include extra preamble lines. §5.1 quantifies the resulting noise.

### 4.2 Tokenizers

We obtain 11 production tokenizers offline from public package or repository artefacts, with no API access (`code/fetch_assets.sh`):

| Developer | Tokenizer (artefact) | Era |
|---|---|---|
| OpenAI | p50k_base (GPT-3 family) | 2020–22 |
| OpenAI | cl100k_base (GPT-4/3.5) | 2023 |
| OpenAI | o200k_base (GPT-4o and later) | 2024– |
| Anthropic | legacy Claude tokenizer (shipped in `anthropic==0.38.0`) | pre-Claude-3 |
| Meta | Llama 3 (128k BPE) [Llama Team 2024] | 2024 |
| Meta | Llama 4 (o200k-style pattern) | 2025 |
| Google | Gemma 1/2 SentencePiece (256k) | 2024 |
| Google | Gemma 3 SentencePiece (262k) [Gemma Team 2025] | 2025 |
| Mistral | v1 SentencePiece (32k) | 2023 |
| Mistral | Tekken (tekken_240911; tiktoken-based, >100 languages) [Mistral AI 2024] | 2024 |
| DeepSeek | DeepSeek-V3 [DeepSeek-AI 2024] | 2024–25 |

The OpenAI files come from tiktoken cache files bundled in `litellm==1.110.2`. Their names are the SHA-1 of the official download URLs, so the files are byte-identical to tiktoken's. The Llama 3/4 regex patterns are copied from Meta's `llama-models` repository. "Current generation" below means {o200k, Llama 4, Gemma 3, Tekken, DeepSeek-V3}. The model-version mapping of the OpenAI encodings is as publicly documented by tiktoken. We do not claim which tokenizer any current Anthropic model uses: the legacy file is old.

### 4.3 Countries

We hand-code a predominant language for 80 countries in `code/country_language.csv`. Usually this is the official or majority language. Where the official language is a colonial lingua franca, we use the largest first language or national language (e.g., Hausa for Nigeria, Hindi for India, Swahili for Kenya). These choices are judgment calls. A sensitivity scenario (§5.5) replaces them with English or French. GNI per capita (Atlas, current US$) and population come from the World Bank WDI API (vintage updated 2026-11-11; GNI year 2025) [World Bank]. Eritrea is excluded because its latest GNI is from 2011, leaving **79 countries**: 23 high income, 22 upper-middle, 23 lower-middle and 11 low income, with 7.33 billion people.

### 4.4 Statistics

The income gradient is the OLS slope of ln π_c on ln GNI_c across countries, with 95% percentile bootstrap CIs (4,000 resamples of countries), plus a population-weighted slope. Upgrade progressivity uses the language-level regression Δln π = α + β_inc·ln GNI_ℓ + β_0·ln π_old + ε over the 65 distinct country languages. Here GNI_ℓ is the population-weighted GNI of the countries coded to ℓ. Because the old and new tokenizers encode the same text, Δln π is free of content-length noise. This within-text differencing is the main identification advantage of comparing generations.

## 5 Results

### 5.1 Validation and noise (code output: `results/summary.json`, `fig1_validation.png`)

We compare UDHR o200k premiums with FLORES-200+ o200k premiums for 16 African languages transcribed from Somide [2026]:

- **Levels agree.** The median UDHR/FLORES ratio is 0.96.
- **Between-script contrasts agree.** On log scale, Pearson r = 0.79, driven by Ge'ez-script Amharic (UDHR 5.41 vs FLORES 7.36) and Tigrinya (5.94 vs 6.79).
- **Fine rankings within Latin script do not agree.** Spearman ρ = 0.21 (p = 0.44). Yoruba is a clear orthographic outlier: the UDHR text uses combining diacritics (e.g., U+0329), giving 4.16 vs 1.85.

Translation-variant pairs show the scale of content noise. Pairs that appear to share a text (Persian 1/2, Romanian 1953/1993, Hausa NE/NG) differ by only 0.005–0.009 in median |Δ ln π|. Independent translations or dialects (Haitian Creole variants: 0.22; Akan Asante vs Akuapem: 0.13) differ by up to ≈0.2. **We therefore trust our corpus for between-script and level comparisons and for within-text differences across tokenizers. We do not interpret small Latin-script differences between individual languages.** The country-level gradients below are regressions over many languages and are much less exposed to this noise. We also check that they survive excluding non-Latin scripts (§5.2).

### 5.2 The premium has an income gradient under every tokenizer (`fig2_income_gradient.png`, `robustness.json`)

Across all 380 files, the median premium fell from 2.37 (cl100k) to 1.91 (o200k), and the share of languages above 2× fell from 76% to 43%. The worst cases remain extreme (max ≈17.8× under o200k). Gemma 3 has the lowest maximum (8.6×).

Table 1 (code output). Income gradient of ln π across 79 countries.

| Tokenizer | slope | 95% CI | pop-weighted slope | pop-wtd mean π, LIC+LMIC | pop-wtd mean π, HIC | pop. share in countries with π>2 |
|---|---|---|---|---|---|---|
| p50k (GPT-3) | −0.101 | [−0.198, −0.013] | −0.310 | 5.98 | 2.07 | 0.88 |
| Claude legacy | −0.120 | [−0.199, −0.047] | −0.337 | 4.72 | 1.66 | 0.63 |
| Mistral v1 | −0.122 | [−0.193, −0.058] | −0.320 | 4.08 | 1.50 | 0.56 |
| cl100k (GPT-4) | −0.129 | [−0.207, −0.059] | −0.318 | 4.34 | 1.58 | 0.56 |
| Llama 3 | −0.143 | [−0.217, −0.081] | −0.281 | 3.01 | 1.36 | 0.44 |
| Gemma 1/2 | −0.131 | [−0.174, −0.093] | −0.206 | 2.07 | 1.20 | 0.32 |
| Tekken | −0.159 | [−0.220, −0.110] | −0.197 | 2.12 | 1.25 | 0.13 |
| o200k | −0.107 | [−0.158, −0.064] | −0.152 | 1.82 | 1.26 | 0.08 |
| DeepSeek-V3 | −0.143 | [−0.199, −0.095] | −0.294 | 2.75 | 1.31 | 0.46 |
| Llama 4 | −0.128 | [−0.164, −0.095] | −0.193 | 1.91 | 1.17 | 0.12 |
| Gemma 3 | −0.102 | [−0.134, −0.070] | −0.138 | 1.59 | 1.20 | 0.11 |

Every slope is negative and every CI excludes zero. A slope of −0.10 to −0.16 means a country ten times poorer faces a premium 10^{0.10}–10^{0.16} ≈ 1.26–1.44 times higher. In our sample, a 10× income gap is roughly the gap between the US (GNI pc $88,810) and Thailand ($7,690). The cross-country (unweighted) slope did not flatten much across generations. The population-weighted slope roughly halved, from −0.32 (cl100k) to −0.15 (o200k) and −0.14 (Gemma 3). Most of the progress came from large Indic and Southeast Asian languages, not from the long tail.

**Robustness.** Controlling for writing system (non-Latin script dummy), the income slope is −0.108 [−0.153, −0.066] for o200k and −0.102 [−0.135, −0.070] for Gemma 3. Within the 51 Latin-script countries alone it is −0.087 [−0.123, −0.054] (o200k) and −0.107 [−0.141, −0.074] (Gemma 3). At the language level (65 distinct languages, deduplicating Spanish, Arabic and so on) it is −0.09 (o200k). The gradient therefore does not simply reflect script. A plausible mechanism, which we do not test, is that tokenizer training mixtures overweight languages with large web presence, and web presence correlates with income [Joshi et al. 2020].

### 5.3 Upgrades help the worst-served most, but favour the rich at a given starting point (`summary.json: upgrade_progressivity`)

Table 2 (code output). Within-vendor tokenizer upgrades, 65 country languages.

| Upgrade | mean Δln π | ρ(income, Δ) | β_inc [95% CI] | β_0 (initial) [95% CI] | SD ln π before → after | # languages worse |
|---|---|---|---|---|---|---|
| p50k → cl100k | −0.263 | −0.36 (p=.003) | −0.053 [−0.080, −0.031] | −0.19 [−0.29, −0.12] | 0.62 → 0.55 | 1 |
| cl100k → o200k | −0.470 | −0.06 (p=.61) | −0.057 [−0.089, −0.028] | −0.62 [−0.82, −0.39] | 0.55 → 0.35 | 0 |
| Llama 3 → Llama 4 | −0.302 | −0.19 (p=.14) | −0.063 [−0.088, −0.039] | −0.55 [−0.66, −0.42] | 0.55 → 0.33 | 0 |
| Gemma 1 → Gemma 3 | −0.078 | +0.29 (p=.02) | −0.018 [−0.039, +0.006] | −0.41 [−0.52, −0.22] | 0.36 → 0.27 | 19 |
| Mistral v1 → Tekken | −0.243 | −0.46 (p<.001) | −0.087 [−0.121, −0.057] | −0.38 [−0.71, −0.10] | 0.47 → 0.48 | 7 |

Two patterns appear. First, upgrades compress the distribution: β_0 < 0 in all five, so languages that started with higher premiums gained more. cl100k→o200k cut the cross-language SD of ln π by 36%. Second, **holding the initial premium fixed, richer countries' languages gained more** in four of five upgrades (β_inc ≈ −0.05 to −0.09, CIs excluding zero). Gemma 1→3 is the exception: its CI includes zero, and its raw correlation is positive. The Mistral change was the most income-regressive. Its raw rank correlation between income and premium change is −0.46, consistent with Mistral's stated focus on European and major Asian languages [Mistral AI 2024]. The SD of its premiums did not fall, and 7 of 65 languages got worse.

β_inc is the *conditional* tendency. Measurement error in ln π_old (§5.1) attenuates β_0 and could bias β_inc, because income correlates with ln π_old. We read the second pattern as suggestive. It is consistent across vendors, but its sign is not robust to every specification: the unconditional ρ for cl100k→o200k is near zero.

### 5.4 Decomposition: language is a minority but non-trivial share of the gap (`country_decomposition.csv`, `fig3_decomposition.png`)

For the 34 low and lower-middle income countries:

- The median log income gap with the US is 4.13: income is e^{4.13} ≈ 62× lower.
- The median o200k premium is 1.75 (ln π = 0.56).
- The language share s_c (Proposition 1c) has a median of **12.5% under o200k**, **20.5% under cl100k**, and 11.4% under the best current tokenizer for each language.
- The largest shares under o200k: Lao 37.8%, Ethiopia (Amharic) 27.8%, Cambodia (Khmer) 25.3%, Myanmar (Burmese) 25.2%, Togo (Ewe) 21.6%.

In years (Proposition 1b with γ = 0), the median income lag is 5.96 / 1.88 / 1.06 years at price declines of 2× / 9× / 50× per year. The median language lag under o200k is 0.81 / 0.26 / 0.14 years. For a fixed capability tier at Epoch-like rates, the language tax therefore delays the median poor country by about one to three months. It delays Lao and Amharic users by much more: ln π = 2.26 and 1.69 under o200k, or 1.0 and 0.77 years at 9×/year.

### 5.5 Thresholds amplify (`fig4_affordability.png`)

Table 3 (code output). Share of the 7.33 billion population for whom a monthly bundle with English cost k exceeds 2% of monthly GNI per capita.

| English bundle cost k | English token counts | cl100k | o200k | best current | lingua-franca scenario, cl100k | lingua-franca scenario, o200k |
|---|---|---|---|---|---|---|
| $1 | 0.7% | 37.3% | 6.6% | 6.5% | 8.8% | 5.6% |
| $2 | 6.8% | 40.8% | 16.4% | 16.0% | 11.4% | 8.4% |
| $5 | 40.7% | 54.7% | 44.4% | 44.2% | 52.4% | 42.8% |
| $20 | 60.0% | 86.1% | 71.4% | 63.5% | 86.1% | 71.4% |

The large jumps come from Proposition 3. India (GNI pc $2,760; 2% of monthly income ≈ $4.60) under cl100k's Hindi premium of 5.60 has its threshold at ≈$0.82, which drives the step at k ≈ $0.8. The 25%-priced-out point sits at k = $4.63 with English counts, $2.77 under o200k and $0.82 under cl100k. That is a horizontal gap of 0.51 and 1.73 log units, or 0.23 and 0.79 years at 9×/year. The lingua-franca scenario prices users in English or French in 26 countries where those are official languages. It shrinks the effect a lot, so the population-level numbers depend heavily on *which language people actually use with AI*. That is an empirical question our data cannot settle.

### 5.6 Levers in common units

All figures are in log-cost units; years at 9×/year are in parentheses.

- **Choosing a tokenizer (free, immediate).** Median gain from the worst to the best current tokenizer for LIC/LMIC languages: 0.24 (0.11 yr). From cl100k to the best: 0.29 (0.13 yr). For specific scripts the gain is large. Lao, o200k (9.61) → Gemma 3 (2.97): 1.17 (0.53 yr). Amharic, 5.41 → 2.23: 0.89 (0.40 yr). Burmese, 4.12 → 3.13: 0.28. Tamil, Llama 4 (4.57) → Gemma 3 (1.89): 0.88 (code output). No single vendor's tokenizer is best for all: o200k is best for 35 of 79 countries, Llama 4 for 24, Gemma 3 for 11, Tekken for 8 and DeepSeek-V3 for 1.
- **Vendor price tiers.** ChatGPT Go launched in India at ₹399/month against ₹1,999 for Plus [TechCrunch 2025; CNBC 2025]. That is a list-price ratio of ln(1999/399) = 1.61 (0.73 yr), larger than the median language premium. Go is a different product with lower limits than Plus, so this is an upper bound on the like-for-like discount.
- **Income growth.** Closing the median income gap of 4.13 through a growth differential of 2 percentage points per year would take ≈ 200 years. Even an exceptional 5 points would take ≈ 80 years. On human time scales, only price declines (for fixed capability) and the two levers above move affordability.

### 5.7 Reasoning dilution (analytical, Proposition 4)

With o200k premiums and r hidden reasoning tokens per visible token, all of them in a pivot language, the effective cost premium for Amharic goes from 5.41 (r=0) to 3.20 (r=1), 2.10 (r=3) and 1.40 (r=10). For Burmese it goes from 4.12 to 2.56, 1.78 and 1.28. For Swahili it goes from 1.49 to 1.25, 1.12 and 1.04. The visible-context premium stays at π. If reasoning models do reason in a pivot language, the token tax will move from the bill to the context window and to latency. This is a prediction (H3), not an observation.

## 6 Discussion

**What this means for "how developing countries avoid being left behind".**

(1) *Price declines are necessary but not sufficient.* Proposition 1 says plainly that falling prices make any fixed capability affordable everywhere *eventually*, but they keep the burden ratio fixed. Users in the Global South keep getting a given capability tier later than users in the North, by about two years for the median low or lower-middle income country at the conservative 9×/year rate, and by about six years for frontier-tier products whose prices fall slowly. Policy can shorten that lag only with multiplicative levers.

(2) *The language component is the cheapest to remove.* Income gaps take generations to close (§5.6). Tokenization premiums fell by 0.47 log units on average in one OpenAI release (cl100k→o200k). For a ministry, a procurement agency or a developer in Ethiopia, Laos, Cambodia or Myanmar, *choosing the model family whose tokenizer suits the script* is worth up to 0.9–1.2 log units today. That matches or exceeds vendor discounts, and it can be done without anyone's permission. Public procurement rules could require reporting π on a national-language test set, and our code computes it offline in seconds. Regulators could ask vendors to publish per-language premiums or to bill per character or per unit of content. Ahia et al. [2023] made a similar recommendation, and our decomposition gives it a size.

(3) *Tokenizer progress is not automatically pro-poor.* Upgrades compress the worst premiums, which is good. Conditional on the initial premium, though, they still tilt toward richer countries' languages (§5.3). Open, auditable tokenizers let developing countries see this tilt. Extending or adapting vocabularies of open-weight models for national languages is a concrete step a developing country can take on its own; the Gemma and Llama artefacts we used are openly available.

(4) *Thresholds matter more than averages.* A modest premium can move a large country across an affordability line (Proposition 3, Table 3). That makes the language people actually use with AI, which is still unmeasured, a first-order policy variable. National statistics offices could add it to ICT household surveys.

**Relation to quality gaps.** Our model is about cost. Ahia et al. [2023] and Lundin et al. [2026] report that high-premium languages also get lower accuracy. If quality enters as a further multiplier on "content per dollar", the language share computed here is a *lower bound* on the true language share of the gap.

## 7 Falsifiable Predictions and Test Protocols

**H1 (income-conditional regressivity of tokenizer upgrades).** For the next generation of tokenizers from major developers, Δln π regressed on ln GNI and ln π_old over the 65 country languages will give β_0 < 0 and β_inc < 0. *Test:* run `code/tokenize_udhr.py` with the new tokenizer added, then rerun `analysis.py`. *Falsified if* β_inc ≥ 0 with a CI excluding negative values for most new releases. A FLORES-based replication would reduce content noise.

**H2 (quota premium under subscriptions).** Under token-metered quotas, local-language users of plan X hit rate limits after roughly 1/π as much content as English users. *Test:* matched-content scripted sessions in English and in the local language on a subscription account, counting messages until the limit is reached (needs accounts; we did not run this).

**H3 (reasoning dilution).** For reasoning models with hidden reasoning, the billed-token ratio for parallel prompts will be closer to (π + r)/(1 + r) than to π, where r is read from reported reasoning-token counts. *Test:* submit 200 FLORES parallel prompts in 10 languages through the API, record input, output and reasoning token counts from the usage fields, and regress the billed ratio on π and r. *Falsified if* the ratio stays near π, which would mean reasoning follows the user's language.

**H4 (adoption).** Conditional on ln GNI per capita and internet penetration, country-level generative-AI user share [Microsoft AI Economy Institute 2026] will fall with ln π of the predominant language under the dominant vendor's tokenizer, with an elasticity around −s_c / (income elasticity of adoption). *Test:* a cross-country regression once country-level diffusion data are published for enough low-income countries. *Falsified if* the coefficient on ln π is ≥ 0 with a tight CI.

## 8 Limitations

- **Corpus.** The UDHR is a single short legal text. Independent translations differ in content by up to ≈0.2 log units. Some orthographies (Yoruba with combining marks) differ from everyday usage. Our validation against FLORES shows good agreement on levels and between scripts but no reliable ranking within Latin-script African languages (ρ = 0.21). Conversational, code-mixed and transliterated text (e.g., romanised Hindi), which is common in actual chat use, may have very different premiums.
- **Language assignment.** Using one predominant language per country is crude. Multilingual countries (India, Nigeria, the DRC) contain many languages with different premiums, and many users write to AI in English or French. The lingua-franca sensitivity (Table 3) shows that population-level conclusions depend strongly on this. Country-level shares of AI-use language would be needed to settle it.
- **Tokenizer coverage.** We cover 11 artefacts available offline. We do not include current Anthropic, Qwen, Gemini-API or newest OpenAI tokenizers whose vocabularies were not obtainable here. The legacy Claude tokenizer does not represent current Claude models. The mapping of OpenAI encodings to products follows tiktoken's public documentation.
- **Economic model.** Exponential price decline at one rate, bundles of fixed content, GNI per capita as income (not the median or the poorest quintile), and a 2% threshold borrowed from broadband are all simplifications. We use them because they make the decomposition exact and transparent, not because they are accurate. Per-request fees, free tiers, local hosting of open-weight models (where cost depends on hardware, not on vendor prices), currency effects and taxes are omitted. Bundle costs in Table 3 are in English-equivalent dollars, chosen to illustrate rather than calibrated to observed spending.
- **Statistics.** 79 countries and 65 languages give limited power. Several countries share a language, so country-level observations are not independent; we report language-level slopes as a check. The progressivity regressions are associational and exposed to measurement error in ln π_old.
- **No model calls.** We did not query any LLM. Claims about accuracy come from cited work, and Propositions 4/H3 are untested.

## 9 Conclusion

Expressed in the same units as income and price, the tokenization premium is a minority but non-trivial part of the AI affordability gap. It is about one-eighth of the log gap for the median low or lower-middle income country today, down from one-fifth under the GPT-4 tokenizer, and a quarter to over a third for several Southeast Asian and Ethiopian languages. The premium falls with income under all 11 tokenizers we measured. Tokenizer upgrades have mainly helped the worst-served languages, yet conditional on starting point they have tended to favour richer countries' languages. Falling prices turn the divide into a lag; they do not remove it. For developing countries, the cheapest available lever is often simply choosing, and requiring vendors to disclose, the tokenizer that fits their languages. Our code and data let anyone recompute these quantities for new tokenizers in about a minute on a CPU.

## Reproducibility

`bash code/run_all.sh` downloads all inputs and regenerates every table and figure (about one minute of CPU after downloads). Outputs: `results/token_counts.csv`, `premium.csv`, `country_decomposition.csv`, `summary.json`, `robustness.json`, and `fig1–fig5`. World Bank data are fetched live; the vintage used here was last updated 2026-11-11.

**Provenance of numbers.** Tables 1–3, §5.1–5.7 and the figures are code outputs. Price-decline rates come from Epoch AI [2025]. Adoption rates come from Microsoft AI Economy Institute [2026]. ChatGPT Go and Plus prices come from TechCrunch [2025] and CNBC [2025]. The 2% threshold comes from A4AI. FLORES premiums used for validation come from Somide [2026].

## References

- A4AI (Alliance for Affordable Internet). "Affordable Internet is '1 for 2'." https://adi.a4ai.org/affordable-internet-is-1-for-2 (accessed Nov. 2026).
- Ahia, O., Kumar, S., Gonen, H., Kasai, J., Mortensen, D. R., Smith, N. A., & Tsvetkov, Y. (2023). Do All Languages Cost the Same? Tokenization in the Era of Commercial Language Models. *Proc. EMNLP 2023*. https://aclanthology.org/2023.emnlp-main.614
- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly. (NLTK data: `udhr2` corpus, https://github.com/nltk/nltk_data)
- CNBC (2025, Oct. 9). OpenAI expands its cheapest ChatGPT plan to 16 more countries in Asia. https://www.cnbc.com/2025/10/09/openai-expands-its-cheapest-chatgpt-plan-to-16-more-countries-in-asia.html
- DeepSeek-AI (2024). DeepSeek-V3 Technical Report. arXiv:2412.19437.
- Epoch AI (2025). LLM inference prices have fallen rapidly but unequally across tasks. https://epoch.ai/data-insights/llm-inference-price-trends
- Gemma Team, Google DeepMind (2025). Gemma 3 Technical Report. arXiv:2503.19786.
- ITU (2022, Mar. 17). Global broadband less affordable (press release on ITU–A4AI policy brief "The affordability of ICT services 2021"). https://www.itu.int/en/mediacentre/Pages/PR-2022-03-17-Global-broadband-less-affordable.aspx
- Joshi, P., Santy, S., Budhiraja, A., Bali, K., & Choudhury, M. (2020). The State and Fate of Linguistic Diversity and Inclusion in the NLP World. *Proc. ACL 2020*, 6282–6293. https://aclanthology.org/2020.acl-main.560
- Korinek, A., & Stiglitz, J. E. (2021). Artificial Intelligence, Globalization, and Strategies for Economic Development. NBER Working Paper 28453. https://www.nber.org/papers/w28453
- Llama Team, AI @ Meta (2024). The Llama 3 Herd of Models. arXiv:2407.21783.
- Lundin, J. M., Zhang, A., Karim, N., Louzan, H., Wei, G., Adelani, D. I., & Carroll, C. (2026). The Token Tax: Systematic Bias in Multilingual Tokenization. *Proc. 7th Workshop on African NLP (AfricaNLP 2026)*, 103–112. https://aclanthology.org/2026.africanlp-main.10 (arXiv:2509.05486)
- Microsoft AI Economy Institute (2026). Global AI Adoption in 2025 — A Widening Digital Divide (AI Diffusion Report, H2 2025). https://www.microsoft.com/en-us/research/wp-content/uploads/2026/01/Microsoft-AI-Diffusion-Report-2025-H2.pdf
- Mistral AI (2024, Jul.). Mistral NeMo (announcement of the Tekken tokenizer). https://mistral.ai/news/mistral-nemo
- Petrov, A., La Malfa, E., Torr, P., & Bibi, A. (2023). Language Model Tokenizers Introduce Unfairness Between Languages. *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*.
- Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., & Gurevych, I. (2021). How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models. *Proc. ACL-IJCNLP 2021*. https://aclanthology.org/2021.acl-long.243
- Somide, O. A. (2026). The African Language Tax: Quantifying the Cost, Latency, and Context Penalty of Tokenizing African Languages in Frontier LLMs. arXiv:2606.24460.
- TechCrunch (2025, Aug. 19). OpenAI launches a sub-$5 ChatGPT plan in India (syndicated at https://finance.yahoo.com/news/openai-launches-sub-5-chatgpt-035715856.html).
- Wendler, C., Veselovsky, V., Monea, G., & West, R. (2024). Do Llamas Work in English? On the Latent Language of Multilingual Transformers. *Proc. ACL 2024*. https://aclanthology.org/2024.acl-long.820
- World Bank. World Development Indicators API: NY.GNP.PCAP.CD, NY.GNP.PCAP.PP.CD, SP.POP.TOTL. https://api.worldbank.org/v2/ (vintage updated 2026-11-11).
