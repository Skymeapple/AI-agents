"""
Coded obligation matrix for a frontier general-purpose generative-AI provider
(training compute > 1e26 FLOP, annual revenue > USD 500M, offering a consumer
chat product and an API) serving six market segments.

Segments
  EU  : European Union (Regulation (EU) 2024/1689, "AI Act")
  CN  : People's Republic of China (Interim Measures for Generative AI Services 2023;
        Measures for Labeling AI-Generated Synthetic Content 2025 + GB 45438-2025)
  USF : US federal procurement market (EO 14319, "Preventing Woke AI in the Federal Government")
  CA  : California (SB 53 TFAIA 2025; SB 942 AI Transparency Act as amended by AB 853; AB 2013)
  NY  : New York (RAISE Act, signed 19 Dec 2025, chapter amendment signed 27 Mar 2026)
  ROW : rest of world, coded as imposing no AI-specific obligation (baseline/outside option)

Each dimension has
  kind   : 'ordered' (requirement is a level; a product at level L satisfies every level <= L)
           'format'  (requirement is a set of specification tokens; different tokens do not
                      satisfy each other but can be stacked at additive cost)
  layer  : 'weights' | 'runtime' | 'process'  -- the product layer at which compliance is
           assessed, which determines the cost of forking to satisfy it
  req    : per-segment requirement (int level, or frozenset of tokens)
  src    : provenance (article / instrument) used for coding
Conflicts (mutually unsatisfiable requirements) are listed separately per hypothesis.

Coding is by the author from the primary texts / verified secondary summaries cited in
paper.md; cells we could not verify are coded as absent and flagged in `unverified`.
"""

SEGMENTS = ["EU", "CN", "USF", "CA", "NY", "ROW"]

F = frozenset

DIMENSIONS = {
    # ---------------------------------------------------------------- process layer
    "D1_incident_reporting": dict(
        kind="ordered", layer="process",
        # levels: 0 none, 1 = 15 days (CA), 2 = 72 h (NY) / 'without undue delay' (EU)
        req=dict(EU=2, CN=0, USF=0, CA=1, NY=2, ROW=0),
        src="EU AI Act Art.55(1)(c) ('without undue delay'); CA SB 53 (15 days, 24h if imminent);"
            " NY RAISE as amended (72 h). EU level is ambiguous -> varied in sensitivity.",
    ),
    "D2_safety_framework": dict(
        kind="format", layer="process",
        req=dict(EU=F({"EU_systemic_risk_assessment"}), CN=F(), USF=F(),
                 CA=F({"CA_frontier_AI_framework"}), NY=F({"NY_safety_protocol"}), ROW=F()),
        src="EU Art.55(1)(b) (+GPAI Code of Practice, Art.56); CA SB 53 framework; NY RAISE safety protocols.",
    ),
    "D4_training_data_summary": dict(
        kind="format", layer="process",
        req=dict(EU=F({"EU_public_summary_template"}), CN=F(), USF=F(),
                 CA=F({"CA_AB2013_dataset_summary"}), NY=F(), ROW=F()),
        src="EU Art.53(1)(d); CA AB 2013 (from 1 Jan 2026).",
    ),
    "D9_illegal_content_reporting": dict(
        kind="format", layer="process",
        req=dict(EU=F(), CN=F({"CN_illegal_content_report"}), USF=F(), CA=F(), NY=F(), ROW=F()),
        src="CN Interim Measures Art.14.",
    ),
    "D10_premarket_filing": dict(
        kind="format", layer="process",
        req=dict(EU=F({"EU_Art52_notification"}), CN=F({"CN_security_assessment_algorithm_filing"}),
                 USF=F({"USF_contract_terms"}), CA=F(), NY=F(), ROW=F()),
        src="EU Art.52(1) notification; CN Art.17 (security assessment + algorithm filing);"
            " EO 14319 contract compliance terms.",
    ),
    "D11_whistleblower": dict(
        kind="format", layer="process",
        req=dict(EU=F({"EU_Directive_2019_1937"}), CN=F(), USF=F(),
                 CA=F({"CA_SB53_whistleblower"}), NY=F(), ROW=F()),
        src="EU AI Act Art.87 (Directive (EU) 2019/1937 applies); CA SB 53.",
    ),
    # ---------------------------------------------------------------- weights layer
    "D3_model_evaluation": dict(
        kind="ordered", layer="weights",
        # 0 none, 1 = evaluation described in published framework, 2 = mandated testing/assessment
        req=dict(EU=2, CN=2, USF=0, CA=1, NY=1, ROW=0),
        src="EU Art.55(1)(a) (adversarial testing); CN Art.17 security assessment; CA/NY frameworks.",
    ),
    "D5_training_data_provenance": dict(
        kind="format", layer="weights",
        req=dict(EU=F({"EU_TDM_optout_policy"}), CN=F({"CN_lawful_source"}), USF=F(),
                 CA=F(), NY=F(), ROW=F()),
        src="EU Art.53(1)(c) (Directive 2019/790 Art.4(3) reservations); CN Art.7(1) lawful sources.",
    ),
    "D8_political_content_norm": dict(
        kind="ordered", layer="weights",
        # 1 = model must uphold CN content norms and be corrected by 'model optimization training'
        req=dict(EU=0, CN=1, USF=0, CA=0, NY=0, ROW=0),
        src="CN Interim Measures Art.4(1) (Core Socialist Values) and Art.14 (model optimization"
            " training); EO 14319 truth-seeking/ideological neutrality for federal procurement (conflict).",
    ),
    # ---------------------------------------------------------------- runtime layer
    "D6_machine_readable_marking": dict(
        kind="format", layer="runtime",
        req=dict(EU=F({"EU_Art50_2_marking"}), CN=F({"CN_GB45438_implicit_metadata"}), USF=F(),
                 CA=F({"CA_SB942_latent_disclosure"}), NY=F(), ROW=F()),
        src="EU Art.50(2); CN Labeling Measures + GB 45438-2025 implicit labels; CA SB 942/AB 853.",
    ),
    "D7_visible_disclosure": dict(
        kind="format", layer="runtime",
        req=dict(EU=F({"EU_Art50_1_interaction_notice"}), CN=F({"CN_GB45438_explicit_label"}), USF=F(),
                 CA=F({"CA_SB942_manifest_option"}), NY=F(), ROW=F()),
        src="EU Art.50(1); CN explicit labels; CA SB 942 manifest disclosure option.",
    ),
}

# cells deliberately coded absent because presence could not be verified
UNVERIFIED = {
    "D11_whistleblower": ["NY"],
    "D4_training_data_summary": ["CN"],
}

# Conflict hypotheses: list of (dimension, segment_a, segment_b)
CONFLICTS = {
    "H0_no_conflict": [],
    "H1_CN_vs_USF": [("D8_political_content_norm", "CN", "USF")],
    "H3_CN_vs_US_wide": [("D8_political_content_norm", "CN", s) for s in ("USF", "CA", "NY")],
    # H4: same conflict, but the US side accepts runtime-level (system-prompt/output-filter)
    # compliance, so the conflict binds only at the runtime layer (CN norm still in weights)
    "H4_USF_runtime": [("D8_political_content_norm", "CN", "USF", "runtime")],
}

# Hypothesis H2 = H1 conflict, but CN content norm assessable at the runtime layer
LAYER_OVERRIDES = {
    "H2_CN_runtime": {"D8_political_content_norm": "runtime"},
}

# reference market sizes (USD tn, approximate 2024 nominal GDP; EU/US/CN from World Bank data via
# secondary aggregators -- see paper; CA/NY/USF/ROW are rough author assumptions (ROW ~ world GDP minus coded segments), sampled widely)
MARKET_REF = dict(EU=19.4, CN=18.7, USF=0.5, CA=4.0, NY=2.3, ROW=65.0)
