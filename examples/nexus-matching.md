# Open problems against active research directions

Corpus: 2444 real arXiv abstracts from `ai2035/frontier` (machine-learning papers from 2025 on, harvested by search query per ai2035 question).
Discovery: 196 directions covering 2037 works (83%); 407 works too isolated to place; vocabulary 9466 terms (similarity floor 0.1).
Catalogues: 204 problems in 6 catalogues.
Fields the corpus is taken to cover: `ai`.

| Result | Count |
| --- | --- |
| Problems matched to at least one direction | 51 |
| Problems neglected (in a field the corpus covers, no direction on them) | 55 |
| Problems outside the corpus (cannot be judged from it) | 98 |
| Directions orphaned (tied to no catalogued problem) | 98 of 196 |

The counts are `observed`: computed from this corpus by the rule below. The rule's constants and the field declarations are `assumed`.

## The rule

- similarity floor = 0.15  <- assumed: cosine below which a problem and a direction share too little vocabulary to call a match; the value that best separated matched from neglected on the tuning half of the known-pairs experiment (6 questions), not derived ** NEEDS REVIEW **
- margin over runner-up = 1.25  <- assumed: how far the best problem of a catalogue must beat the runner-up before a direction is linked to it. The tuning half could not tell 1.0 from 1.5 at the shipped floor, so this is a precaution, not a fitted value. Lower than sciscout.classify's 1.5 because neighbouring problems in one catalogue overlap far more than commercial sectors do ** NEEDS REVIEW **
- problem fields  <- assumed: each catalogue's field, and the fields a corpus covers, are declared by hand: AI-100 by its own category column (technical categories -> 'ai'), Human Frontier by its field column (harness / pre / post -> 'ai'), the rest by catalogue ** NEEDS REVIEW **
- fields the corpus covers = `ai`  <- assumed: the ai2035 frontier harvest holds technical machine-learning papers only; it was built from search queries for 20 questions, so it is not a sample of all of machine learning either ** NEEDS REVIEW **

A direction is linked to a problem when the problem is its best match within its catalogue, clears the floor, and beats that catalogue's runner-up by the margin. An unmatched problem is *neglected* only if its field is one the corpus covers; otherwise it is *outside* the corpus and nothing is concluded.

## By catalogue

| Catalogue | Problems | Matched | Neglected | Outside corpus | Matching text |
| --- | --- | --- | --- | --- | --- |
| AI-100 open questions (`ai100`) | 100 | 24 | 39 | 37 | assumed: ai100_en.yaml: English rendering and matching terms written by the model for le… |
| ai2035 basic-science questions (`ai2035`) | 20 | 19 | 1 | 0 | sourced: arXiv search phrases listed for each question |
| Human Frontier challenges (`hfm`) | 54 | 7 | 15 | 32 | sourced: challenge title and field |
| Ten cosmology problems (`cosmos`) | 10 | 0 | 0 | 10 | sourced: module docstring |
| Ten mathematics problems (`maths`) | 10 | 1 | 0 | 9 | sourced: problem name and what is simulated |
| worldsim's ten global problems (`world`) | 10 | 0 | 0 | 10 | sourced: problem names and outcome labels |

## How well the matcher works, measured

The ai2035 frontier papers are labelled by the question whose search returned them, which gives a leave-one-out test of all three lists. For each problem with known papers: is it linked to a direction made mostly of those papers (match recall, and link precision over every link it gets); with the papers removed and the corpus re-clustered, does it come back unmatched (neglect detection); with the problem removed from the catalogue, do its papers' directions come back orphaned (orphan detection).

| Test | Problems | Match recall | Link precision | Neglect detection | Orphan detection |
| --- | --- | --- | --- | --- | --- |
| ai2035 questions on their own search phrases (easy) | 20 | 19/20 (95%) | 69/70 (99%) | 16/20 (80%) | 164/165 (99%) |
| the same with every harvest-query word withheld | 20 | 1/20 (5%) | 2/5 (40%) | 16/20 (80%) | 163/165 (99%) |
| AI-100 known pairs, tuning half (001-016) | 6 | 5/6 (83%) | 10/11 (91%) | 4/6 (67%) | not defined |
| **AI-100 known pairs, held-out half (017-099)** | 7 | 7/7 (100%) | 11/15 (73%) | 4/7 (57%) | not defined |

How to read it:

- The first row is the easy case: a question's matching text is nearly the query its papers were found with. It shows the plumbing works and little more.
- Withholding the query words leaves a median of 4 terms per question, so almost nothing matches. The matcher runs on a topic's name-words and has nothing else to go on.
- The known pairs are the fairest test here: 13 AI-100 questions paired by hand (`data/nexus/matching/known_pairs.yaml`, `assumed`) with the ai2035 question on the same topic, matched on a gloss written separately from the harvest queries, against the other 99 AI-100 questions. Orphan detection is not defined for them: other AI-100 questions legitimately claim the same papers.
- The floor was chosen on the tuning half only, by the share of matched / neglected verdicts it got right, ties broken by link precision (best there: floor 0.15 at margin 1, 1.25, 1.5, 2; shipped: floor 0.15, margin 1.25). The tuning half cannot tell the margins apart.
- **Neglect detection is the weak number.** When a problem's own literature is taken out, a neighbouring direction often still clears the floor and the problem is reported matched. So the neglected list below is an undercount, and a "matched" verdict deserves a look at the direction's terms.
- Every figure is `observed` on this corpus, and the samples are tiny: 20, 6 and 7 problems.

## Matched

| Problem | Direction (distinctive terms) | Works | Active | Similarity | Margin |
| --- | --- | --- | --- | --- | --- |
| `ai100:002` How much more can test-time compute deliver? | test-time, compute, scaling, reasoning, search | 20 | 2025-01..2026-09 | 0.37 | 2.76x |
| `ai100:002` How much more can test-time compute deliver? | reasoning, lrms, compute, constraint, length | 5 | 2025-07..2026-06 | 0.18 | 2.47x |
| `ai100:003` Can reinforcement learning drive fast progress outside math… | verifiable, rewards, rlvr, reasoning, reinforcement | 51 | 2025-03..2026-11 | 0.43 | 5.51x |
| `ai100:003` Can reinforcement learning drive fast progress outside math… | imitation, agent, policy, demonstrations, multi | 6 | 2025-07..2026-11 | 0.17 | 2.42x |
| `ai100:005` Do models truly generalise, or is it sophisticated interpol… | compositional, generalization, clip, compositions, combinat… | 21 | 2025-01..2026-07 | 0.28 | 4.53x |
| `ai100:005` Do models truly generalise, or is it sophisticated interpol… | compositional, generalization, compositionality, compositio… | 9 | 2025-09..2026-11 | 0.21 | 2.84x |
| `ai100:005` Do models truly generalise, or is it sophisticated interpol… | compositional, syntactic, cogs, generalization, lexical | 3 | 2025-02..2026-08 | 0.17 | 3.93x |
| `ai100:007` What are the limits of unified multimodal models? | generation, multimodal, semantic, image, preservation | 4 | 2025-11..2026-07 | 0.21 | 2.54x |
| `ai100:008` How small can distillation and small models compress fronti… | compression, compressed, pruning, robustness, certified | 4 | 2025-06..2025-11 | 0.21 | 2.25x |
| `ai100:009` Can new architectures replace the Transformer? | mha, attention, transformer, hopfield, non-linear | 3 | 2025-05..2025-11 | 0.20 | 3.02x |
| `ai100:012` How can a model reliably know what it does not know? | conformal, prediction, coverage, uncertainty, sets | 60 | 2025-01..2026-11 | 0.22 | 6.44x |
| `ai100:013` Can hallucination be eliminated at the root? | hallucination, detection, hallucinations, llms, language | 55 | 2025-01..2026-11 | 0.34 | 2.42x |
| `ai100:014` Can models reason causally rather than by correlation? | causal, visual, vision-language, reasoning, benchmark | 4 | 2025-03..2026-05 | 0.19 | 3.21x |
| `ai100:014` Can models reason causally rather than by correlation? | causal, interventions, pearl, interventional, confounders | 4 | 2025-01..2026-05 | 0.17 | 2.85x |
| `ai100:016` Can models learn continually without catastrophic forgettin… | continual, forgetting, catastrophic, replay, tasks | 44 | 2025-02..2026-10 | 0.38 | 5.94x |
| `ai100:016` Can models learn continually without catastrophic forgettin… | plasticity, continual, loss, deep, neural | 25 | 2025-02..2026-10 | 0.36 | 4.96x |
| `ai100:016` Can models learn continually without catastrophic forgettin… | backdoor, continual, iot, cps, persistent | 3 | 2026-06..2026-09 | 0.17 | 2.73x |
| `ai100:017` Do models build internal representations of the world? | world, video, planning, latent, interactive | 38 | 2025-03..2026-11 | 0.33 | 4.10x |
| `ai100:017` Do models build internal representations of the world? | world, causal, representations, whether, discovery | 4 | 2025-06..2026-08 | 0.25 | 2.59x |
| `ai100:017` Do models build internal representations of the world? | latents, world, linear, boolean, provably | 3 | 2025-02..2026-10 | 0.18 | 2.20x |
| `ai100:017` Do models build internal representations of the world? | internal, reward-hacking, reward-hack, monitoring, action | 3 | 2026-03..2026-10 | 0.17 | 1.28x |
| `ai100:017` Do models build internal representations of the world? | causal, latent, representation, variables, crl | 34 | 2025-03..2026-10 | 0.16 | 1.34x |
| `ai100:019` Can AI propose and verify original scientific hypotheses? | hypothesis, co-scientist, scientific, discovery, hypotheses | 6 | 2025-02..2026-09 | 0.36 | 4.74x |
| `ai100:019` Can AI propose and verify original scientific hypotheses? | scientific, science, discovery, scientists, disciplines | 12 | 2025-03..2026-10 | 0.27 | 2.55x |
| `ai100:023` How should multi-agent systems divide work, communicate and… | communication, multi-agent, reinforcement, emergent, inter-… | 8 | 2025-04..2026-10 | 0.21 | 1.40x |
| `ai100:026` How are an agent's memory, identity and state made persiste… | cognitive, episodic, memory, act-r, declarative | 5 | 2025-04..2026-08 | 0.17 | 1.76x |
| `ai100:034` How is reward hacking detected and prevented? | hacking, reward, rlhf, rewards, reward-hacking | 51 | 2025-01..2026-11 | 0.32 | 1.63x |
| `ai100:036` Can jailbreaks and prompt injection be solved fundamentally? | jailbreak, safety, attacks, harmful, attack | 10 | 2025-04..2026-10 | 0.23 | 2.47x |
| `ai100:036` Can jailbreaks and prompt injection be solved fundamentally? | adversarial, robustness, attacks, attack, defense | 35 | 2025-01..2026-11 | 0.21 | 3.02x |
| `ai100:042` Can mechanistic interpretability scale to frontier models? | mechanistic, interpretability, saes, circuit, features | 34 | 2025-01..2026-11 | 0.42 | 1.36x |
| `ai100:042` Can mechanistic interpretability scale to frontier models? | mechanistic, interpretability, gemma, side-channel, sca | 3 | 2025-02..2026-09 | 0.20 | 1.29x |
| `ai100:048` Are emergent abilities real or artefacts of measurement? | abilities, emergent, capabilities, language, emergence | 6 | 2025-01..2025-11 | 0.23 | 3.27x |
| `ai100:048` Are emergent abilities real or artefacts of measurement? | grokking, generalization, delayed, transition, weight | 47 | 2025-01..2026-11 | 0.22 | 3.03x |
| `ai100:049` Where does a model's persona come from, and is it stable? | personality, personalities, traits, emotional, emotion-aware | 4 | 2025-01..2025-12 | 0.17 | 1.53x |
| `ai100:050` How exactly does training data shape a model's beliefs and… | belief, epistemic, friction, cognitive, linguistic | 3 | 2025-04..2025-06 | 0.19 | 2.02x |
| `ai100:056` Are emotional relationships between people and AI good or b… | advice, reliance, appropriate, decision-making, people | 5 | 2025-11..2026-06 | 0.17 | 2.94x |
| `ai100:065` How should education adapt to the AI era? | education, science, ai-supported, chapter, ethical | 4 | 2025-08..2026-02 | 0.21 | 2.95x |
| `ai100:067` Does over-reliance on AI erode human skills? | reflection, workflow, creative, cognitive, workshop | 3 | 2026-03..2026-07 | 0.15 | 1.32x |
| `ai100:099` How do humans and AI collaborate best rather than substitut… | human-ai, collaboration, human, collaborative, humans | 32 | 2025-01..2026-10 | 0.30 | 2.89x |
| `ai100:099` How do humans and AI collaborate best rather than substitut… | human-ai, recommendations, collaboration, perceptions, human | 4 | 2025-05..2026-06 | 0.20 | 1.86x |
| `ai2035:Q01` Cognition and computation: the computational nature of inte… | associative, memory, hopfield, retrieval, dense | 51 | 2025-02..2026-11 | 0.34 | 8.31x |
| `ai2035:Q01` Cognition and computation: the computational nature of inte… | cognitive, cognition, metacognition, architecture, human-li… | 8 | 2025-01..2026-04 | 0.28 | 7.68x |
| `ai2035:Q01` Cognition and computation: the computational nature of inte… | cognitive, episodic, memory, act-r, declarative | 5 | 2025-04..2026-08 | 0.22 | 8.85x |
| `ai2035:Q01` Cognition and computation: the computational nature of inte… | cnn, hopfield, prototypes, classification, class-specific | 4 | 2025-01..2025-07 | 0.17 | 3.39x |
| `ai2035:Q01` Cognition and computation: the computational nature of inte… | cognitive, workspace, artificial, intelligence, architecture | 3 | 2025-05..2026-02 | 0.16 | 1.65x |
| `ai2035:Q01` Cognition and computation: the computational nature of inte… | hopfield, synchronous, big, ensuring, asymmetric | 3 | 2025-03..2026-07 | 0.16 | 2.61x |
| `ai2035:Q03` Scaling laws and emergent abilities in large models | abilities, emergent, capabilities, language, emergence | 6 | 2025-01..2025-11 | 0.25 | 3.93x |
| `ai2035:Q04` Statistical foundations: generalisation under over-paramete… | overfitting, benign, regression, covariance, kernel | 25 | 2025-01..2026-10 | 0.45 | 5.16x |
| `ai2035:Q04` Statistical foundations: generalisation under over-paramete… | double, descent, phenomenon, error, theory | 29 | 2025-01..2026-10 | 0.37 | 5.24x |
| `ai2035:Q04` Statistical foundations: generalisation under over-paramete… | overparameterized, interpolating, tradeoff, criterion, inte… | 4 | 2025-12..2026-11 | 0.27 | 7.63x |
| `ai2035:Q04` Statistical foundations: generalisation under over-paramete… | regime, underparameterized, capacity, descent, double | 3 | 2025-09..2026-08 | 0.18 | 1.47x |
| `ai2035:Q04` Statistical foundations: generalisation under over-paramete… | out-of-sample, double, risk, in-sample, portfolio | 3 | 2026-02..2026-07 | 0.16 | 3.92x |
| `ai2035:Q05` Theory of reinforcement learning: exploration and sample co… | regret, bound, sqrt, bounds, log | 55 | 2025-01..2026-11 | 0.37 | 4.48x |
| `ai2035:Q06` Learning from few samples and few labels | few-shot, labeled, classification, fsl, class | 21 | 2025-01..2026-10 | 0.27 | 5.37x |
| `ai2035:Q06` Learning from few samples and few labels | few-shot, disentangled, meta-learning, classes, fast | 3 | 2025-03..2026-02 | 0.16 | 3.57x |
| `ai2035:Q07` Continual learning and catastrophic forgetting | continual, forgetting, catastrophic, replay, tasks | 44 | 2025-02..2026-10 | 0.46 | 10.01x |
| `ai2035:Q07` Continual learning and catastrophic forgetting | backdoor, continual, iot, cps, persistent | 3 | 2026-06..2026-09 | 0.22 | 9.25x |
| `ai2035:Q07` Continual learning and catastrophic forgetting | plasticity, continual, loss, deep, neural | 25 | 2025-02..2026-10 | 0.21 | 1.62x |
| `ai2035:Q07` Continual learning and catastrophic forgetting | continual, memory, on-chip, platforms, imc | 3 | 2025-09..2026-07 | 0.20 | 3.01x |
| `ai2035:Q07` Continual learning and catastrophic forgetting | experience, continual, skills, agents, skill | 6 | 2025-12..2026-10 | 0.18 | 4.19x |
| `ai2035:Q08` Causality-driven machine learning | invariant, irm, minimization, risk, ood | 12 | 2025-02..2026-11 | 0.37 | 2.36x |
| `ai2035:Q08` Causality-driven machine learning | causal, latent, representation, variables, crl | 34 | 2025-03..2026-10 | 0.27 | 6.52x |
| `ai2035:Q08` Causality-driven machine learning | correlations, spurious, multimodal, unimodal, causal | 4 | 2025-02..2025-11 | 0.25 | 3.42x |
| `ai2035:Q08` Causality-driven machine learning | causal, interventions, pearl, interventional, confounders | 4 | 2025-01..2026-05 | 0.23 | 3.04x |
| `ai2035:Q08` Causality-driven machine learning | causal, visual, vision-language, reasoning, benchmark | 4 | 2025-03..2026-05 | 0.15 | 2.85x |
| `ai2035:Q09` Out-of-distribution generalisation and extrapolation | augmentation, domain, sdg, generalization, retinopathy | 3 | 2025-02..2025-09 | 0.20 | 2.38x |
| `ai2035:Q10` Fusing knowledge and data: neuro-symbolic and physics priors | neuro-symbolic, symbolic, concepts, reasoning, programs | 16 | 2025-03..2026-09 | 0.20 | 3.76x |
| `ai2035:Q10` Fusing knowledge and data: neuro-symbolic and physics priors | physics-informed, pinns, neural, networks, equations | 48 | 2025-01..2026-11 | 0.16 | 1.88x |
| `ai2035:Q11` Interpretability of AI | mechanistic, interpretability, saes, circuit, features | 34 | 2025-01..2026-11 | 0.32 | 9.75x |
| `ai2035:Q11` Interpretability of AI | patching, mechanistic, interpretability, activation, llms | 6 | 2025-02..2026-10 | 0.18 | 15.84x |
| `ai2035:Q11` Interpretability of AI | mechanistic, interpretability, gemma, side-channel, sca | 3 | 2025-02..2026-09 | 0.16 | 3.88x |
| `ai2035:Q11` Interpretability of AI | mechanistic, interpretability, stream, statements, basis | 3 | 2026-05..2026-09 | 0.15 | 6.16x |
| `ai2035:Q12` Robustness and adversarial safety | adversarial, robustness, attacks, attack, defense | 35 | 2025-01..2026-11 | 0.45 | 17.22x |
| `ai2035:Q12` Robustness and adversarial safety | certified, robustness, smoothing, randomized, lipschitz | 33 | 2025-02..2026-11 | 0.36 | 12.02x |
| `ai2035:Q12` Robustness and adversarial safety | compression, compressed, pruning, robustness, certified | 4 | 2025-06..2025-11 | 0.22 | 9.80x |
| `ai2035:Q12` Robustness and adversarial safety | adversarial, robustness, perturbations, probabilistic, fing… | 4 | 2025-02..2025-09 | 0.17 | 4.07x |
| `ai2035:Q12` Robustness and adversarial safety | verification, certified, robustness, formally, neural | 3 | 2025-05..2026-07 | 0.16 | 2.25x |
| `ai2035:Q13` Data sharing and privacy protection | federated, clients, privacy, client, aggregation | 51 | 2025-01..2026-11 | 0.43 | 11.46x |
| `ai2035:Q13` Data sharing and privacy protection | membership, inference, attack, attacks, privacy | 55 | 2025-01..2026-11 | 0.42 | 16.63x |
| `ai2035:Q13` Data sharing and privacy protection | privacy, differential, varepsilon, gdp, membership | 3 | 2025-01..2026-09 | 0.38 | 6.42x |
| `ai2035:Q13` Data sharing and privacy protection | federated, generic, communicate, python, centralized | 3 | 2025-06..2025-11 | 0.21 | 8.91x |
| `ai2035:Q14` Fairness and value alignment | fairness, algorithmic, fair, bias, groups | 29 | 2025-01..2026-11 | 0.34 | 7.68x |
| `ai2035:Q14` Fairness and value alignment | monitoring, runtime, fairness, algorithmic, specification | 4 | 2025-01..2026-08 | 0.18 | 8.19x |
| `ai2035:Q14` Fairness and value alignment | fairness, justice, distributive, algorithmic, definitions | 5 | 2025-04..2026-09 | 0.18 | 3.85x |
| `ai2035:Q15` Uncertainty quantification and trustworthy decisions | conformal, prediction, coverage, uncertainty, sets | 60 | 2025-01..2026-11 | 0.47 | 16.64x |
| `ai2035:Q16` Brain-inspired intelligence and low-energy computing | spiking, snns, neural, neuromorphic, networks | 90 | 2025-01..2026-11 | 0.39 | 7.59x |
| `ai2035:Q16` Brain-inspired intelligence and low-energy computing | spiking, neurons, neuron, neuromorphic, integrate-and-fire | 10 | 2025-03..2026-10 | 0.27 | 8.47x |
| `ai2035:Q16` Brain-inspired intelligence and low-energy computing | spiking, plasticity, spike-timing-dependent, synaptic, spike | 7 | 2025-04..2026-09 | 0.24 | 5.13x |
| `ai2035:Q16` Brain-inspired intelligence and low-energy computing | recurrent, temporal, rsnns, credit, spiking | 3 | 2025-05..2026-03 | 0.18 | 5.95x |
| `ai2035:Q16` Brain-inspired intelligence and low-energy computing | tracking, spiking, rgb, uav, cameras | 3 | 2025-03..2026-03 | 0.17 | 13.12x |
| `ai2035:Q16` Brain-inspired intelligence and low-energy computing | neuromorphic, computing, hiaer-spike, routing, synapses | 5 | 2025-02..2026-02 | 0.17 | 5.35x |
| `ai2035:Q16` Brain-inspired intelligence and low-energy computing | neuroai, dynamical, neuromorphic, artificial, systems | 3 | 2025-09..2026-04 | 0.16 | 2.36x |
| `ai2035:Q17` Embodied intelligence and world models | embodied, intelligence, world, perception, environments | 52 | 2025-01..2026-11 | 0.38 | 4.56x |
| `ai2035:Q17` Embodied intelligence and world models | world, video, planning, latent, interactive | 38 | 2025-03..2026-11 | 0.23 | 9.87x |
| `ai2035:Q17` Embodied intelligence and world models | safe, offline, reinforcement, online, policies | 3 | 2026-03..2026-06 | 0.23 | 3.12x |
| `ai2035:Q18` Multi-agent cooperation and collective intelligence | communication, multi-agent, reinforcement, emergent, inter-… | 8 | 2025-04..2026-10 | 0.28 | 3.24x |
| `ai2035:Q18` Multi-agent cooperation and collective intelligence | multi-agent, marl, reinforcement, decentralized, agents | 16 | 2025-01..2026-09 | 0.28 | 1.89x |
| `ai2035:Q18` Multi-agent cooperation and collective intelligence | swarm, swarms, lifelong, controllers, multi-robot | 3 | 2025-03..2026-11 | 0.25 | 1.78x |
| `ai2035:Q18` Multi-agent cooperation and collective intelligence | traffic, multi-agent, reinforcement, distributionally, cont… | 6 | 2025-11..2026-11 | 0.19 | 2.38x |
| `ai2035:Q18` Multi-agent cooperation and collective intelligence | social, game, cooperative, collective, dilemmas | 4 | 2025-01..2026-01 | 0.19 | 3.65x |
| `ai2035:Q18` Multi-agent cooperation and collective intelligence | social, multi-agent, reinforcement, marl, observable | 3 | 2025-05..2025-10 | 0.17 | 1.72x |
| `ai2035:Q18` Multi-agent cooperation and collective intelligence | reinforcement, environment, encountered, multi-robot, solve | 3 | 2025-01..2025-07 | 0.15 | 1.60x |
| `ai2035:Q19` Human-machine hybrid augmented intelligence | human-ai, collaboration, human, collaborative, humans | 32 | 2025-01..2026-10 | 0.35 | 8.23x |
| `ai2035:Q19` Human-machine hybrid augmented intelligence | defer, deferral, expert, experts, human | 21 | 2025-02..2026-11 | 0.30 | 5.74x |
| `ai2035:Q19` Human-machine hybrid augmented intelligence | human-ai, recommendations, collaboration, perceptions, human | 4 | 2025-05..2026-06 | 0.22 | 4.94x |
| `ai2035:Q19` Human-machine hybrid augmented intelligence | genai, verification, human-ai, collaboration, legal | 3 | 2025-08..2026-04 | 0.17 | 5.73x |
| `ai2035:Q19` Human-machine hybrid augmented intelligence | interview, interviews, human-ai, preparation, collaboration | 4 | 2025-03..2026-05 | 0.16 | 4.22x |
| `ai2035:Q20` AI for science: discovering laws from data | symbolic, regression, expressions, equations, expression | 54 | 2025-03..2026-11 | 0.45 | 14.64x |
| `ai2035:Q20` AI for science: discovering laws from data | scientific, science, discovery, scientists, disciplines | 12 | 2025-03..2026-10 | 0.25 | 4.98x |
| `ai2035:Q20` AI for science: discovering laws from data | grids, power, bus, symbolic, regression | 5 | 2025-04..2026-11 | 0.15 | 1.60x |
| `hfm:ai4x_materials_gen` Crystal Generation for Unseen Chemical Systems with MatterG… | molecular, chemical, molecules, drug, science | 7 | 2025-03..2026-07 | 0.17 | 5.75x |
| `hfm:live_post_fabner_auto` FabNER Material Manufacturing Entity Extraction | extraction, few-shot, entity, information, nlp | 3 | 2025-03..2025-09 | 0.16 | 3.44x |
| `hfm:live_post_mapqa_auto` MapQA Map Visual Question Answering | document, docvqa, ocr, documents, chinese | 4 | 2025-02..2026-04 | 0.20 | 6.19x |
| `hfm:live_post_synlogic_auto` SynLogic logical reasoning training | neuro-symbolic, logical, symbolic, reasoning, neuro | 11 | 2025-01..2026-07 | 0.29 | 6.00x |
| `hfm:live_post_topv2_fixed_auto` TOPv2 compositional semantic parsing | compositional, generalization, clip, compositions, combinat… | 21 | 2025-01..2026-07 | 0.21 | 2.46x |
| `hfm:rsi_agent_harness_opus5` Agent Harness Evolution Under a Frozen API Model | harness, harnesses, agent, herd, continual | 4 | 2026-10..2026-10 | 0.30 | 5.26x |
| `hfm:rsi_wind_wake_calibration_opus5` Wind-Farm Wake Calibration and Power Prediction from Real S… | conformal, prediction, coverage, uncertainty, sets | 60 | 2025-01..2026-11 | 0.19 | 1.88x |
| `maths:m08` Twin primes | twins, digital, twin, dating, cognitive | 5 | 2025-02..2026-04 | 0.17 | 28.83x |

## Neglected problems

In a field the corpus covers, with no direction plainly about them. The nearest direction is shown so each verdict can be checked.

| Problem | Nearest direction | Similarity | Why not linked |
| --- | --- | --- | --- |
| `ai100:001` Are pre-training scaling laws slowing down, and where is the next sou… | test-time, compute, scaling, reasoning | 0.14 | under the floor |
| `ai100:004` Can synthetic data take over as high-quality human data runs out? | preference, alignment, aligning, dpo | 0.08 | under the floor |
| `ai100:006` Is a long context the same as real long-term memory? | associative, memory, hopfield, retrieval | 0.11 | under the floor |
| `ai100:010` How far are we from AGI, and how should it be defined? | cognitive, workspace, artificial, intelligence | 0.19 | another problem claims it, or too close to call |
| `ai100:011` Does chain-of-thought faithfully reflect the model's actual reasoning? | cot, chain-of-thought, reasoning, obfuscation | 0.11 | under the floor |
| `ai100:015` Can models complete long-horizon tasks spanning days or weeks? | skill, compositional, skills, long-horizon | 0.09 | under the floor |
| `ai100:018` Can AI independently prove important new theorems? | neuro-symbolic, logical, symbolic, reasoning | 0.09 | under the floor |
| `ai100:020` Is model creativity the same kind of thing as human creativity? | reflection, workflow, creative, cognitive | 0.12 | under the floor |
| `ai100:021` How do agents become reliable enough to deploy in real environments? | experience, continual, skills, agents | 0.13 | under the floor |
| `ai100:022` How should long-running autonomous agents recover from errors? | attribution, multi-agent, mas, llm | 0.11 | under the floor |
| `ai100:024` How much authority should agents be given, and how are permission bou… | traffic, multi-agent, reinforcement, distribution… | 0.08 | under the floor |
| `ai100:025` When will computer use be as fluent as a person's? | experience, continual, skills, agents | 0.07 | under the floor |
| `ai100:027` How will agent protocol standards such as MCP evolve? | communication, multi-agent, reinforcement, emerge… | 0.15 | another problem claims it, or too close to call |
| `ai100:028` Can AI complete whole software engineering projects autonomously? | topology, topologies, multi-agent, credit | 0.09 | under the floor |
| `ai100:029` Who is liable when an agent's mistake causes loss? | legal, fairness, algorithmic, discrimination | 0.15 | under the floor |
| `ai100:030` How do we tell an agent is actually useful, not just good on benchmar… | counterfactual, explanations, erasure, explanation | 0.06 | under the floor |
| `ai100:031` How do we keep increasingly capable models aligned with human intent? | preference, alignment, aligning, dpo | 0.09 | under the floor |
| `ai100:032` How can humans oversee systems smarter than themselves? | human-ai, collaboration, human, collaborative | 0.07 | under the floor |
| `ai100:033` Do models deceive, scheme or hide goals? | preference, alignment, aligning, dpo | 0.07 | under the floor |
| `ai100:035` Do models behave the same in evaluation and deployment? | cognitive, embodied, reactive, autonomy | 0.06 | under the floor |
| `ai100:037` How are dangerous biological, chemical and cyber capabilities assesse… | molecular, chemical, molecules, drug | 0.09 | under the floor |
| `ai100:038` How large is the risk of autonomous replication and self-improvement? | invariant, irm, minimization, risk | 0.07 | under the floor |
| `ai100:039` How likely are loss-of-control scenarios, and how much should be spen… | losses, erm, regret, minimization | 0.08 | under the floor |
| `ai100:040` Who should write the constitution and values of AI? | monitoring, runtime, fairness, algorithmic | 0.09 | under the floor |
| `ai100:041` Can we really read what a model is thinking internally? | mechanistic, interpretability, saes, circuit | 0.31 | another problem claims it, or too close to call |
| `ai100:043` Can lying or harmful intent be detected from internal activations? | internal, reward-hacking, reward-hack, monitoring | 0.13 | under the floor |
| `ai100:044` Are features and circuits the right units for understanding models? | mechanistic, interpretability, saes, circuit | 0.16 | another problem claims it, or too close to call |
| `ai100:045` Can interpretability be used to edit and correct model behaviour dire… | patching, mechanistic, interpretability, activati… | 0.15 | under the floor |
| `ai100:046` How far can a model's introspective self-reports be trusted? | cognitive, cognition, metacognition, architecture | 0.10 | under the floor |
| `ai100:047` How do we explain AI decisions to ordinary users? | counterfactual, explanations, erasure, explanation | 0.13 | under the floor |
| `ai100:091` Are current benchmarks saturated or contaminated? | mllms, multimodal, reasoning, understanding | 0.06 | under the floor |
| `ai100:092` How is real economic value measured rather than leaderboard scores? | personality, personalities, traits, emotional | 0.09 | under the floor |
| `ai100:093` How are open-ended, subjective tasks evaluated? | agents, libraries, paradigm, artifacts | 0.09 | under the floor |
| `ai100:094` How are a model's risk capabilities measured systematically? | jailbreak, safety, attacks, harmful | 0.09 | under the floor |
| `ai100:095` Can AI accelerate AI research itself? | high-cost, lab, low-cost, agent | 0.09 | under the floor |
| `ai100:096` Can capability gains be forecast and warned of early? | spatio-temporal, forecasting, urban, graph | 0.07 | under the floor |
| `ai100:097` How can AI research be made more reproducible and transparent? | scientific, science, discovery, scientists | 0.10 | under the floor |
| `ai100:098` How are third-party audits and red teaming institutionalised? | neuroai, dynamical, neuromorphic, artificial | 0.07 | under the floor |
| `ai100:100` What future do we want to create together with AI? | cognitive, embodied, reactive, autonomy | 0.04 | under the floor |
| `ai2035:Q02` The mathematics of deep learning: approximation, optimisation, implic… | plasticity, continual, loss, deep | 0.13 | under the floor |
| `hfm:embspatial` Training Recipes for Qwen2-VL-2B on EmbSpatial-Bench Embodied Spatial… | embodied, intelligence, world, perception | 0.12 | under the floor |
| `hfm:exp_agentic_pretrain` Agentic Pretraining with a Frozen Corpus and Tokenizer | pretraining, next-token, llm, agrees | 0.15 | another problem claims it, or too close to call |
| `hfm:exp_agentic_pretrain_research` Pretraining architecture development from scratch | pretraining, next-token, llm, agrees | 0.16 | another problem claims it, or too close to call |
| `hfm:exp_agentic_pretrain_v2` Agentic Pretraining Under Fixed Inference-Cost Constraint | pretraining, next-token, llm, agrees | 0.15 | another problem claims it, or too close to call |
| `hfm:exp_agentic_rl_k3` Agentic RL Training Recipe Exploration with verl and agent-harness | verifiable, rewards, rlvr, reasoning | 0.06 | under the floor |
| `hfm:exp_looped_pretrain` Looped-Transformer Pretraining Under a Fixed Compute Budget | pretraining, next-token, llm, agrees | 0.12 | under the floor |
| `hfm:exp_looped_pretrain_research_v1` Looped Transformer Pretraining with Recurrent Depth Architectures | pretraining, next-token, llm, agrees | 0.11 | under the floor |
| `hfm:exp_looped_pretrain_research_v2` Looped pretraining with two concurrent four-GPU ideas per batch | pretraining, next-token, llm, agrees | 0.10 | under the floor |
| `hfm:exp_pretrain_proposals_20261110` Pretraining architecture development on frozen public training pool | pretraining, next-token, llm, agrees | 0.15 | under the floor |
| `hfm:live_harness_formal_proofs` Agent for Formal Theorem Proving in Lean | hopfield, machines, networks, energy-based | 0.09 | under the floor |
| `hfm:live_harness_grid_reasoning` Agent for ARC Grid Transformation Tasks | operator, transformation, navier-stokes, deeponet | 0.11 | under the floor |
| `hfm:live_harness_paper_memory` Building an Agent for Preserving Knowledge from Research Papers | unlearning, preserving, selective, relearning | 0.07 | under the floor |
| `hfm:live_harness_table_analysis` Agent for Questions over Real Tables | high-cost, lab, low-cost, agent | 0.09 | under the floor |
| `hfm:live_harness_verified_programming` Agent for Verified Programming with Dafny | verification, certified, robustness, formally | 0.08 | under the floor |
| `hfm:live_post_scienceworld_auto` ScienceWorld task training on Qwen2.5-7B-Instruct | examples, hard, easy, post-training | 0.06 | under the floor |

## Outside the corpus

98 unmatched problems are in fields this corpus does not cover. That is an absence of data, not a finding that nobody works on them.

- `ai-governance`: 10 (How should frontier AI be regulated, and by…; How can national AI rules (EU, US, China) be…; How will US-China AI competition evolve, and…; …)
- `ai-infrastructure`: 10 (Can compute demand keep growing exponentiall…; How is data-centre energy use met; can nucle…; How are water, carbon and other environmenta…; …)
- `cosmology`: 10 (Dark matter; Is dark energy constant; The Hubble tension; …)
- `global`: 10 (Climate change; War; Extreme poverty; …)
- `ai-nature`: 9 (Is AI conscious, or could it become so?; Could AI have welfare and deserve moral cons…; Do internal states correspond to a model's e…; …)
- `optimization`: 9 (Citywide Bicycle Rebalancing Policy for Stat…; Columbia and Snake River Dam Spill Optimizat…; Control Policy for Frequent Bus Services; …)
- `physics`: 9 (64-Frame Rayleigh–Bénard Convection Forecast…; Battery Voltage Model for Cold Operating Con…; Estimating Wave-Energy Resources from Short…; …)
- `mathematics`: 9 (Riemann hypothesis; P vs NP; Navier-Stokes; …)
- `ai-society`: 8 (How many jobs will AI replace, and which sec…; Will AI productivity gains show up in GDP?; How will the gains be distributed, and will…; …)
- `quant`: 8 (Cost-aware long-only portfolio construction…; Cross-Venue Allocation of Perpetual Orders; Event-Driven Equity Strategy Based on Financ…; …)
- `bio`: 2 (Cross-Study Cell-Type Annotation with UCE on…; Predicting human regulatory DNA activity wit…)
- `astro`: 1 (Galaxy Photometric Redshift Prediction Acros…)
- `earth`: 1 (24-Hour Global Near-Surface Weather Forecast…)
- `infra`: 1 (Single-Machine Weighted Tardiness Scheduling…)
- `medical`: 1 (Patient-level lung cancer histology classifi…)

## Orphan directions (largest 25)

Active directions no catalogued problem claims. Either the catalogues have a gap, or the direction's vocabulary differs from the problem statement's.

| Direction (distinctive terms) | Works | Active | Cohesion | Nearest problem | Similarity |
| --- | --- | --- | --- | --- | --- |
| potentials, interatomic, machine-learning, mlips, materials… | 47 | 2025-01..2026-11 | 0.12 | `cosmos:p04` Which inflation model, if any | 0.11 |
| stability, edge, gradient, descent, sharpness, eos | 43 | 2025-01..2026-11 | 0.13 | `ai2035:Q02` The mathematics of deep learning: appro… | 0.09 |
| preference, alignment, aligning, dpo, preferences, human | 31 | 2025-01..2026-11 | 0.15 | `ai2035:Q14` Fairness and value alignment | 0.13 |
| vla, vision-language-action, manipulation, robotic, action,… | 17 | 2025-03..2026-11 | 0.12 | `ai100:007` What are the limits of unified multimod… | 0.13 |
| computing, reservoir, neuromorphic, photonic, optical, power | 10 | 2025-01..2026-07 | 0.15 | `ai2035:Q16` Brain-inspired intelligence and low-ene… | 0.12 |
| attribution, multi-agent, mas, llm, systems, failures | 10 | 2025-04..2026-10 | 0.13 | `ai100:023` How should multi-agent systems divide w… | 0.14 |
| learning-to-defer, expert, experts, one-stage, mathcal, sur… | 9 | 2025-02..2026-05 | 0.17 | `maths:m10` abc | 0.08 |
| icl, in-context, transformers, context, number, diversity | 8 | 2025-08..2026-08 | 0.16 | `ai2035:Q17` Embodied intelligence and world models | 0.06 |
| multilingual, languages, hallucination, translation, detect… | 8 | 2025-03..2026-02 | 0.17 | `ai100:013` Can hallucination be eliminated at the… | 0.13 |
| memristors, resistive, switching, devices, computing, neuro… | 8 | 2025-05..2026-10 | 0.19 | `ai2035:Q16` Brain-inspired intelligence and low-ene… | 0.07 |
| navigation, underwater, autonomous, imu, robot, motion | 7 | 2025-07..2026-11 | 0.13 | `ai100:076` Where are the limits of AI use in the m… | 0.09 |
| muon, optimizer, adam, adamw, optimizers, stability | 7 | 2025-04..2026-11 | 0.16 | `ai2035:Q02` The mathematics of deep learning: appro… | 0.05 |
| mllms, multimodal, reasoning, understanding, visual, embodi… | 7 | 2025-01..2026-04 | 0.15 | `ai100:007` What are the limits of unified multimod… | 0.13 |
| graph, gnns, graphs, heterophilic, node, graph-structured | 7 | 2025-04..2025-12 | 0.13 | `ai2035:Q12` Robustness and adversarial safety | 0.07 |
| prompts, few-shot, vision-language, prompt, visual, tuning | 7 | 2025-03..2026-05 | 0.16 | `ai2035:Q06` Learning from few samples and few labels | 0.13 |
| cognitive, embodied, reactive, autonomy, architecture, agen… | 6 | 2025-03..2026-08 | 0.13 | `ai2035:Q01` Cognition and computation: the computat… | 0.13 |
| recommendation, multi-behavior, user, preference, invariant… | 6 | 2025-07..2026-07 | 0.16 | `ai2035:Q08` Causality-driven machine learning | 0.14 |
| dexterous, manipulation, hand, contact, task-specific, robo… | 6 | 2025-11..2026-10 | 0.16 | `ai100:060` How should AI treat users' autonomy and… | 0.09 |
| review, survey, aiming, comprehensive, multimodal, continual | 6 | 2025-02..2025-10 | 0.14 | `ai100:007` What are the limits of unified multimod… | 0.09 |
| skill, compositional, skills, long-horizon, generalization,… | 6 | 2025-09..2026-08 | 0.17 | `ai100:067` Does over-reliance on AI erode human sk… | 0.19 |
| neuromorphic, computing, magnetic, spintronic, spin, torque | 6 | 2025-08..2026-05 | 0.16 | `ai2035:Q16` Brain-inspired intelligence and low-ene… | 0.15 |
| unlearning, preserving, selective, relearning, remove, forg… | 5 | 2025-06..2026-10 | 0.16 | `ai100:003` Can reinforcement learning drive fast p… | 0.10 |
| cross-lingual, sentiment, arabic, few-shot, languages, mono… | 5 | 2025-05..2026-10 | 0.14 | `ai2035:Q06` Learning from few samples and few labels | 0.11 |
| quantization, bit, ptq, post-training, bits, weight | 5 | 2025-11..2026-11 | 0.17 | `ai100:008` How small can distillation and small mo… | 0.13 |
| quantum, qml, adversarial, classical, robustness, supercond… | 5 | 2025-05..2026-05 | 0.20 | `ai2035:Q12` Robustness and adversarial safety | 0.13 |

73 smaller orphan directions not shown.

## Refused

6 direction-catalogue pairs cleared the floor but not the margin: the direction sits between two problems and was linked to neither. 69 more fell just under the floor.

- trading, market, financial, price -> `hfm:live_quant_ashare_intraday` at 0.33: only 1.17x the runner-up (hfm:live_quant_perpetual_minute_strategy), needing 1.25x
- cognitive, workspace, artificial, intelligence -> `ai100:051` at 0.21: only 1.11x the runner-up (ai100:010), needing 1.25x
- patching, mechanistic, interpretability, activati… -> `ai100:041` at 0.20: only 1.01x the runner-up (ai100:042), needing 1.25x
- mechanistic, interpretability, stream, statements -> `ai100:042` at 0.20: only 1.08x the runner-up (ai100:041), needing 1.25x
- skill, compositional, skills, long-horizon -> `ai100:067` at 0.19: only 1.11x the runner-up (ai100:005), needing 1.25x
- pretraining, next-token, llm, agrees -> `hfm:exp_agentic_pretrain_research` at 0.16: only 1.08x the runner-up (hfm:exp_agentic_pretrain_v2), needing 1.25x

## Limitations

- Matching is lexical. A problem stated in different words from the papers working on it is reported neglected, and a direction sharing a method's vocabulary with a problem can be linked to it wrongly. Read a neglected verdict as "no direction in this corpus uses this problem's words", not "nobody is working on it".
- The default corpus is not a sample of machine learning. It was harvested with search queries for the 20 ai2035 questions, so an AI-100 or Human Frontier problem with no direction may simply be one nobody queried for. A corpus harvested without reference to any catalogue is what a real neglect claim needs.
- The AI-100 catalogue is in Chinese. Its English rendering and matching terms (`data/nexus/matching/ai100_en.yaml`) were written by the model, not taken from the source and not reviewed; a poor gloss produces a false neglected verdict.
- The measured scores come from one corpus in one field, whose papers were found by lexical search in the first place. The matcher has never been tested on cosmology, mathematics or global-development literature.
- The floor and margin are judgement calls; the floor was tuned on six questions. Which fields a corpus covers is declared by hand; a vocabulary-overlap test was tried and could not tell cosmology from machine learning.
- Catalogues differ in how much text they give: Human Frontier challenges are matched on a title alone, worldsim problems on a name and outcome labels. Short text matches badly.
- Science's 125 questions, discussed elsewhere in this project, are not in the repository and are not matched.
- "Active" means present in the corpus. No growth rate is computed here; run `sciscout rank` on the same directions for momentum and priority.
