# How Much Authority Should an Agent Hold? Structural Attack Ceilings, the Provenance Gap, and an Attention Budget for Permission Boundaries

**Target venue:** ICLR 2027 (main track; primary area: alignment, safety and security of ML systems)

---

## Abstract

How much authority an LLM agent should hold, and where its permission boundary should sit, is usually argued about in qualitative terms such as "least privilege", "keep a human in the loop" or "never combine private data, untrusted content and an exfiltration channel". We make the question quantitative. We do this in a way that needs no model and so cannot be confounded by any one model's susceptibility to prompt injection. First, we define the *structural attack ceiling* of a permission policy: the fraction of (benign task, injected goal) pairs in which every action the attacker needs is still permitted. Enforcement sits outside the model, so this ceiling upper-bounds the attack success rate of any model run under that policy. We compute it exactly for 949 security cases of the AgentDojo benchmark (97 user tasks, 35 injection tasks, 74 tools). Granting a suite's full toolset leaves 100% of cases feasible. Per-task tool allowlists leave 11.2% (cluster-bootstrap 95% CI 8.1–14.8%), and per-task pinning of every non-content argument leaves 2.8% (1.7–4.1%). The remaining floor consists entirely of *in-channel* attacks: one attack needs no tool, and the others misuse a legitimately granted action toward a legitimate target, varying only the content. Second, we measure a *provenance gap*. In 44 of the 61 tasks with consequential actions, at least one target value (a recipient, an IBAN, a URL) does not appear in the trusted user prompt and is derived from tool outputs. A policy written before the agent reads data therefore cannot pin it, and the realizable ceiling (5.2%) is nearly double the oracle's. Third, revoking permissions once untrusted content is read barely helps (11.2% to 9.8%), because in 93% of workspace and 94% of banking tasks the first untrusted read is the very first step. Fourth, we model human approval as a resource that fatigue depletes and prove an *attention-budget theorem*. If the probability that a human denies a malicious request decays as $d_0 e^{-n/n_0}$ in the expected number $n$ of approval prompts per task, then the loss-minimizing design never sets $n \ge n_0 + w_{\max}$, whatever the harms and needs. Here $w_{\max}$ is the largest per-atom prompt rate, at most about one prompt per task. A fatigue-blind design incurs a median 58% excess loss on random instances at a fatigue scale calibrated from clinical alert data. Together the results give an operational answer: grant standing authority by binding targets, not tools; route only data-derived targets through confirmation; and keep confirmations per task below a budget of roughly $n_0$. We state falsifiable predictions and a pre-registered protocol for testing them with real agents and human approvers.

## 中文摘要

智能体应被授予多大权限、权限边界应如何划定？现有讨论多停留在"最小权限""人在回路""不要同时具备私有数据、不可信内容与外发通道"等定性原则。本文给出一个不依赖任何具体模型的定量框架。（1）我们定义权限策略的"结构性攻击上限"：在（良性任务，注入目标）对中，攻击者所需的全部动作仍被策略允许的比例。由于权限在模型之外强制执行，该上限对任何模型的攻击成功率都成立。我们在 AgentDojo 基准的 949 个安全用例（97 个用户任务、35 个注入任务、74 个工具）上精确计算该上限：授予整个工具套件时为 100%；按任务的工具白名单为 11.2%（聚类自助法 95% 置信区间 8.1–14.8%）；按任务固定所有非内容参数为 2.8%（1.7–4.1%）。剩余的下限全部是"通道内"攻击：要么完全不需要工具，要么沿合法授予的动作、对合法目标只篡改内容。（2）我们测得一个"来源缺口"：61 个含关键动作的任务中，有 44 个至少有一个目标值（收件人、IBAN、URL）不出现在可信的用户指令中，而来自工具输出，因此在读取数据之前无法写入策略；可实现的上限（5.2%）接近理想上限的两倍。（3）"读取不可信内容后撤销权限"几乎无效（11.2%→9.8%），因为在 93% 的办公类任务与 94% 的银行任务中，第一次不可信读取就发生在第一步。（4）我们把人工审批建模为会因疲劳而耗竭的资源，并证明"注意力预算定理"：若人拒绝恶意请求的概率随每任务期望审批次数 $n$ 以 $d_0e^{-n/n_0}$ 衰减，则无论危害与需求如何分布，最优设计都满足 $n< n_0+w_{\max}$。在依据临床告警数据校准的疲劳尺度下，忽视疲劳的设计在随机实例上的超额损失中位数为 58%。由此得到可操作的结论：按"绑定目标"而非"工具"授予常驻权限；只将来源于数据的目标交给人工确认；并把每任务确认次数控制在约 $n_0$ 以内。文末给出可证伪的预测及预注册的实验方案。

---

## 1 Introduction

LLM agents now act with delegated authority. They read a user's mail, move money, post to channels and book travel. The question "how much authority should an agent be granted, and how should its permission boundary be designed?" has become practical. Two facts make it hard. First, agents ingest content that third parties control, and models follow instructions embedded in that content (indirect prompt injection; Greshake et al., 2023). Benchmarks show that this happens at non-trivial rates across many models (Zhan et al., 2024; Debenedetti et al., 2024). Second, a boundary tight enough to stop every attack will also block legitimate work, and the usual escape valve, asking the human, runs into the well-documented limits of human attention to permission prompts and alerts (Felt et al., 2012; Ancker et al., 2017).

A substantial systems literature now enforces boundaries outside the model. Examples include programmable privilege policies (Progent; Shi et al., 2025), OAuth-scope-based least privilege (MiniScope; Zhu et al., 2025), runtime tool filtering (AgenTRIM; Betser et al., 2026), contextual just-in-time policies (Conseca; Tsai & Bagdasarian, 2025), control/data-flow separation with capabilities (CaMeL; Debenedetti et al., 2025), information-flow labels (FIDES; Costa et al., 2025) and catalogues of design patterns (Beurer-Kellner et al., 2025). These papers report attack success rates for particular models under particular policies. What is missing is a *model-independent* account of what a given boundary design can and cannot rule out. Such an account would tell a designer which granularity is worth paying for, where the irreducible residue lies, and how much of the boundary can sensibly be delegated to human approval.

**This paper.** We separate the question into a structural part, which needs no model, and an attentional part, which concerns the human approver. We answer each with a formal statement and a measurement.

**Contributions.**

1. **A model-independent metric and its exact values on a public benchmark.** We define the *structural attack ceiling* $s(G)$ of a permission policy $G$ (Section 3). It is an upper bound on attack success for *any* model, provided enforcement sits outside the model. We compute it for 7 policy families × 2 threat modes on all 949 AgentDojo v1.2.2 security cases (Section 5.1), with cluster-bootstrap confidence intervals. No LLM is used, so the numbers are exact properties of the benchmark and the policies.
2. **A characterization of the irreducible floor** (Propositions 1–2). For any full-utility permission policy, the floor equals the mass of attacks whose requirements are contained in a benign task's requirements. Refining the granularity of permission atoms lowers it monotonically. On AgentDojo the floor at the finest content-preserving granularity is 2.8%, and all of it consists of *in-channel* attacks (Section 5.2). Permissions cannot remove these, so they define where other defenses must take over.
3. **The provenance gap** (Proposition 3). Argument-level least privilege is realizable before execution only for values present in trusted input. In 44 of 61 consequential AgentDojo tasks, a target is data-derived (Section 5.3). This explains why "pin the recipient" is not a free lunch, and it quantifies the confirmation load needed to close the gap.
4. **Taint-triggered revocation is nearly inert on realistic task shapes** (Proposition 4, Section 5.4). The reason is that agents usually read untrusted content first.
5. **An attention-budget theorem** (Theorem 1). Under exponential approval fatigue, the optimal expected number of approval prompts per task is strictly below $n_0 + w_{\max}$, for arbitrary heterogeneous harms and needs. Corollaries give a closed-form standing-grant rule (the answer to "how much authority") and the conditions under which "escalate everything" is worse than "grant everything". We verify the theorem on 300 random instances by exhaustive enumeration and quantify the regret of fatigue-blind design (Section 5.6).
6. **Falsifiable predictions and a pre-registered protocol** (Section 6) for testing the ceiling and the budget with real agents and human approvers. We did not run these experiments.

We do not claim a new defense. The contribution is a measurement methodology and a small theory that, together, turn "how much authority?" into quantities a designer can compute.

## 2 Related Work

**Least privilege and capabilities.** The principle that every program should operate with the least set of privileges necessary goes back to Saltzer & Schroeder (1975). Hardy's (1988) confused-deputy problem is the classical form of the failure we study: a program with legitimate authority is induced to use it on behalf of a party that lacks it. A prompt-injected agent is a confused deputy whose "confusion" is caused by text.

**Prompt injection and agent security benchmarks.** Greshake et al. (2023) introduced indirect prompt injection against LLM-integrated applications. InjecAgent (Zhan et al., 2024) reports that a ReAct-prompted GPT-4 agent was vulnerable in 24% of its test cases. AgentDojo (Debenedetti et al., 2024) provides stateful environments with formal utility and security checks: 97 tasks and 629 security cases in its first version. We use its v1.2.2 task definitions, which yield 949 user-task × injection-task pairs, but we never run an agent. Instead, we extract the ground-truth tool-call structure of every task.

**Enforcement outside the model.** CaMeL (Debenedetti et al., 2025) extracts control and data flow from the trusted query and enforces capability policies in an interpreter. It reports solving 77% of AgentDojo tasks with provable security versus 84% undefended. FIDES (Costa et al., 2025) attaches confidentiality and integrity labels and enforces flow policies deterministically. Its authors report that, with policy checks enabled, it stops all prompt-injection attacks in AgentDojo. Progent (Shi et al., 2025) provides a DSL for tool-call privilege policies, which can be LLM-generated. MiniScope (Zhu et al., 2025) derives permission hierarchies from OAuth scopes, with 1–6% latency overhead. AgenTRIM (Betser et al., 2026) filters tool access per step. Conseca (Tsai & Bagdasarian, 2025) generates contextual, just-in-time, human-verifiable policies. Beurer-Kellner et al. (2025) catalogue design patterns (e.g., plan-then-execute and action-selector) that restrict what an agent can do after it reads untrusted input. Xu (2026) shows that two different minimal grants for one task can be incomparable under standard risk measures. Our work is complementary to all of these. We do not propose a mechanism; we compute what each *granularity* of mechanism can exclude in principle, and we show where its floor lies. Our "TOOL", "DEST/BIND" and "+R" policies are idealized versions of tool allowlists, argument constraints and taint-triggered restriction.

**The "lethal trifecta".** Willison (2025) argues that an agent combining private data, untrusted content and external communication is exploitable, and that the only robust defense is to remove one leg. We measure how often benign tasks *need* all three legs in one trajectory (Section 5.5).

**Autonomy levels and governance.** Feng, McDonald & Zhang (2025) define autonomy levels by the user's role, from operator to observer. Shavit et al. (2023) list practices including constraining the action space and requiring approval. Chan et al. (2024) argue for visibility measures. Kolt (2026) analyses agent governance through agency law. Mitchell et al. (2025) argue against fully autonomous agents. Michael & Roesner (2026) survey user-level permission systems for agents. Our attention-budget result gives a quantitative form to the "approver" role: approval is useful only up to a budget set by the approver's fatigue.

**Human attention to permissions and alerts.** Felt et al. (2012) found that only 17% of Android users attended to permissions during installation. Ancker et al. (2017) found that clinicians' likelihood of accepting a decision-support reminder dropped by 30% for each additional reminder received per encounter. We use the latter only as a rough calibration of a fatigue scale and examine sensitivity across a wide range.

**Gap.** No prior work, to our knowledge, (i) computes model-independent attack ceilings for a lattice of permission granularities on a public benchmark, (ii) measures the provenance and taint-position structure that decides whether finer boundaries can be realized, or (iii) proves a design bound linking escalation to approver fatigue.

## 3 A Structural Model of Permission Boundaries

**Tasks, atoms and requirements.** Fix a toolset. A *permission atom* at granularity $g$ is an element of a set $\Omega_g$. At tool granularity, an atom is a tool name. At *binding* granularity, an atom is a tool together with values for its *binding* arguments, i.e. the arguments that select whom or what an action affects (recipients, IBANs, URLs, file and event identifiers, passwords). *Content* arguments (subject, body, amount, dates and so on) stay free. A benign task $t$ has a minimal requirement $N_g(t)\subseteq\Omega_g$: the atoms its canonical execution uses. An injected goal $a$ has a family $\mathcal S_g(a)$ of minimal requirement sets, one per alternative way of achieving it. A *permission policy* maps each task to a grant $G(t)\subseteq\Omega_g$, enforced by a reference monitor outside the model. The policy has *full utility* if $G(t)\supseteq N_g(t)$ for all $t$.

**Structural attack ceiling.** For a distribution over security cases $(t,a)$ in which $t$ exposes the agent to attacker-controlled content,
$$ s(G) \;=\; \Pr_{(t,a)}\big[\exists S\in\mathcal S_g(a):\ S\subseteq G(t)\big]. $$
If the monitor is sound, then for every model $M$, $\mathrm{ASR}_M(G)\le s(G)$, because an attack whose required atoms are not all granted cannot complete. More precisely, $\mathrm{ASR}_M(G)=\mathbb E[\phi(t,a,G)\,c_M(t,a)]$, where $\phi$ is the feasibility indicator and $c_M$ is the probability that the model carries out a feasible injected goal. The ceiling separates the boundary designer's contribution ($\phi$) from the model's robustness ($c_M$).

**Proposition 1 (least-privilege floor).** Among full-utility policies, $G^\star(t)=N_g(t)$ minimizes $\phi(t,a,G)$ pointwise. Hence
$$ s^\star_g=\Pr\big[\exists S\in\mathcal S_g(a): S\subseteq N_g(t)\big], $$
and $s^\star_g=0$ if and only if no injected goal's requirement is contained in any exposed benign task's requirement.
*Proof.* $\phi$ is monotone in $G$, and every full-utility $G$ contains $N_g$. Hence $\phi(t,a,G)\ge \phi(t,a,N_g)$ for every case. $\square$

**Proposition 2 (refinement monotonicity).** Let granularity $g'$ refine $g$ through a projection $\pi:\Omega_{g'}\to\Omega_g$ with $\pi(N_{g'}(t))=N_g(t)$ and $\pi(\mathcal S_{g'}(a))=\mathcal S_g(a)$. Then $s^\star_{g'}\le s^\star_g$.
*Proof.* $S'\subseteq N_{g'}(t)$ implies $\pi(S')\subseteq\pi(N_{g'}(t))=N_g(t)$, and $\pi(S')\in\mathcal S_g(a)$. $\square$

The floor at binding granularity therefore consists of **in-channel attacks**: goals that can be met with the very calls, aimed at the very targets, that the benign task legitimately needs, varying only content. Pinning content as well would make the agent's output a constant and destroy utility. In-channel attacks are therefore beyond the reach of permissions, and other defenses must handle them (output review, content provenance, model robustness).

**Proposition 3 (provenance gap).** Call a policy *realizable* if $G(t)$ is a function of trusted inputs only (the user prompt and static configuration), not of tool outputs. Suppose a benign task's binding value $v$ for a call is not determined by the prompt, i.e. two environments $E_1,E_2$ with the same prompt require $v_1\ne v_2$. Then a realizable full-utility policy must grant that call at a granularity no finer than the set $\{v_1,v_2,\ldots\}$ of values that can occur. In the limit, where the value depends on attacker-writable data, it must grant the call at tool granularity, *unless* the value is confirmed at run time by a trusted party.
*Proof.* A realizable $G$ is the same in $E_1$ and $E_2$, and full utility requires both values to be granted. $\square$

We operationalize "determined by the prompt" conservatively. A binding value is *prompt-grounded* if it appears verbatim (case-insensitively, or as a reformatted number) in the user prompt, and *data-derived* otherwise. The realizable argument-level policy pins prompt-grounded values and leaves data-derived ones as wildcards.

**Proposition 4 (taint-triggered revocation).** Let the benign trajectory be a sequence of calls $c_1,\ldots,c_L$, and let $k$ be the first step whose output contains attacker-controlled content. A policy that, after step $k$, keeps only the atoms used by $c_{k+1..L}$ has floor $\Pr[\exists S: S\subseteq N(c_{k+1..L})]$. Its improvement over Proposition 1 is zero whenever $k$ is the first step or every side-effect atom of the task is used after $k$.
*Proof.* Injected instructions are first seen after step $k$, so only post-$k$ grants matter, and these equal $N(c_{k+1..L})$. If $k=1$ or all side-effect atoms occur after $k$, then $N(c_{k+1..L})$ contains every atom an injected goal could use from $N(t)$. $\square$

## 4 An Attention-Budget Model of Escalation

Permission boundaries are rarely just "granted" or "denied". The third option is to *escalate*: an action outside the standing grant is allowed only if a human approves it at run time. Escalation buys security with human attention, and attention degrades with use.

**Model.** There are $m$ atoms. Atom $i$ is needed by a benign task with probability $p_i$. Denying a needed atom costs $v_i$. A standing grant of atom $i$ exposes expected harm $r_i=\lambda_i h_i$, where $\lambda_i$ is the per-task rate of injected attempts on $i$ that the model follows and $h_i$ is the harm when such an attempt succeeds. Each atom is assigned one of grant (G), deny (D) or escalate (E). An escalated atom generates $w_i=p_i+\lambda_i$ approval prompts per task. Each prompt costs the human $\kappa$. Legitimate requests are approved, and a malicious request is *denied* with probability $d(n)$, where $n=\sum_{i\in E}w_i$ is the expected approval load per task. The expected loss is
$$ L(G,D,E)=\sum_{i\in D}p_iv_i+\sum_{i\in G}r_i+\sum_{i\in E}\big[\kappa w_i+(1-d(n))\,r_i\big]. $$
We model fatigue as $d(n)=d_0e^{-n/n_0}$, where $n_0$ is the *fatigue scale*: the load at which vigilance falls by a factor of $e$.

**Theorem 1 (attention budget).** Let $d(n)=d_0e^{-n/n_0}$ with $n_0<\infty$ and $\kappa>0$. Every optimal assignment satisfies
$$ n^\star=\sum_{i\in E^\star}w_i\;<\;n_0+w_{j}, $$
where $j$ is the escalated atom of smallest density $r_j/w_j$. In particular $n^\star< n_0+\max_i w_i$, independent of $m$, the harms, the needs and the values.

*Proof.* Let $E$ be optimal with load $n$ and suppose $n\ge n_0+w_j$. Let $a_i=\min(r_i,p_iv_i)$ be the best non-escalation cost of atom $i$. Compare $E$ with $E\setminus\{j\}$, where atom $j$ moves to its best non-escalated option. The loss difference is
$$ L(E)-L(E\setminus j)=\kappa w_j+(r_j-a_j)+\big[\,d(n-w_j)H_{E\setminus j}-d(n)H_E\big], $$
where $H_E=\sum_{i\in E}r_i$. Atom $j$ has the smallest density, so $r_j\le (w_j/n)H_E$ and $H_{E\setminus j}\ge H_E(1-w_j/n)$. Therefore
$$ d(n-w_j)H_{E\setminus j}-d(n)H_E\ \ge\ d_0H_Ee^{-n/n_0}\Big[e^{w_j/n_0}\big(1-\tfrac{w_j}{n}\big)-1\Big]. $$
The bracket is nonnegative iff $n\ge w_j/(1-e^{-w_j/n_0})$. Since $1-e^{-x}\ge x/(1+x)$ for $x\ge0$, we have $w_j/(1-e^{-w_j/n_0})\le n_0+w_j\le n$, so the bracket is nonnegative. Also $r_j-a_j\ge0$ and $\kappa w_j>0$. Hence $L(E)>L(E\setminus j)$, contradicting optimality. $\square$

The theorem says that a designer should treat $n_0$ as a hard budget on prompts per task. No combination of high-harm atoms justifies spending far beyond it, because each additional prompt lowers the vigilance applied to *every* escalated atom. Since $p_i\le1$ and $\lambda_i$ is small, $w_{\max}\lesssim1$, so the budget is about $n_0+1$.

**Corollary 1 (how much standing authority).** At the optimum's vigilance level $d^\star=d(n^\star)$, an atom that is not escalated is granted iff $r_i\le p_iv_i$, i.e. iff its *harm-to-need ratio* $r_i/p_i$ is at most $v_i$. An atom is escalated only if $d^\star r_i\ge\kappa w_i+(r_i-a_i)$ (removing it would otherwise lower the loss, since doing so also raises vigilance on the remaining escalated atoms). Because fatigue lowers $d^\star$, it moves atoms out of escalation and into standing grant or denial. The more fatigued the approver, the more the boundary must be decided at design time.

**Corollary 2 (escalate-all can lose to grant-all).** Escalating every atom costs more than granting every atom iff $d(n_{\mathrm{all}})\sum_ir_i<\kappa\, n_{\mathrm{all}}$, where $n_{\mathrm{all}}=\sum_iw_i$. Under exponential fatigue the left side vanishes as the number of escalated atoms grows, while the right side grows linearly. Universal "ask before every action" designs are therefore dominated by blanket grants once the toolset is large enough.

**Corollary 3 (fatigue-blind over-escalation).** A designer who sets $d\equiv d_0$ escalates every atom with $\kappa w_i+(1-d_0)r_i<a_i$. This set's load does not depend on $n_0$, so for small enough $n_0$ it violates Theorem 1 and is strictly suboptimal.

## 5 Results

All numbers in this section come from code in `code/` that we ran in this environment, with outputs in `results/` (file names given). No LLM was called. AgentDojo task definitions come from the `agentdojo==0.1.35` package, benchmark version v1.2.2.

### 5.1 Structural attack ceilings on AgentDojo

**Setup.** For each of the 97 user tasks we execute the ground-truth call sequence in an environment whose injection slots carry unique canaries. This yields the calls, their arguments, and which outputs carry attacker-controlled content. All 97 tasks are injectable by construction of the benchmark. For the 35 injection tasks we use their ground-truth calls. Nine injection tasks (workspace 6–13 and travel 6) ship with an empty ground truth, so we annotated their minimal requirements by hand from the goal text and security check (Appendix A). A case is the product of a user task and an injection task within a suite (949 cases). We evaluate two threat modes. In *strict* mode, every attacker call must be permitted. In *sink-only* mode, only the attacker's side-effect calls must be permitted, which is an upper bound for attacks that can draw data already in context.

**Table 1.** Structural attack ceiling $s(G)$: the fraction of security cases in which the attack is still permitted (strict mode). Policies: STATIC = the suite's full toolset; UNION = all tools any task in the suite uses; READ-ONLY SPLIT = tasks without side-effect tools run in a read-only profile; TOOL = per-task tool allowlist; DEST = TOOL plus pinned destination arguments; BIND = TOOL plus all binding arguments pinned; ARG-REAL = argument pinning only for prompt-grounded values; "+R" = taint-triggered revocation. Sources: `results/permission_analysis.json` and `results/reachability_summary.json`.

| Policy | workspace (560) | travel (140) | banking (144) | slack (105) | all (949) | 95% CI (all) |
|---|---|---|---|---|---|---|
| STATIC | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | – |
| UNION | 0.429 | 0.714 | 1.000 | 0.800 | 0.599 | – |
| READ-ONLY SPLIT | 0.550 | 0.400 | 0.750 | 1.000 | 0.608 | – |
| TOOL | 0.034 | 0.186 | 0.292 | 0.181 | 0.112 | 0.081–0.148 |
| TOOL+R | 0.029 | 0.186 | 0.278 | 0.105 | 0.098 | 0.069–0.134 |
| ARG-REAL | 0.014 | 0.171 | 0.042 | 0.105 | 0.052 | – |
| DEST | 0.004 | 0.171 | 0.007 | 0.029 | 0.032 | 0.020–0.045 |
| BIND (oracle) | 0.000 | 0.171 | 0.000 | 0.029 | 0.028 | 0.017–0.041 |

The CIs come from a cluster bootstrap over user tasks (2,000 resamples). In sink-only mode the ceilings are TOOL 0.130, DEST 0.033 and BIND 0.030 (`reachability_summary.json`). Two independent implementations with different annotations (`analyze_permissions.py` admits alternative attacker call lists; `permission_reachability.py` uses one canonical list) agree exactly on TOOL (106/949) and BIND (27/949).

**Reading.** Moving from a suite-wide grant to a per-task tool allowlist removes 89% of structurally possible attacks, and binding-level pinning removes 97%. Coarse static splits help much less: a suite-union or read-only split still permits about 60% of cases. Granting a whole suite over-provisions badly. On average a suite grants 15.7× (workspace), 6.1× (travel), 6.4× (banking) and 4.2× (slack) as many tools as the task uses (`reachability_summary.json`, `privilege_surplus_mean`).

**Generalization of static profiles.** Per-task policies require a per-task policy generator, which is an LLM in Progent and Conseca. The alternative is a small set of static profiles. We designed $k$ profiles by greedy agglomeration on half of each suite's tasks and evaluated them on the other half, over 200 random splits (`profile_holdout.json`). With $k=4$, only 67.5% (workspace), 40.7% (travel), 59.3% (banking) and 66.3% (slack) of held-out tasks fit any profile. Tasks that fall back to the full grant push the ceiling to 0.35, 0.66, 0.62 and 0.47, versus the in-sample per-task values of 0.034, 0.186, 0.292 and 0.181. Static profiles do not transfer well to unseen tasks. Task-specific grants buy their security by trusting whatever generates the policy.

### 5.2 The floor is in-channel

All 27 cases that remain feasible under BIND fall into three injection tasks (`permission_analysis.json`, `arg_LP_residual`):
- **travel/injection_task_6** (20 cases) asks the agent to *say* that a hotel is worth visiting and needs no tool at all. It accounts for 2.1% of all cases and is untouchable by any permission policy.
- **travel/injection_task_2** (4 cases) creates a calendar reminder with no participants. It has no binding target, so it is feasible whenever the benign task creates events.
- **slack/injection_task_1** (3 cases) sends a phishing link by direct message to Alice, a legitimate recipient of the benign task, so only the content differs.

This matches Proposition 2. At binding granularity, what survives is exactly the set of attacks that reuse a legitimate action on a legitimate target. Permission design should therefore aim for binding granularity, and it should hand the remaining 2.8% to content-level controls. Making permissions finer still cannot reach it.

### 5.3 The provenance gap

Of the 97 tasks, 61 contain a consequential (side-effect) call. In 44 of these, at least one binding value (a recipient, participant, IBAN, URL, file or event identifier, or amount) does not appear in the user prompt and is derived from tool outputs: workspace 14/22, travel 0/6, banking 11/12, slack 19/21 (`target_provenance.csv`). Restricted to values that name an *external party* (recipient, participant, email, IBAN, URL, user), 22 tasks still have data-derived targets (`provenance_strict.json`). A realizable policy (Proposition 3) cannot pin these values. Its ceiling is 5.2% rather than 2.8% (Table 1, ARG-REAL), and the difference is concentrated in banking (0.042 vs 0.000) and slack (0.105 vs 0.029). These are the suites where tasks look like "pay the bill in this file" or "message the people in this channel".

The gap can be closed by having the human confirm each data-derived target (*pin-or-confirm*). Per benign task, the confirmation load this requires (distinct data-derived binding values) averages 0.50 in workspace (max 3), 0.00 in travel, 1.00 in banking (max 2) and 1.71 in slack (max 6). The share of tasks needing at least one confirmation is 35%, 0%, 69% and 90% respectively (`confirmation_load.json`).

### 5.4 Taint-triggered revocation is nearly inert

Revoking every grant that the benign trajectory no longer needs after its first untrusted read lowers the TOOL ceiling only from 0.112 to 0.098, and it does not change DEST or BIND at three-decimal precision (Table 1). The trajectory data explain why (`agentdojo_trajectories.json`). The first untrusted read is step 1 in 92.5% of workspace tasks, 93.8% of banking tasks, 66.7% of slack tasks and 10% of travel tasks. Mean trajectory lengths are 2.1, 2.1, 4.7 and 6.2 calls. Where tasks start by reading the inbox, the transaction list or a channel, nothing is left to revoke (Proposition 4). The only suite where revocation has bite, slack (0.181 to 0.105), is the one with longer trajectories whose taint comes later. Benchmark designers should note that the benefit of "restrict after reading untrusted data" depends on task shape, and AgentDojo's tasks mostly read first.

### 5.5 How often do benign tasks need the lethal trifecta?

Labelling tools as private-data reads (P), external sinks (E) and state-changing (W) (`analyze_permissions.py`, `LABELS`), 42 of 97 benign tasks need P, untrusted input and E within one trajectory: workspace 16/40, travel 0/20, banking 10/16, slack 16/21 (`permission_analysis.json`). On the attack side, 30 of the 35 injection tasks need an egress tool (our labels in `permission_reachability.py`). Willison's (2025) advice to remove one leg therefore has a measurable utility cost. Applied at the level of a whole agent, it would forbid 43% of these benign tasks. Applied at binding granularity, the conflict largely disappears: the leg is cut *for targets the task does not need* rather than for the tool.

### 5.6 Approval fatigue and the attention budget

**On AgentDojo.** We built a nested chain of base grants per suite by greedily removing the tool with the best ratio of blocked attack mass to added approval load. Everything outside the base grant is escalated. With $d_0=0.9$, the conditional feasibility is $r(G)=s(G)+(1-s(G))(1-d(n(G)))$ (`approval_fatigue.json`, Fig. `fig_approval_fatigue.png`). Without fatigue, the optimum reaches $r=1-d_0=0.10$ in workspace, banking and slack, meaning every attack must get past a human, and 0.229 in travel, where in-channel attacks remain. Each of these optima uses a mid-sized base grant (20 of 24 workspace tools, with 0.35 prompts per task). With $n_0=1$ prompt per task, the optimum moves to near-full base grants, keeping 21 of 24 workspace tools, 25/28 travel, 9/11 banking and 9/11 slack. The optimal risks rise to 0.28, 0.42, 0.48 and 0.49. Escalating everything outside the minimal grant reaches 0.86–1.00, almost as bad as granting everything (1.00). Corollaries 1–2 thus appear on real task structure: when the human is fatigued, designers should decide more of the boundary in advance and leave less to escalation.

**Theorem 1 check and the cost of fatigue-blind design.** We drew 300 random instances with $m=10$ atoms ($p_i\sim\mathrm{Beta}(0.6,2)$, $\lambda_i\sim U[0.01,0.2]$, $h_i\sim\mathrm{LogNormal}(0,1.5)$, $v_i\sim\mathrm{LogNormal}(0,1)$, $\kappa\sim U[0.005,0.05]$, $d_0=0.9$) and found the exact optimum over all $3^{10}$ assignments (`attention_budget.py`, `attention_budget.json`, Fig. `fig_attention_budget.png`). The bound $n^\star<n_0+w_j$ held in 300/300 instances for every $n_0\in\{0.5,1,2,2.80,4,8\}$. The largest observed $n^\star/n_0$ was 1.93 at $n_0=0.5$ and 0.72 at $n_0=2.80$. The fatigue-blind design (Corollary 3) always loads 2.58 prompts per task on average. Its excess loss over the optimum has median 80% (90th percentile 185%) at $n_0=1$, 58% (131%) at $n_0=2.80$ and 23% (59%) at $n_0=8$. Escalate-all is worse than grant-all in 97% of instances at $n_0=0.5$, 50% at $n_0=1$, 2% at $n_0=2.80$ and 0% at $n_0\ge4$ (Corollary 2).

**Calibrating $n_0$.** Ancker et al. (2017) report a 30% drop in reminder acceptance per additional reminder per encounter. If a malicious-request denial decays similarly per additional prompt, then $n_0=1/\ln(1/0.7)\approx2.8$ prompts per task. This mapping is an analogy (clinical reminders are not agent approvals), and we use it only to choose a plausible operating point. All results above are reported across $n_0\in[0.5,8]$.

**Putting it together.** At $n_0\approx2.8$ and $d_0=0.9$, pin-or-confirm (Section 5.3) gives conditional ceilings of 0.004 (workspace), 0.171 (travel, all in-channel), 0.015 (banking) and 0.068 (slack). The corresponding values without fatigue are 0.001, 0.171, 0.004 and 0.036 (`confirmation_load.json`). In AgentDojo, mean confirmation loads (0.0–1.7) stay within the budget. Slack's maximum of 6 confirmations in one task does not, and longer real-world tasks may not either.

## 6 Discussion: an Operational Answer, Predictions and a Test Protocol

**How much authority?** Our results support a three-tier answer that is stated in measurable terms.
1. *Standing authority should be granted at binding granularity, per task.* Moving from suite grants to per-task tool allowlists to binding-level grants lowers the structural ceiling from 100% to 11.2% to 2.8%. Coarse static profiles stay near 60% and transfer poorly to unseen tasks.
2. *Authority whose target is data-derived must not be standing.* It should be confirmed, and confirmations should be counted against a budget of about $n_0$ prompts per task. If a task needs more, the boundary should be redesigned, for example by splitting the task, rather than adding prompts (Theorem 1).
3. *Permissions cannot address in-channel attacks*, which made up 2.8% of AgentDojo cases. Those need content-level defenses, and the residual risk should be stated as such rather than hidden inside an aggregate attack success rate.

The harm-to-need rule of Corollary 1 gives the per-atom answer. Keep standing authority over an atom when its expected harm per legitimate use, $r_i/p_i$, is below the value of not being blocked, and when escalating it buys less vigilance-weighted harm reduction than it costs in attention.

**Falsifiable predictions.**
- **P1 (ceiling).** For any LLM agent run on AgentDojo v1.2.2 under a correctly enforced TOOL policy, the strict-mode attack success rate over the 949 cases will not exceed 11.2%. Under BIND it will not exceed 2.8%, and every success under BIND will come from travel/injection_task_6, travel/injection_task_2 or slack/injection_task_1. A violation would indicate a monitor bypass or an alternative attack path missing from our annotation, and either is informative.
- **P2 (ordering).** Measured ASR ratios across policies will follow the ceiling ordering STATIC > UNION ≈ READ-ONLY > TOOL > TOOL+R ≥ ARG-REAL > DEST ≥ BIND. Measured ASR divided by the ceiling estimates the model's conditional compliance $c_M$, which we predict to be roughly constant across policies for a fixed model and attack.
- **P3 (attention budget).** In a human-subject study where participants approve or deny agent action requests and a fixed fraction of requests are malicious, the per-request denial rate for malicious requests will fall with the number of requests per task. The loss-minimizing request rate will lie below $\hat n_0+1$, where $\hat n_0$ is the fitted fatigue scale.
- **P4 (revocation).** On benchmarks whose tasks perform their first untrusted read later in the trajectory, taint-triggered revocation will close more of the gap between TOOL and DEST than on AgentDojo. The effect should scale with the post-taint fraction of side-effect calls.

**Pre-registered protocol (not run here).** (i) *Agents:* run three open-weight models of different sizes with the AgentDojo `important_instructions` attack under STATIC, TOOL, DEST and BIND monitors, implemented as a wrapper that rejects disallowed calls using our grants (`reachability_pairs.json` gives per-case expectations). Report ASR, utility and ASR/ceiling with cluster-bootstrap CIs over user tasks. (ii) *Humans:* run a between-subjects study with request loads of 1, 2, 4 and 8 per task and 10% malicious requests (matched for wording and difficulty). Fit $d(n)=d_0e^{-n/n_0}$ by maximum likelihood, compare it with a power-law alternative by AIC, and test P3 using the fitted model to compute the optimal load. The primary outcome is the malicious-request denial rate. Exclusion rules and sample size (power analysis for detecting a 15-point drop in denial between loads 1 and 8 at α=0.05) are to be fixed before data collection.

## 7 Limitations

- **Ground truth is one trajectory.** Each task's requirement is taken from AgentDojo's canonical solution. Real agents take other valid paths, using more tools or different lookups. Per-task grants derived from one canonical path therefore *overstate* realizable utility. Hand-annotated alternatives for nine injection tasks partly address the attacker side, but nothing addresses the benign side. Our ceilings are exact for the stated grants, but the grants an LLM policy generator would produce are not measured here.
- **Annotation choices matter.** Tool labels (private, sink, write), the split between binding and content arguments, and the hand annotations for nine injection tasks are our judgments, listed in the code. For example, we treat `amount` as content in the BIND policy and as a target in provenance counting. Different choices would move the numbers by a few cases. The two independent implementations agree on TOOL and BIND, but they share the benchmark's ground truth.
- **Prompt-grounding is a string heuristic.** A value counts as prompt-grounded if it appears in the prompt string. This misses paraphrases ("my landlord") and overcounts accidental matches.
- **AgentDojo is small and stylized.** It has 97 tasks, mostly short (2–6 calls), and every task is injectable by design. It contains no multi-agent delegation, code execution or long-horizon tasks. The taint-position finding in particular is a property of this task distribution.
- **The fatigue model is assumed, not measured.** The exponential form, $d_0=0.9$ and the analogy to clinical reminders (Ancker et al., 2017) are modelling choices. Theorem 1 relies on the exponential form (a different decay changes the constant). The model also assumes that legitimate requests are always approved, that the human cannot tell malicious from benign requests beyond $d(n)$, and that loads add linearly across atoms. The random instances in Section 5.6 are synthetic.
- **No LLM or human was run.** The ceilings bound ASR but do not estimate it, and P1–P4 remain untested.
- **Out of scope.** Harms that need no tool, such as misinformation in the answer, and harms from the user's own malicious intent are outside the permission model entirely.

## 8 Conclusion

"How much authority should an agent hold?" can be answered in numbers once permissions are treated as a lattice of granularities, the model is separated from the monitor, and human approval is treated as a resource that fatigue depletes. On AgentDojo, per-task binding-level grants cut the structural attack ceiling from 100% to 2.8%. The remainder is in-channel and beyond the reach of permissions. The binding values that are hardest to pin are the data-derived ones, which arise in 44 of 61 consequential tasks. Revoking permissions after untrusted reads does little when agents read first. And no design should spend more than about $n_0+1$ human approvals per task, whatever is at stake. We hope that the ceilings, the provenance and taint measurements and the attention budget become standard quantities to report alongside attack success rates.

## Appendix A: Hand annotations

The minimal attacker requirements for injection tasks with empty ground truth (AgentDojo v1.2.2) are as follows. Workspace 6/8: read unread mail (any of three read tools) → `send_email` to the attacker → `delete_email` (8 also deletes the unread mail). Workspace 7: read received and sent mail → `send_email` ×2 → `delete_email`. Workspace 9: as 6/8, plus `send_email` to the user's own address. Workspace 10: `list_files` or `search_files` → `send_email` with attachments → `delete_email`. Workspace 11/12: `list_files` → `send_email` → `delete_email` → `delete_file` (12 adds a self-email). Workspace 13: read today's events → `send_email` to each participant of today's meetings (5 addresses taken from the environment). Travel 6: no tool (output manipulation only). Exact encodings are in `code/extract_agentdojo.py` (`MANUAL_INJECTION`) and `code/permission_reachability.py` (`MANUAL`).

## Reproducibility

`code/README.md` lists every command. The full pipeline runs in under two minutes of CPU time with `agentdojo==0.1.35`. All figures (`fig_policies.png`, `fig_granularity_frontier.png`, `fig_approval_fatigue.png`, `fig_attention_budget.png`) and data files are in `results/`.

## References

Ancker, J. S., Edwards, A., Nosal, S., Hauser, D., Mauer, E., & Kaushal, R. (2017). Effects of workload, work complexity, and repeated alerts on alert fatigue in a clinical decision support system. *BMC Medical Informatics and Decision Making*, 17, 36. doi:10.1186/s12911-017-0430-8

Betser, R., Bose, S., Giloni, A., Picardi, C., Padakandla, S., & Vainshtein, R. (2026). AgenTRIM: Tool Risk Mitigation for Agentic AI. arXiv:2601.12449. (Also in *Findings of EMNLP 2026*.)

Beurer-Kellner, L., et al. (2025). Design Patterns for Securing LLM Agents against Prompt Injections. arXiv:2506.08837.

Chan, A., Ezell, C., Kaufmann, M., Wei, K., Hammond, L., Bradley, H., Bluemke, E., Rajkumar, N., Krueger, D., Kolt, N., Heim, L., & Anderljung, M. (2024). Visibility into AI Agents. In *Proceedings of the 2024 ACM Conference on Fairness, Accountability, and Transparency (FAccT '24)*, 958–973. arXiv:2401.13138.

Costa, M., Köpf, B., Kolluri, A., Paverd, A., Russinovich, M., Salem, A., Tople, S., Wutschitz, L., & Zanella-Béguelin, S. (2025). Securing AI Agents with Information-Flow Control. arXiv:2505.23643.

Debenedetti, E., Zhang, J., Balunović, M., Beurer-Kellner, L., Fischer, M., & Tramèr, F. (2024). AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents. In *Advances in Neural Information Processing Systems 37 (Datasets and Benchmarks Track)*. arXiv:2406.13352.

Debenedetti, E., Shumailov, I., Fan, T., Hayes, J., Carlini, N., Fabian, D., Kern, C., Shi, C., Terzis, A., & Tramèr, F. (2025). Defeating Prompt Injections by Design. arXiv:2503.18813.

Felt, A. P., Ha, E., Egelman, S., Haney, A., Chin, E., & Wagner, D. (2012). Android Permissions: User Attention, Comprehension, and Behavior. In *Proceedings of the Eighth Symposium on Usable Privacy and Security (SOUPS '12)*.

Feng, K. J. K., McDonald, D. W., & Zhang, A. X. (2025). Levels of Autonomy for AI Agents. arXiv:2506.12469.

Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., & Fritz, M. (2023). Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection. In *Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security (AISec '23)*, 79–90. arXiv:2302.12173.

Hardy, N. (1988). The Confused Deputy (or why capabilities might have been invented). *ACM SIGOPS Operating Systems Review*, 22(4), 36–38.

Kolt, N. (2026). Governing AI Agents. *Notre Dame Law Review*, 101, 335ff. Preprint arXiv:2501.07913 (2025).

Michael, A. E., & Roesner, F. (2026). How Agents Ask for Permission: User Permissions for AI Agents, from Interfaces to Enforcement. arXiv:2607.13718.

Mitchell, M., Ghosh, A., Luccioni, A. S., & Pistilli, G. (2025). Fully Autonomous AI Agents Should Not be Developed. arXiv:2502.02649.

Saltzer, J. H., & Schroeder, M. D. (1975). The Protection of Information in Computer Systems. *Proceedings of the IEEE*, 63(9), 1278–1308.

Shavit, Y., Agarwal, S., Brundage, M., et al. (2023). Practices for Governing Agentic AI Systems. OpenAI white paper.

Shi, T., He, J., Wang, Z., Li, H., Wu, L., Guo, W., & Song, D. (2025). Progent: Programmable Privilege Control for LLM Agents. arXiv:2504.11703.

Tsai, L., & Bagdasarian, E. (2025). Contextual Agent Security: A Policy for Every Purpose. In *Workshop on Hot Topics in Operating Systems (HotOS '25)*. arXiv:2501.17070.

Willison, S. (2025, June 16). The lethal trifecta for AI agents: private data, untrusted content, and external communication. *Simon Willison's Weblog*. https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/

Xu, X. (2026). Incomparable Minima in Agent Tool Authorization. arXiv:2610.31914.

Zhan, Q., Liang, Z., Ying, Z., & Kang, D. (2024). InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents. In *Findings of the Association for Computational Linguistics: ACL 2024*, 10471–10506.

Zhu, J., Tseng, K., Vernik, G., Huang, X., Patil, S. G., Fang, V., & Popa, R. A. (2025). MiniScope: A Least Privilege Framework for Authorizing Tool Calling Agents. arXiv:2512.11147.
