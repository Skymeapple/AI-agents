# Stranded or Broken: Dependency Conventions, Not Negotiation, Govern How a Breaking Agent-Protocol Revision Diffuses — A Census of the MCP Registry

**Target venue:** ACM Internet Measurement Conference (IMC)

## Abstract

How agent protocols evolve depends on whether a deployed ecosystem can follow a revision. We measure this for the Model Context Protocol (MCP), whose 2026-07-28 revision removed the session handshake and is the first revision that the official SDKs deliver only across a major-version boundary. From the official MCP Registry (70,033 servers, snapshot 2026-11-22) we resolve every listed npm package (11,375 with history, 112,249 published versions) and PyPI package (5,107) to the protocol revisions its declared SDK dependency can speak, using a release-by-release map built by reading the published SDK artifacts. Three results follow. First, the three earlier revisions reached 78–84% of npm servers within one day of the SDK release and 89–96% within 118 days, because 94.7% of dependencies are caret ranges that float within a major version. The 2026-07-28 revision reached 0.4% in one day and 7.0% in 118 days. Second, the two packaging ecosystems fail in opposite ways. npm servers are stranded: incumbents migrate at a hazard of 0.135 per year (4.2% in 117 days, with no acceleration), a legacy half-life of about five years. PyPI servers are split: 23.3% declare ranges that float into the v2 SDK, and a static scan shows 67.3% of those import a module path that v2 removed, so 20.1% of PyPI servers with a declared SDK dependency cannot import under a fresh resolution. Third, the share of new servers born on the new SDK is the controlling variable. It is 5.4% on npm, where v2 has a new package name, and 58.1% on PyPI, where the name is unchanged; bulk publishers, who produce 69% of new npm servers, are born modern 0.14% of the time. A two-parameter entry–migration model formalises this: with relative inflow g, birth share b and migration hazard h, the modern share converges to (h + bg)/(g + h). At measured rates a 0.1 increase in b is worth 2.8 times a 0.1-per-year increase in h. We give dated, falsifiable projections: under constant inflow the npm modern share on 2027-07-28 is 13.1% (95% interval 12.1–14.1%).

## 中文摘要

智能体协议如何演化，取决于已部署的生态能否跟上规范的修订。本文以模型上下文协议（MCP）为对象做了一次普查式测量。MCP 的 2026-07-28 修订版取消了会话握手，也是官方 SDK 首次只能跨主版本号提供的修订版。我们从官方 MCP Registry（70,033 个服务器，快照日期 2026-11-22）出发，把其中登记的每个 npm 包（11,375 个有发布历史，共 112,249 个版本）和 PyPI 包（5,107 个）按其声明的 SDK 依赖解析到它能使用的协议修订版；SDK 版本与协议修订版之间的对应关系来自逐个读取已发布的 SDK 制品。结论有三点。第一，此前三个修订版在 SDK 发布后一天内就覆盖了 78–84% 的 npm 服务器，118 天内达到 89–96%，原因是 94.7% 的依赖写成在主版本内自动浮动的 caret 范围；而 2026-07-28 修订版一天内只到 0.4%，118 天只到 7.0%。第二，两个包生态以相反的方式失败。npm 服务器被“搁浅”：存量包的迁移风险率为每年 0.135（117 天内 4.2%，没有加速迹象），旧版存量的半衰期约五年。PyPI 服务器则出现分裂：23.3% 的包声明的版本范围会自动浮动到 v2 SDK，而静态扫描显示其中 67.3% 引用了 v2 已删除的模块路径，因此在声明了 SDK 依赖的 PyPI 服务器中，有 20.1% 在全新解析依赖时无法导入。第三，新服务器“出生”时是否使用新 SDK 是决定性变量：在 v2 换了包名的 npm 上该比例为 5.4%，在包名不变的 PyPI 上为 58.1%；批量发布者贡献了 69% 的新 npm 服务器，其出生即新版的比例只有 0.14%。我们用一个两参数的“进入—迁移”模型刻画这一点：设相对流入率为 g、出生新版比例为 b、迁移风险率为 h，则新版份额收敛到 (h + bg)/(g + h)。按实测参数，b 提高 0.1 的效果是 h 每年提高 0.1 的 2.8 倍。我们给出带日期、可证伪的预测：在流入量不变的情形下，2027-07-28 时 npm 的新版份额为 13.1%（95% 区间 12.1–14.1%）。

## 1 Introduction

The question "how will agent protocol standards evolve?" is usually answered by reading specifications and roadmaps. Specifications say what a protocol may become. Whether it becomes that depends on a second quantity that specifications do not contain: how quickly the deployed population of implementations follows a revision. A protocol whose population follows within days can change freely. A protocol whose population follows over years is constrained by its own installed base, the condition the networking literature calls ossification [10, 11, 12].

MCP offers an unusually clean setting in which to measure this. Since December 2025 it has been governed under the Linux Foundation's Agentic AI Foundation [9], and it revises on a dated cadence (2024-11-05, 2025-03-26, 2025-06-18, 2025-11-25, 2026-07-28). Almost all servers are built on one of two official SDKs, so a server's protocol capability is determined by which SDK release it resolves to. And since September 2025 an official registry lists servers together with the npm or PyPI packages that implement them. The dependency metadata of those packages is public and timestamped back to each package's first release.

The 2026-07-28 revision is the interesting case. It removes the `initialize` handshake and protocol-level sessions, carries version and capabilities on every request, and replaces server-initiated requests with a retry pattern [1]. The official SDKs deliver it only in a new major version: on PyPI as `mcp` 2.0.0 under the existing name, and on npm as a new package, `@modelcontextprotocol/server` 2.0.0, while `@modelcontextprotocol/sdk` 1.x continues as a maintenance line that, in its own README, does not plan to support the revision. The same revision also adopted a feature lifecycle policy with a minimum twelve-month deprecation window [1]. The practical question for the protocol's future is therefore concrete: twelve months after a breaking revision, what fraction of servers can speak it?

We narrow the broad question to this sub-question: **how fast does a protocol revision reach the server population, what mechanism sets that speed, and what does the mechanism predict for the 2026-07-28 revision?**

Contributions:

1. **A measurement method.** We map every release of the official TypeScript and Python SDKs to the protocol revisions it implements by reading the published artifacts, then resolve each registry-listed package's declared dependency range against the SDK releases available at any date. This yields two reach curves per revision: fresh-install reach (the newest SDK the range admits) and declared-floor reach (the oldest it admits).
2. **A census.** We apply the method to all 11,375 npm and 5,107 PyPI packages listed in the official MCP Registry on 2026-11-22, with 112,249 npm versions giving a weekly history from March 2025.
3. **A mechanism finding.** Within a major SDK version, revisions diffuse in days through floating dependency ranges, without publisher action. Across the major boundary, npm servers are stranded and a fifth of PyPI servers are broken, and the difference between the ecosystems traces to a packaging decision (new package name versus same name).
4. **A model with proved steady state and dated predictions.** An entry–migration model shows that in a growing ecosystem the share of servers born on the new SDK dominates the migration rate of incumbents. We fit it, backtest it, report where the simple form fails, and state projections that data collected in 2027 can falsify.

All numbers below come from code run for this paper, with outputs in `results/summary.json` and `results/model.json`, unless a citation is given.

## 2 Related Work

**Agent protocols.** Surveys compare MCP with the Agent2Agent, Agent Communication and Agent Network protocols and classify them by architecture and intended layer [2, 3]; Hou et al. describe the MCP landscape and its security threats [4]. These works analyse designs. None measures how a deployed population follows a revision.

**MCP ecosystem measurement.** Hasan et al. study the health, security and maintainability of 1,899 open-source MCP servers [5]. Guo et al. crawl six marketplaces, collect 8,401 entries, and report that more than half of listed projects are invalid or low-value and that servers share a dependency monoculture [6]. Majeed et al. study 1,723 client applications [7]. Kraishan harvests 21,643 servers and 72,606 version records and links silent behavioural drift to security findings [8]. We build on the monoculture observation in [6]: because almost every server depends on one of two SDKs, the SDK dependency graph is a usable instrument for protocol capability. To our knowledge no prior study maps SDK releases to protocol revisions or measures revision reach over time.

**Protocol ossification and deployment.** Honda et al. showed that middleboxes constrain which TCP extensions can be deployed [10]. RFC 9170 argues that extension mechanisms that are not exercised stop working [11]. Holz et al. tracked TLS 1.3 and found fast uptake driven by a few large operators who control both endpoints [12]. MCP differs in that capability is delivered through a library dependency, so the relevant inertia sits in package metadata, not in network equipment.

**Dependency lag.** Zerouali et al. define technical lag for npm dependencies and attribute it to how constraints are written [13]. Kula et al. find that 81.5% of studied systems keep outdated dependencies [14]. Bogart et al. describe how ecosystems differ in who bears the cost of a breaking change [15]. We connect this literature to protocol evolution: the same range conventions that cause or prevent lag determine the reach of a protocol revision.

**Standards economics.** Farrell and Saloner show that an installed base can produce excess inertia against a better standard [16]. Our model is a population-accounting counterpart with entry: it asks how the installed base's weight changes when the population is growing quickly.

## 3 Data and Method

### 3.1 Sources

*Registry.* We downloaded the latest version of every server from the official MCP Registry API (`registry.modelcontextprotocol.io/v0/servers`). The snapshot holds 70,033 servers published between 2025-09-09 and 2026-11-22, in 42,991 namespaces. 42,686 (61.0%) are remote-only, 21,675 package-only, 4,968 both, 704 neither. 18,213 servers reference an npm package and 5,701 a PyPI package.

*npm.* For each of the 11,488 distinct npm identifiers we fetched the packument and kept, for every published version, the publish time and the declared ranges on the official SDK packages. 113 identifiers did not resolve. The remaining 11,375 packages have 112,249 stable versions.

*PyPI.* For each of 5,148 PyPI identifiers we fetched the metadata of the latest release (requirements on `mcp` and `fastmcp`) and the upload dates of all releases; 5,107 resolved. PyPI history of requirements was not collected, so PyPI results are cross-sectional.

*SDK-to-revision map.* For all 84 releases of `@modelcontextprotocol/sdk` and 73 releases of `mcp` we downloaded the published tarball or wheel and read the source constants that list supported protocol versions. For the v2 lines we did the same for `@modelcontextprotocol/server` (15 releases) and `mcp-types` (12 releases), which `mcp` 2.x pins exactly. Table 1 gives the first stable release supporting each revision.

**Table 1. Specification date and first stable TypeScript SDK release implementing each revision (run here).**

| Revision | First stable SDK release | SDK date | Days after spec |
|---|---|---|---|
| 2025-03-26 | sdk 1.11.0 | 2025-05-01 | 36 |
| 2025-06-18 | sdk 1.13.0 | 2025-06-18 | 0 |
| 2025-11-25 | sdk 1.24.1 | 2025-12-02 | 7 |
| 2026-07-28 | server 2.0.0 | 2026-07-27 | −1 |

The latest v1 release, 1.32.2 (2026-10-23), supports revisions up to 2025-11-25 and contains no reference to 2026-07-28. On PyPI the first stable release supporting 2026-07-28 is `mcp` 2.0.0 (2026-07-28); the v1 line continued to 1.30.0 (2026-09-07).

### 3.2 Dependency classes

Each published npm version falls into one class: **sdk1** (runtime or peer dependency on `@modelcontextprotocol/sdk`), **v2** (runtime or peer dependency on `@modelcontextprotocol/server` or `/core`), **dev-only** (the SDK appears only in `devDependencies`, which indicates a bundled build), **framework** (another MCP framework only), or **opaque** (no declared dependency on any tracked package). At the snapshot the 11,375 packages are 63.9% sdk1, 29.5% opaque, 4.8% v2, 1.4% dev-only on v1, 0.2% dev-only on v2 and 0.2% framework. Reach is reported over the 7,988 packages with a resolvable declared SDK dependency. Opaque packages are excluded from denominators; Section 7 discusses the consequence.

### 3.3 Reach

For a package version with range ρ on an SDK line, and a date t, let V(t) be the SDK releases published by t. *Fresh-install reach* of revision r is whether the maximum element of V(t) satisfying ρ supports r. This is what `npx` or `uvx` obtains when resolving dependencies anew, since published packages carry ranges and not lockfiles. *Declared-floor reach* is whether the minimum satisfying element supports r, which is what the range guarantees under any cache or lockfile. Bundled (dev-only) packages are frozen at the SDK that their range resolved to on their publish date. A package's state at t is its most recently published stable version. We evaluate both measures weekly from March 2025 and at fixed ages after each SDK release. npm ranges are resolved with a Python port of node-semver; PyPI specifiers with the `packaging` library. For PyPI packages that reach the SDK through FastMCP we first resolve the FastMCP release, then its own requirement on `mcp`.

### 3.4 Migration, entry and the import scan

*Migration.* The incumbent cohort is the 3,372 npm packages in class sdk1 when v2 became generally available (GA, 2026-07-27). The event is the first later version in class v2. We estimate the cumulative migration function by Kaplan–Meier with Greenwood intervals, and a constant hazard as events over exposure.

*Entry.* For packages first published after GA we record whether the first version is in class v2 ("born modern"). Because a few namespaces publish many packages, we report three units: packages; publisher namespaces counted once at their first post-GA package; and a bulk/non-bulk split, where an entrant is *bulk* if its namespace launched at least five npm servers in the same week.

*Import scan.* `mcp` 2.x removed 47 module paths that exist in 1.30.0, among them `mcp.server.fastmcp` and its submodules; from 2.2.0 the SDK ships a stub at that path whose only statement raises `ModuleNotFoundError` with a message that names the rename to `mcp.server.mcpserver` and advises pinning `mcp<2`. We obtained the removed set by diffing wheel contents and treating such stubs as absent. For all 3,268 PyPI packages with a direct `mcp` requirement we downloaded the wheel of the recorded release (2 MB cap; 3,198 scanned, 68 over the cap, 2 metadata errors) and extracted the `mcp.*` modules imported outside test directories. No downloaded code was executed.

## 4 Results

### 4.1 Within a major version, revisions arrive in days

Figure 1 (`results/fig1_reach.png`) and Table 2 show reach by age. For the three revisions delivered inside SDK 1.x, fresh-install reach is 78–84% one day after the SDK release. No publisher did anything in that day. The mechanism is the range convention: of 7,265 sdk1 packages at the snapshot, 94.7% use a caret range, 4.75% an exact pin, and under 1% anything else. A caret range admits every later 1.x release, so a new SDK release changes what those servers speak on their next cold start.

**Table 2. Reach among npm servers with a declared SDK dependency, by days since the first stable SDK release supporting the revision (run here; n is the population alive at that age).**

| Revision | Fresh, day 1 | Fresh, day 30 | Fresh, day 118 | Floor, day 118 | n at day 118 |
|---|---|---|---|---|---|
| 2025-03-26 | 84.0% | 83.9% | 89.0% | 57.6% | 118 |
| 2025-06-18 | 82.4% | 88.2% | 89.9% | 57.0% | 179 |
| 2025-11-25 | 77.7% | 84.3% | 96.1% | 44.7% | 1,258 |
| 2026-07-28 | 0.4% | 2.7% | 7.0% | 7.0% | 7,988 |

At the snapshot the three earlier revisions have fresh-install reach of 99.7%, 99.7% and 99.5%, and declared-floor reach of 55.2%, 47.4% and 44.2%. The gap between the two measures is the share of servers whose capability depends on when their dependencies were last resolved. The populations for the first two revisions are small because most listed packages did not yet exist in 2025; the pattern is the same at n = 1,258.

For 2026-07-28 the two measures coincide (7.0% and 6.97%) and both are low. The revision is 118 days old at the snapshot, the same age at which its predecessor had 96.1% fresh-install reach. Even on the stricter floor measure the predecessor had 44.7%.

The registry's own metadata offers a contrast. The share of newly published `server.json` records using the latest schema version (2025-12-11) was 40.8% in December 2025 and 99.6% in January 2026, and has stayed between 89% and 98% since. Among remote endpoints, 1.56% (744 of 47,654) advertise only the HTTP+SSE transport that was deprecated in 2025-03-26, and 0.07% of those published in November 2026. Artifacts that pass through one publishing tool follow a change within a month.

### 4.2 npm incumbents are stranded

Of the 3,372 incumbents, 143 had migrated by the snapshot: 4.2% (95% CI 3.6–5.0%) in 117.5 days (Figure 2, left). The Kaplan–Meier estimate is 1.3% at 30 days, 2.2% at 60, 3.1% at 90 and 4.2% at 117. Among the 2,389 incumbents that had released in the 90 days before GA it is 5.4% (4.5–6.3%); among the 983 dormant ones 1.5% (0.8–2.3%).

This is not explained by inactivity. 47.5% of incumbents (1,603) published at least one new version after GA, and 58.2% of the active ones did. Of those that released, 8.9% (7.6–10.4%) migrated. Nine in ten incumbents that shipped a release after GA shipped it on the v1 SDK.

The constant-hazard estimate is 0.135 per year. Splitting the window in half gives 0.136 and 0.133: there is no sign of acceleration. At this hazard the legacy half-life is 5.1 years.

For comparison we built the same cohort for the previous revision: the 229 sdk1 packages alive when SDK 1.24.1 shipped, observed for the same 117.5 days. 35.8% (29.9–42.2%) published a version whose declared floor already implied the revision. Publishers therefore made an explicit, optional range bump at 8.4 times the rate at which they crossed the major boundary.

### 4.3 New servers are mostly born on the old SDK

After GA, 5,019 npm packages with a runtime SDK dependency were first published. 269 were born on v2: 5.4% (4.8–6.0%). By month the share is 8.0% (July, n = 50), 9.5% (August), 7.5% (September), 8.2% (October) and 2.5% (November, n = 2,445). There is no upward trend.

The November figure reflects bulk publishing. 3,460 of the 5,019 post-GA entrants (69%) come from namespaces that launched five or more servers in a single week, and 5 of those 3,460 were born modern (0.14%). The remaining 1,559 were born modern 16.9% of the time (15.2–18.9%). Counting each namespace once at its first post-GA package gives 16.4% (220 of 1,343), and by month 10.3%, 12.6%, 21.3%, 15.4% and 17.3%. So for independent publishers the birth share is about one in six and roughly flat since September; for automated producers it is near zero.

Entrants dominate the stock. Post-GA entrants are 56.7% of all npm servers with a runtime SDK dependency at the snapshot. The modern share is 5.0% among pre-GA packages and 8.5% among post-GA packages, 6.95% overall (543 of 7,808). Of the 543 modern packages, 50.3% were born modern, 29.7% are packages that predate GA and 20.1% are post-GA entrants that were born on v1 and migrated (at a hazard of 0.221 per year, from 95 events).

Robustness: weighting each namespace equally gives a modern share of 9.9%; excluding the ten largest namespaces, 10.9%; restricting to packages released in the last 90 days, 8.8%. None changes the order of magnitude. 13.7% of sdk1 packages have had no release for 180 days.

### 4.4 PyPI: the same revision, the opposite failure

Table 3 classifies the 5,107 PyPI packages (Figure 3, `results/fig3_pypi.png`).

**Table 3. PyPI servers by how their declared requirement resolves on 2026-11-22 (run here).**

| Class | Packages | Share |
|---|---|---|
| Floats into v2 (range admits `mcp` 2.x, floor is 1.x) | 1,191 | 23.3% |
| Requires v2 (floor is 2.x) | 1,168 | 22.9% |
| Opaque (no declared requirement on `mcp` or `fastmcp`) | 1,146 | 22.4% |
| Capped at v1 (upper bound below 2) | 906 | 17.7% |
| Via FastMCP, resolves to v2 | 500 | 9.8% |
| Via FastMCP, resolves to v1 | 192 | 3.8% |
| Unresolvable | 4 | 0.1% |

Among the 3,957 packages with a declared requirement, nominal fresh-install reach of 2026-07-28 is 72.3%, ten times the npm figure. The floor measure is 29.5%. The difference is the 1,191 packages whose authors wrote a lower bound and no upper bound. Python convention does not cap at the next major version, so these ranges float across the boundary that npm's caret respects.

Floating across a breaking boundary is not an upgrade. Of the 1,182 scanned packages in that class, 795 (67.3%, CI 64.5–69.9%) import a module that `mcp` 2.x removed and import no v2-only module; in 794 of them the path is `mcp.server.fastmcp` or a submodule. A further 122 (10.3%) import both old and new paths, which is the pattern of a compatibility shim; 37 (3.1%) import only v2 paths; 213 (18.0%) import only modules present in both lines; 15 have no `mcp` import we could find. The 795 are 20.1% of all PyPI servers with a declared requirement. Under a fresh resolution they receive an SDK in which their first import raises. 720 of the 795 have had no release since GA. 75 have released since, and 29 were first published after GA with the old import and an unbounded range.

Counting only packages that require v2, resolve to it through FastMCP, or import v2 paths, effective reach is 46.2%; counting also the packages whose compatibility is undetermined, it is at most 52.2%.

The SDK maintainers responded at the choke point. Release 2.2.0, 41 days after GA, added the stub module whose error message explains the rename and suggests pinning `mcp<2`. That repairs the diagnosis for every affected server at once, without any server publisher acting. It does not repair the servers.

### 4.5 The birth share differs by a factor of seven to eleven between ecosystems

Among PyPI packages first released after GA with a declared requirement, 58.1% (922 of 1,587; CI 55.7–60.5%) are on a v2 stack at the snapshot, rising by birth month from 33.3% (July, n = 24) to 44.9%, 58.8%, 65.4% and 58.7%. The comparable npm figure is 8.5%. Among packages that predate GA the figures are 31.5% on PyPI and 5.0% on npm.

The protocol revision, its date, its governance and its documentation are the same in both ecosystems. One difference is the default: `pip install mcp` and an unpinned `mcp` requirement yield 2.x, while `npm install @modelcontextprotocol/sdk` yields 1.32.2 because v2 has a different name. The PyPI figure is a state at the snapshot and the npm birth figure is a state at first publication, so the cleanest like-for-like comparison is state at snapshot among post-GA packages: 58.1% against 8.5%. We cannot exclude that the two developer communities differ in other ways (Section 7), but the direction and size are what the default would produce.

## 5 An Entry–Migration Model

### 5.1 Model and steady state

Let L(t) and M(t) be the numbers of legacy and modern servers, N = L + M, E(t) the entry flow, b the share of entrants born modern and h the migration hazard of legacy servers:

dM/dt = bE + hL,  dL/dt = (1 − b)E − hL.

Write g = E/N for relative inflow and ℓ = L/N for the legacy share.

**Proposition 1.** ℓ obeys dℓ/dt = (1 − b)g − (g + h)ℓ. If g, b and h are constant, ℓ converges exponentially at rate g + h to ℓ* = (1 − b)g/(g + h), so the modern share converges to m* = (h + bg)/(g + h).

*Proof.* dℓ/dt = (dL/dt)/N − ℓ(dN/dt)/N = (1 − b)g − hℓ − gℓ, using dN/dt = E = gN. This is linear in ℓ with constant coefficients; its solution is ℓ(t) = ℓ* + (ℓ(0) − ℓ*)e^{−(g+h)t}. ∎

**Corollary 1 (birth dominance).** ∂m*/∂b = g/(g + h) and ∂m*/∂h = (1 − b)g/(g + h)². Their ratio is (g + h)/(1 − b), which exceeds 1 whenever g + h > 1 − b. In an ecosystem whose relative inflow exceeds about one per year, a unit of birth share is worth more than a unit per year of migration hazard.

**Corollary 2 (ceiling under growth).** If g ≫ h then m* ≈ b. While the population is growing quickly the modern share cannot rise much above the birth share, whatever the incumbents do.

**Corollary 3 (after growth).** If g = 0 then ℓ(t) = ℓ(0)e^{−ht}: the legacy share decays only at the migration hazard.

**Corollary 4 (deprecation feasibility).** For the steady-state legacy share to be at most ε, the birth share must satisfy b ≥ 1 − ε(g + h)/g.

A discrete-time simulation with constant g reproduces the closed form to four decimals for three parameter sets (`model.json`, `closed_form_check`).

### 5.2 Fit

We fit the weekly discrete model to the 16 complete weeks after GA (through 2026-11-16), over npm packages with a runtime SDK dependency: L(0) = 3,372, M(0) = 20, 4,208 entrants of which 258 born modern, 218 migrations. The pooled hazard is h = 0.162 per year (incumbents and migrating entrants together), b = 0.061 at package level and 0.166 at namespace level. Weekly entry over the last eight weeks has median 357.5, giving g = 2.45 per year at the end of the window. The observed modern share at the end of week 16 is 6.53% (6.95% at the snapshot six days later).

At these rates m* is 12.0% with package-level b and 21.8% with namespace-level b; the relaxation time 1/(g + h) is 0.38 years; ∂m*/∂b = 0.94 and ∂m*/∂h = 0.34 per unit per year. Raising b by 0.1 raises m* by 0.094. Raising h by 0.1 per year, a 62% increase, raises it by 0.034. The ratio is 2.8.

### 5.3 Backtest

We fit b and h on the first K weeks and predicted the share at week 16 given the realised weekly entry counts (Table 4, Figure 4 left).

**Table 4. Backtest: absolute error in the week-16 modern share (observed 6.53%) (run here).**

| Fit window | One-class model | Persistence baseline | Two-class model |
|---|---|---|---|
| Weeks 0–3 | 0.019 | 0.038 | 0.011 |
| Weeks 0–5 | 0.040 | 0.023 | 0.003 |
| Weeks 0–7 | 0.033 | 0.012 | 0.0003 |
| Weeks 0–9 | 0.013 | 0.010 | 0.001 |

The one-class model over-predicts and loses to a no-change baseline in three of four windows. The cause is that b is not stationary at package level: later weeks contain bursts of bulk entrants that are almost never born modern. A two-class variant with separate birth shares for bulk and non-bulk entrants, fitted on the same windows, has errors between 0.0003 and 0.011 and beats persistence in all four. We introduced the two-class variant after seeing the one-class failure, so it is a diagnosis and not a pre-specified test; a prospective test is given in Section 5.5.

### 5.4 Projections and interventions

Parameter uncertainty is propagated by 4,000 draws (h from a Gamma posterior on events and exposure, b from a Beta posterior). Table 5 gives the modern share on the anniversary of the revision, the earliest date at which the lifecycle policy's twelve-month window could have elapsed for a feature deprecated at release, and one year later.

**Table 5. Projected modern share among npm servers with a runtime SDK dependency (median and 95% interval; run here).**

| Scenario | 2027-07-28 | 2028-07-28 |
|---|---|---|
| Constant inflow, b package-level (status quo) | 13.1% (12.1–14.1) | 19.9% (18.3–21.7) |
| Constant inflow, b namespace-level | 19.3% (17.9–20.7) | 27.4% (25.3–29.4) |
| No further entry | 16.4% (15.3–17.7) | 28.8% (26.4–31.5) |
| Entry keeps growing exponentially, b package-level | 8.3% (7.5–9.0) | 8.2% (7.4–8.9) |
| Migration hazard ×2 | 19.2% (17.7–20.9) | 31.1% (28.5–33.9) |
| Migration hazard ×5 | 34.7% (31.8–37.8) | 54.0% (50.3–57.8) |
| Birth share 0.58 (the PyPI level) | 43.9% (43.4–44.6) | 56.5% (55.6–57.6) |
| Birth share 0.90 | 63.0% (62.5–63.5) | 79.1% (78.5–79.7) |
| Birth share 0.90 and hazard ×5 | 75.4% (73.8–77.0) | 91.0% (89.8–92.1) |

Under every status-quo scenario, more than 80% of npm servers with a declared SDK dependency cannot speak 2026-07-28 on its first anniversary. With no further entry, the modern share reaches 50% after 3.9 years and 90% after 13.8 years. Moving the birth share to the level PyPI already shows does more by the anniversary than a fivefold increase in the migration hazard (43.9% against 34.7%). Only the combination passes 90%, and only in the second year. The intervals reflect parameter uncertainty under each scenario's assumptions; they do not cover the choice of scenario.

### 5.5 Falsifiable predictions and test protocol

Re-running `01`–`06` and `08` on a later date gives the test. We predict:

- **P1.** On 2027-07-28 the modern share among npm registry servers with a runtime SDK dependency is below 21% unless, before that date, `@modelcontextprotocol/sdk` receives a release that depends on or re-exports v2, or its `latest` tag moves to a 2.x line. If neither happens and the share exceeds 21%, the model is wrong.
- **P2.** The incumbent cohort's cumulative migration at 365 days is between 9% and 18% (the constant-hazard value is 12.6%).
- **P3.** If the default changes as in P1's exception, the modern share moves toward the new birth share with a time constant of about 1/(g + h), under half a year at current inflow, not at the incumbent hazard.
- **P4.** The share of bulk-published entrants born modern stays below 5% until the templates or generators that produce them change, and then changes abruptly.
- **P5.** On PyPI, the count of servers with an unbounded `mcp` range and a v1-only import falls mainly through new upper bounds (`mcp<2`), not through migration, because the SDK's own error message recommends the pin.

## 6 Discussion

**What governs evolution.** Version negotiation is the mechanism the specification provides for evolution. In the data it is not the binding constraint. Every TypeScript SDK release since 1.13.0 lists all earlier handshake revisions as supported, and the v2 SDKs retain them. What determined reach was which SDK release a server resolves to, and that was decided by three things outside the specification: the range convention of the package ecosystem, whether the new SDK kept its package name, and what the tools that generate new servers emit.

**Two conventions, two failure modes.** A caret range is a statement that minor releases are safe and major ones are not. It delivered three protocol revisions to most servers within a day and then stopped the fourth completely. An unbounded lower-bound range is a statement that the future is safe. It delivered the fourth revision to 23% of PyPI servers and broke two-thirds of them. Neither convention was chosen with a protocol in mind. A protocol whose reference implementations are libraries inherits the semantics of the host ecosystem's version ranges, and should expect minor-version delivery to be nearly free and major-version delivery to be nearly impossible to push.

**Implication for protocol design.** The 2026-07-28 changes were bundled with an SDK redesign. The measurements suggest that the cost of this bundling is carried almost entirely by reach. Had a 1.x release been able to answer stateless requests in addition to the handshake, the within-major channel would have applied. The v1 README states that this is not planned. Our data cannot say whether it was technically feasible; they do say what the alternative costs.

**Implication for deprecation.** The lifecycle policy sets a minimum of twelve months [1]. Corollary 4 with ε = 0.1 and the fitted g and h requires b ≥ 0.89. The measured npm birth share is 0.05 to 0.17. On present trends a client that dropped the handshake revisions on the anniversary would lose more than four-fifths of npm servers with a declared SDK dependency. The handshake will therefore persist in clients for years, and the stateless core will run alongside it, which is the dual-stack outcome familiar from other protocol transitions [12].

**Implication for who implements protocols.** Bulk publishers account for 69% of new npm servers and are born modern 0.14% of the time. We did not determine how these servers are produced. Templates, code generators and language models that reproduce the SDK usage they were built or trained on are candidate explanations, and all share one property: the protocol version is fixed upstream of the publisher. If so, the actors a standards body must reach to change the birth share are scaffolding and generator maintainers, a far smaller set than server authors. This is a hypothesis; P4 is its test.

**Other agent protocols.** The method needs only a registry or index, an SDK monoculture, and public package metadata. It should transfer to any agent protocol delivered through SDKs [2, 3]. The model's qualitative prediction is general: agent protocols are in a phase where g is large, so their trajectory is set by defaults at birth, and the window in which a breaking change is cheap closes when growth slows and Corollary 3 takes over.

## 7 Limitations

- **Declared dependencies, not behaviour.** We infer capability from metadata. We did not connect to any server. Outbound connections to remote endpoints were blocked in our environment, so the 61% of registry entries that are remote-only are outside the reach analysis entirely, as are opaque packages (29.5% of npm, 22.4% of PyPI). Remote services are operated continuously and may migrate faster; our conclusions concern installable servers.
- **Import breakage is static.** We did not execute any server or the v2 SDK; an attempt to install `mcp` 2.3.0 and run import statements was not permitted in the sandbox. The breakage claim rests on wheel contents and on the SDK's own stub module, whose text states that the import raises. Imports inside `try` blocks that catch the error and degrade gracefully would be misclassified as broken; packages using only retained module paths may still break on changed APIs and are reported as undetermined, not as working.
- **Direct requirements only.** Other dependencies of a PyPI package may cap `mcp` below 2 and prevent the float. This would lower both the nominal reach and the broken share.
- **Fresh-install reach is an upper envelope.** Caches, lockfiles in container images, and vendored installs hold servers at older SDKs. The declared floor bounds the other side. The gap between the two is large (44–55% against 99.5–99.7%) and we cannot locate real deployments within it.
- **Survivorship and selection.** The population is what the registry listed on 2026-11-22. Packages deleted earlier are missing, and history before a package's listing is reconstructed from npm, not from the registry. The registry is not the whole ecosystem and over-represents recent, possibly low-value, entries [6]; we report namespace-weighted and top-namespace-excluded figures for that reason, and they differ from package-level figures by a factor of about 1.5.
- **Packages are unweighted by use.** We have no download or invocation counts. A few widely used servers may have migrated while the long tail has not.
- **Short window.** The 2026-07-28 revision is 118 days old. A constant hazard fits those 118 days (half-window estimates 0.136 and 0.133 per year) but hazards can rise when ecosystem tooling changes. Projections are conditional on stated scenarios, and the entry-growth estimate (a 5.2-week doubling time from a noisy log-linear fit) is the least reliable input.
- **Post hoc model revision.** The two-class backtest was devised after the one-class model failed. The bulk threshold of five packages per namespace-week was chosen once and not tuned, but it was not pre-registered.
- **The cross-ecosystem comparison is observational.** Python and TypeScript server authors may differ in ways other than the default install. The PyPI figures are cross-sectional because we did not collect requirement history.
- **Clients are not measured.** Reach is a server-side quantity. Whether hosts send stateless requests is a separate question.
- **"Latest version" approximation.** A package's state is its most recently published stable version, which can differ from its `latest` dist-tag when maintenance releases are back-ported.

## 8 Conclusion

We measured how MCP revisions reach the servers listed in its official registry. Revisions delivered within a major SDK version reached about four-fifths of npm servers in a day, through floating dependency ranges. The 2026-07-28 revision, delivered across a major boundary, reached 7.0% in 118 days. npm incumbents migrate at 0.135 per year; a fifth of PyPI servers with a declared dependency float into an SDK they cannot import. New servers are born on the new SDK 5% of the time where it has a new package name and 58% of the time where it does not. A two-parameter model explains why the birth share matters more than the migration rate while an ecosystem is growing, and gives dated predictions. The contribution is a method, a census and a mechanism, limited to installable servers and to declared dependencies. For the question of how agent protocols will evolve, the measurements point to one answer: at the speed of their SDKs' default install, and across breaking boundaries hardly at all unless that default moves.

## References

[1] Model Context Protocol. "Key Changes" (changelog of specification revision 2026-07-28, including SEP-2567, SEP-2575, SEP-2322 and the feature lifecycle policy SEP-2596). modelcontextprotocol.io/specification/2026-07-28/changelog. Retrieved 2026-11-28 from the specification repository (github.com/modelcontextprotocol/modelcontextprotocol, `docs/specification/2026-07-28/changelog.mdx`). See also the release announcement, blog.modelcontextprotocol.io/posts/2026-07-28/.

[2] A. Ehtesham, A. Singh, G. K. Gupta, S. Kumar. "A survey of agent interoperability protocols: Model Context Protocol (MCP), Agent Communication Protocol (ACP), Agent-to-Agent Protocol (A2A), and Agent Network Protocol (ANP)." arXiv:2505.02279, 2025.

[3] Y. Yang, H. Chai, Y. Song, S. Qi, M. Wen, N. Li, J. Liao, H. Hu, J. Lin, G. Chang, W. Liu, Y. Wen, Y. Yu, W. Zhang. "A Survey of AI Agent Protocols." arXiv:2504.16736, 2025.

[4] X. Hou, Y. Zhao, S. Wang, H. Wang. "Model Context Protocol (MCP): Landscape, Security Threats, and Future Research Directions." arXiv:2503.23278, 2025.

[5] M. M. Hasan, H. Li, E. Fallahzadeh, G. K. Rajbahadur, B. Adams, A. E. Hassan. "Model Context Protocol (MCP) at First Glance: Studying the Security and Maintainability of MCP Servers." arXiv:2506.13538, 2025.

[6] H. Guo, Y. Hao, Y. Zhang, M. Xu, P. Lv, J. Chen, X. Cheng. "A Measurement Study of Model Context Protocol Ecosystem." arXiv:2509.25292, 2025.

[7] M. H. A. Majeed, M. Mahmoud, S. Nadi. "An Empirical Study of Model Context Protocol Applications." arXiv:2607.25635, 2026.

[8] O. Kraishan. "Same Name, Different Server: A Security Census of Silent Drift in the Model Context Protocol Ecosystem." arXiv:2609.14119, 2026.

[9] Linux Foundation. "Linux Foundation Announces the Formation of the Agentic AI Foundation (AAIF)." Press release, 9 December 2025. See also blog.modelcontextprotocol.io/posts/2025-12-09-mcp-joins-agentic-ai-foundation/.

[10] M. Honda, Y. Nishida, C. Raiciu, A. Greenhalgh, M. Handley, H. Tokuda. "Is it Still Possible to Extend TCP?" Proc. ACM Internet Measurement Conference (IMC), 2011, pp. 181–192.

[11] M. Thomson, T. Pauly. "Long-Term Viability of Protocol Extension Mechanisms." RFC 9170, December 2021.

[12] R. Holz, J. Hiller, J. Amann, A. Razaghpanah, T. Jost, N. Vallina-Rodriguez, O. Hohlfeld. "Tracking the deployment of TLS 1.3 on the Web: A story of experimentation and centralization." ACM SIGCOMM Computer Communication Review 50(3), 2020, pp. 3–15. doi:10.1145/3411740.3411742.

[13] A. Zerouali, E. Constantinou, T. Mens, G. Robles, J. González-Barahona. "An Empirical Analysis of Technical Lag in npm Package Dependencies." Proc. 17th International Conference on Software Reuse (ICSR), LNCS 10826, Springer, 2018, pp. 95–110.

[14] R. G. Kula, D. M. German, A. Ouni, T. Ishio, K. Inoue. "Do developers update their library dependencies? An empirical study on the impact of security advisories on library migration." Empirical Software Engineering 23(1), 2018, pp. 384–417.

[15] C. Bogart, C. Kästner, J. Herbsleb, F. Thung. "How to break an API: cost negotiation and community values in three software ecosystems." Proc. ACM SIGSOFT International Symposium on Foundations of Software Engineering (FSE), 2016, pp. 109–120. doi:10.1145/2950290.2950325.

[16] J. Farrell, G. Saloner. "Standardization, Compatibility, and Innovation." RAND Journal of Economics 16(1), 1985, pp. 70–83.
