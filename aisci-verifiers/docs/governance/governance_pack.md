# Governance pack (initiative I6)

Version 1.0, 1 November 2026. These are drafts for a program's partner institutions to adopt or amend; they are not legal advice and have not been adopted by anyone yet.

## G1. Authorship and disclosure policy for agent contributions (task 6.1)

1. **AI systems are not authors.** Authorship requires accountability, which only people and institutions can hold. This matches the stated policies of major publishers (Nature, Science and ICMJE) as of 2025.
2. **Disclose by role.** Every paper states, in Methods, which of these steps an agent performed: question selection, literature review, hypothesis generation, experiment design, experiment execution, analysis, and writing. It also names the system, its version, and whether a human approved each step.
3. **Ship the provenance.** Each agent-derived claim in the paper links to its world-model record (claim, evidence, code, data digest). Reviewers get the full run log.
4. **Credit people for what they did.** The humans who chose the problem, curated the data, validated results and took responsibility are listed under CRediT roles.
5. **Correction path.** If a world-model claim is later retracted, the store flags every derived claim (implemented in `WorldModel.retract`), and the authors must check every paper that cites a flagged claim.

## G2. Dual-use screening (task 6.2)

- Every agent-proposed protocol passes `aisci.screen.screen` and the lab safety interlocks before execution. A block is logged as `blocked_by_safety`, never silently dropped.
- High-hazard reagents need a named human sign-off in the protocol (`signoff`), which the interlock enforces.
- The controlled list in code is placeholder names. Deployment must load the institution's own controlled-agent and precursor lists, maintained by its biosafety office.
- **Red-team record.** The first round found 9 bypasses in 50 attempts, because zero-width characters survived text conversion. After the fix, there were 0 bypasses in 50 attempts on each of 4 seeds, and 0 false blocks on 11 benign reagents. A string screen is one layer, so institutional biosafety committee review remains mandatory for any agent campaign in biology or chemistry.

## G3. Pre-registration template (task 6.3)

Register before the first experiment of any campaign:

| Field | Content |
| --- | --- |
| Campaign id | Matches `campaign_id` in the experiment store |
| Question and hypothesis | One falsifiable sentence each |
| Primary metric | One metric, with its unit |
| Sample size and power | With the calculation |
| Stopping rule | Budget cap plus the conditions for an early stop |
| Kill criterion | The result that ends the line of work |
| Agent configuration | System, version, tools, autonomy level (L1 to L5), human checkpoints |
| Safety | Hazard class, sign-offs, dual-use screen version |
| Analysis plan | Tests, corrections for multiple comparisons, exclusions |
| Data release | Store location, licence (default CC-BY-4.0), embargo |

## G4. Independent replication fund (task 6.4)

- 10% of lab time goes to replications chosen by an independent committee, not by the campaign team.
- Every finding the program announces as novel needs one external replication before it is called a discovery (Gate 3 in the roadmap).
- Replications use the original protocol records from the store, which makes them exact by construction. A conceptual replication is also funded where possible.
