export const meta = {
  name: 'ai-100-focus',
  description: 'Deep iteration on the most promising AI-100 papers: independent review and ambitious revision for up to 8 review rounds',
  phases: [
    { title: 'Revise', detail: 'author substantially strengthens the paper against all prior reviews' },
    { title: 'Review', detail: 'fresh independent reviewer scores the current version' },
  ],
}

const ROOT = '/home/user/AI-agents/research'
const MAX_ROUND = 8      // highest review round number allowed
const ACCEPT = 8
const PLATEAU = 3        // stop after this many consecutive reviews with no new best score

const RESEARCH_SCHEMA = {
  type: 'object',
  properties: {
    title: { type: 'string' },
    one_paragraph_summary: { type: 'string' },
    what_changed: { type: 'string' },
  },
  required: ['title', 'one_paragraph_summary', 'what_changed'],
}

const REVIEW_SCHEMA = {
  type: 'object',
  properties: {
    scores: {
      type: 'object',
      properties: {
        novelty: { type: 'number' }, rigor: { type: 'number' }, correctness: { type: 'number' },
        significance: { type: 'number' }, clarity: { type: 'number' }, reproducibility: { type: 'number' },
      },
      required: ['novelty', 'rigor', 'correctness', 'significance', 'clarity', 'reproducibility'],
    },
    overall: { type: 'number' },
    decision: { type: 'string', enum: ['reject', 'major-revision', 'minor-revision', 'accept', 'strong-accept'] },
    fatal_issues: { type: 'array', items: { type: 'string' } },
    major_issues: { type: 'array', items: { type: 'string' } },
    minor_issues: { type: 'array', items: { type: 'string' } },
    unverifiable_citations: { type: 'array', items: { type: 'string' } },
    numbers_match_code_outputs: { type: 'boolean' },
    path_to_higher_score: { type: 'string' },
  },
  required: ['scores', 'overall', 'decision', 'fatal_issues', 'major_issues', 'minor_issues', 'unverifiable_citations', 'numbers_match_code_outputs', 'path_to_higher_score'],
}

const INTEGRITY = `
RESEARCH INTEGRITY (non-negotiable):
- Never invent a citation; verify every reference by web search (WebSearch or Tavily via ToolSearch). Remove what you cannot verify.
- Never invent data or results. Every number comes from code you ran here (outputs saved under results/) or from a verified, cited source.
- No GPU and no LLM API key here. CPU-scale experiments, simulations, proofs, analytical models and public-data analyses only. Do not describe experiments you did not run.
- Do not call the work a breakthrough. Keep the Limitations section honest and specific.
- Python 3 with numpy, scipy, matplotlib, networkx is installed; pip install more if needed. Keep any single run under ~30 minutes of CPU.`

function revisePrompt(q, round) {
  const d = `${ROOT}/${q.dir}`
  return `You are the lead author of the paper at ${d}/paper.md (research question: "${q.question}"). This paper was selected as one of the ~10 most promising in a 100-paper programme, and it now gets extended iteration. The goal is a paper an independent top-venue area chair would score 8+/10 (strong accept): genuinely novel, rigorous, significant.

Read ALL prior reviews and responses in ${d}/reviews/ (the latest is review_r${round}.md). Across reviewers the binding constraints are usually NOVELTY and SIGNIFICANCE (typically scored 4-5): the work is called incremental, or its key results are known. Patching small issues will not get to 8. So:
1. Diagnose honestly why the paper is not yet a strong accept. What would make an expert say "I did not know that, and it matters"?
2. Find and pursue the strongest available new contribution within reach: a sharper theorem, a surprising empirical regularity on real public data, a new method that clearly beats baselines, or a decisive test that settles a debated point. Re-scope, retitle, or cut weak material if that yields a stronger paper.
3. Do the work (proofs, code, experiments, public-data analysis) and fix every fatal and major issue from review_r${round}.md. Rebut wrong criticisms with evidence.
4. Rewrite ${d}/paper.md as a coherent paper, not a patched one (keep the 中文摘要). Write ${d}/reviews/response_r${round}.md, a point-by-point response that also states the new contribution.
${INTEGRITY}

Return the structured summary.`
}

function reviewPrompt(q, round) {
  const d = `${ROOT}/${q.dir}`
  return `You are an independent, demanding senior reviewer (area chair level) at a top venue. You did not write this paper. Review the current version at ${d}/paper.md (research question: "${q.question}"), with its code in ${d}/code/ and outputs in ${d}/results/.

Earlier reviews and author responses are in ${d}/reviews/. Read them so you can check that the earlier issues are actually resolved, but form your OWN judgement of the current paper; you are not bound by earlier scores in either direction.

Required:
- Rerun the main code and check that the paper's numbers and figures match the outputs. Unbacked or mismatched results are fatal.
- Spot-check at least 6 references by web search (WebSearch or Tavily via ToolSearch), prioritising ones added since the last round. Nonexistent or misdescribed references are fatal.
- Search for prior work and judge novelty honestly.
- Check proofs line by line. Check statistics: uncertainty, baselines, ablations.

Calibration (strict): 3-4 clear reject; 5 weak reject; 6 borderline/workshop; 7 solid accept at a good venue; 8 strong accept at a top venue; 9 best-paper calibre; 10 landmark.

Write the full review to ${d}/reviews/review_r${round}.md, then return the structured verdict.`
}

const accepted = r => r && r.overall >= ACCEPT && r.fatal_issues.length === 0 && r.unverifiable_citations.length === 0 && r.numbers_match_code_outputs

// args: [{id, dir, question, next: 'revise'|'review', round, best}]
// next='revise' with round=N: revise against existing review_rN, then review r(N+1)
// next='review' with round=N: review_rN is the next review to write
const items = args
log(`Focus set: ${items.map(q => q.dir).join(', ')}`)

const results = await pipeline(items, async (q) => {
  const tag = q.dir.split('/').pop()
  let round = q.round
  let best = q.best || 0
  let sinceBest = 0
  let last = null
  const history = []
  if (q.next === 'revise') {
    await agent(revisePrompt(q, round), { label: `revise:${tag}:r${round}`, phase: 'Revise', schema: RESEARCH_SCHEMA })
    round += 1
  }
  while (round <= MAX_ROUND) {
    const r = await agent(reviewPrompt(q, round), { label: `review:${tag}:r${round}`, phase: 'Review', schema: REVIEW_SCHEMA })
    if (!r) break
    last = r
    history.push({ round, overall: r.overall, decision: r.decision, novelty: r.scores.novelty, significance: r.scores.significance })
    if (r.overall > best) { best = r.overall; sinceBest = 0 } else { sinceBest += 1 }
    log(`${tag} r${round}: ${r.overall}/10 ${r.decision} (best ${best})`)
    if (accepted(r)) { log(`${tag} ACCEPTED at r${round}`); break }
    if (sinceBest >= PLATEAU) { log(`${tag} stopped: ${PLATEAU} rounds without improvement`); break }
    if (round === MAX_ROUND) break
    await agent(revisePrompt(q, round), { label: `revise:${tag}:r${round}`, phase: 'Revise', schema: RESEARCH_SCHEMA })
    round += 1
  }
  return { dir: q.dir, question: q.question, best, final: last ? last.overall : null, decision: last ? last.decision : null, accepted: accepted(last), history }
})

return results.filter(Boolean)
