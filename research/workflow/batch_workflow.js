export const meta = {
  name: 'ai-100-research-batch',
  description: 'Research a batch of AI open questions: write paper, independent review, iterate until accept or round cap',
  phases: [
    { title: 'Research', detail: 'one researcher per question writes paper + runs experiments' },
    { title: 'Review', detail: 'independent reviewer scores and critiques each draft' },
    { title: 'Revise', detail: 'researcher addresses every review point' },
  ],
}

const ROOT = '/home/user/AI-agents/research'
const MAX_REVIEWS = 4   // initial review + up to 3 revisions
const ACCEPT = 8

const RESEARCH_SCHEMA = {
  type: 'object',
  properties: {
    title: { type: 'string' },
    contribution_type: { type: 'string', description: 'theory | empirical-cpu | simulation | measurement-method | quantitative-model | synthesis-with-predictions | mixed' },
    one_paragraph_summary: { type: 'string' },
    experiments_run: { type: 'boolean' },
    files: { type: 'array', items: { type: 'string' } },
  },
  required: ['title', 'contribution_type', 'one_paragraph_summary', 'experiments_run', 'files'],
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
    overall: { type: 'number', description: '1-10 calibrated per the rubric' },
    decision: { type: 'string', enum: ['reject', 'major-revision', 'minor-revision', 'accept', 'strong-accept'] },
    fatal_issues: { type: 'array', items: { type: 'string' } },
    major_issues: { type: 'array', items: { type: 'string' } },
    minor_issues: { type: 'array', items: { type: 'string' } },
    unverifiable_citations: { type: 'array', items: { type: 'string' } },
    numbers_match_code_outputs: { type: 'boolean', description: 'true if no experiments, or reran and numbers match' },
    path_to_higher_score: { type: 'string', description: 'the single most valuable change that would raise the score' },
  },
  required: ['scores', 'overall', 'decision', 'fatal_issues', 'major_issues', 'minor_issues', 'unverifiable_citations', 'numbers_match_code_outputs', 'path_to_higher_score'],
}

const INTEGRITY = `
RESEARCH INTEGRITY (non-negotiable):
- Never invent a citation. Every reference must be verified to exist via web search (WebSearch or the Tavily tools; load them with ToolSearch). If you cannot verify one, remove it.
- Never invent data or results. Every number in the paper comes from (a) code you wrote AND ran in this environment, with outputs saved under results/, or (b) a verified, cited source. Label which.
- There is no GPU and no LLM API key here. CPU-scale experiments, simulations, formal proofs, analytical models, and analyses of public data are fine. Do not describe experiments on large models that you did not run; you may propose them as future work or as a pre-registered protocol.
- State the contribution plainly. Do not call it a breakthrough. Limitations section must be honest and specific.
- Python 3 with numpy, scipy, matplotlib, networkx is installed; pip install more if needed. Keep runs under ~20 minutes of CPU.`

function researchPrompt(q) {
  const d = `${ROOT}/${q.dir}`
  return `You are a senior AI researcher. Your project is one open question in AI (category: ${q.category}):

  "${q.question}"

Goal: produce the most original, rigorous, publishable contribution you can on this question, at the standard of a top venue (NeurIPS/ICML/ICLR for technical work; FAccT/AIES/Nature Machine Intelligence/economics or policy journals for socio-technical work — pick the fitting one and name it).

How:
1. Survey the literature with web search (current date is late 2026). Identify what is already known and the sharpest tractable gap.
2. Narrow to a specific sub-question you can actually make progress on with the resources here. A broad essay is NOT acceptable. Good contribution shapes: a formal model with proved results; a new measurement or evaluation methodology validated on real data or simulation; a CPU-scale experiment on small models/toy settings that isolates a mechanism; a quantitative model built on sourced data with sensitivity analysis; a synthesis that yields novel, falsifiable predictions with a concrete test protocol.
3. Do the work. Put code in ${d}/code/, run it, save outputs/figures in ${d}/results/. Make it reproducible (a README or run command).
4. Write the paper to ${d}/paper.md in English, in venue format: Title; Abstract; 中文摘要 (Chinese abstract); 1 Introduction (with explicit contribution list); 2 Related Work; 3+ Method/Model/Theory; Results; Discussion; Limitations; Conclusion; References (full, verified). Target 4,000–8,000 words. Include the target venue on a line under the title.
${INTEGRITY}

Work only inside ${d}. Return the structured summary.`
}

function reviewPrompt(q, round, prevIssues) {
  const d = `${ROOT}/${q.dir}`
  const prev = prevIssues ? `\nThe previous review round raised these issues; check each one is actually resolved, not just acknowledged:\n${prevIssues}\n` : ''
  return `You are an independent, demanding senior reviewer (area chair level) at a top venue. You did not write this paper and owe the authors nothing. Review the submission at ${d}/paper.md (research question: "${q.question}"), including its code in ${d}/code/ and outputs in ${d}/results/.

Do all of this:
- Rerun the code (or the main script) and check that the numbers and figures in the paper match the outputs. Any mismatch or result not backed by run code or a verified source is a fatal issue.
- Spot-check at least 6 references by web search (WebSearch or Tavily via ToolSearch). Any reference that does not exist or does not say what is claimed goes in unverifiable_citations and is a fatal issue.
- Search for prior work to judge novelty honestly: has this been done? Is the contribution incremental?
- Check proofs and derivations line by line where present. Check statistical claims (sample sizes, uncertainty, baselines, ablations).
- Judge whether the sub-question is sharp and whether the conclusions follow from the evidence.
${prev}
Calibration for overall score (be strict; most papers are not 8+):
 3-4 = clear reject; 5 = weak reject; 6 = borderline / workshop level; 7 = solid accept at a good venue; 8 = strong accept at a top venue (genuinely novel, rigorous, significant); 9 = best-paper calibre; 10 = landmark.

Write your full review to ${d}/reviews/review_r${round}.md (summary, strengths, weaknesses, questions, scores, decision). Then return the structured verdict.`
}

function revisePrompt(q, round, review) {
  const d = `${ROOT}/${q.dir}`
  return `You are the author of the paper at ${d}/paper.md (research question: "${q.question}"). An independent reviewer's full report is at ${d}/reviews/review_r${round}.md. Their overall score was ${review.overall}/10 (${review.decision}). Acceptance needs 8+ with no fatal issues.

Fatal: ${JSON.stringify(review.fatal_issues)}
Major: ${JSON.stringify(review.major_issues)}
Minor: ${JSON.stringify(review.minor_issues)}
Unverifiable citations: ${JSON.stringify(review.unverifiable_citations)}
Highest-leverage change suggested: ${review.path_to_higher_score}

Read the full review. Then substantively improve the work: fix every fatal and major issue, run new experiments/analyses or tighten proofs where needed, sharpen the contribution, and update ${d}/paper.md (and code/results). If a criticism is wrong, rebut it with evidence rather than ignoring it. Reducing a claim to what the evidence supports is a legitimate fix; hiding a problem is not.
Write ${d}/reviews/response_r${round}.md: a point-by-point response listing each issue and exactly what changed.
${INTEGRITY}

Return the structured summary of the revised paper.`
}

function issuesText(r) {
  return [...r.fatal_issues.map(x => 'FATAL: ' + x), ...r.major_issues.map(x => 'MAJOR: ' + x), ...r.unverifiable_citations.map(x => 'CITATION: ' + x)].join('\n')
}

const items = args
log(`Batch of ${items.length} projects: ${items.map(q => q.id).join(', ')}`)

const results = await pipeline(items, async (q) => {
  const tag = `p${String(q.id).padStart(3, '0')}`
  let paper = await agent(researchPrompt(q), { label: `research:${tag}`, phase: 'Research', schema: RESEARCH_SCHEMA })
  if (!paper) return { id: q.id, status: 'research-failed' }
  const history = []
  let review = null
  for (let round = 1; round <= MAX_REVIEWS; round++) {
    review = await agent(reviewPrompt(q, round, review ? issuesText(review) : null), { label: `review:${tag}:r${round}`, phase: 'Review', schema: REVIEW_SCHEMA })
    if (!review) break
    history.push({ round, overall: review.overall, decision: review.decision, fatal: review.fatal_issues.length, major: review.major_issues.length })
    const accepted = review.overall >= ACCEPT && review.fatal_issues.length === 0 && review.unverifiable_citations.length === 0 && review.numbers_match_code_outputs
    log(`${tag} r${round}: ${review.overall}/10 ${review.decision}${accepted ? ' — ACCEPTED' : ''}`)
    if (accepted || round === MAX_REVIEWS) break
    const rev = await agent(revisePrompt(q, round, review), { label: `revise:${tag}:r${round}`, phase: 'Revise', schema: RESEARCH_SCHEMA })
    if (rev) paper = rev
  }
  return {
    id: q.id, question: q.question, dir: q.dir, title: paper.title,
    contribution_type: paper.contribution_type, summary: paper.one_paragraph_summary,
    final_score: review ? review.overall : null, final_decision: review ? review.decision : null,
    accepted: !!review && review.overall >= ACCEPT && review.fatal_issues.length === 0 && review.unverifiable_citations.length === 0,
    history,
  }
})

return results.filter(Boolean)
