# Working in this repository

sciscout tracks research directions, scores them, and models the investment
case. README.md has the full design; read the section you are changing before
you change it.

## Commands

- Tests: `python -m pytest tests/ -q` (about 2 seconds, no network)
- Demo corpus: `python scripts/make_demo_corpus.py`
- CLI: `python -m sciscout.cli <discover|rank|model|report|scenarios|sectors|harvest>`

Only `harvest` touches the network (arXiv, OpenAlex). Everything else runs
offline against `data/demo_corpus.json`.

## Rules

- **Run before you report.** Do not say a change works until you have run the
  tests, and the relevant CLI command if behaviour changed. Report failures
  with their output. If you could not run something, say so.
- **Stay in scope.** Change what was asked. No unrequested refactors, new
  abstractions, new dependencies or new files. Mention other problems you
  notice; do not fix them unasked.
- **Every number carries provenance.** A figure is `observed`, `sourced` or
  `assumed` (see `sciscout/provenance.py`). Never add a bare numeric constant
  to scoring, priors or the investment model, and never upgrade an `assumed`
  value to `sourced` without a named reference.
- **Do not invent data or citations.** If a prior has no source, mark it
  `assumed` with a rationale.
- **Never weaken a test to make it pass.** Fix the code, or explain why the
  test is wrong.
- **Look things up.** Check the code or the README instead of answering from
  memory about how a module behaves.

## Long sessions

- Start a fresh session for each unrelated task; do not carry one session
  across many.
- When a decision is made that later work depends on, write it into the code,
  a test or the README. Do not rely on it surviving in the conversation.
- After the conversation has been summarized, re-read this file and the files
  you are editing before continuing.
