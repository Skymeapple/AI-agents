"""Live collectors: structured APIs read directly by the Python process.

These need outbound network access to github.com, huggingface.co,
hn.algolia.com, api.npmjs.org and pypistats.org. Where that is blocked -- as in
some sandboxed agent environments -- they fail cleanly into the run log, the
report says which sources are down, and the daily agent session fills in what it
can through the inbox instead. A blocked source is never papered over.

Every observation from here is graded ``observed`` (``method="api"``): the
number came verbatim from a structured response.
"""
