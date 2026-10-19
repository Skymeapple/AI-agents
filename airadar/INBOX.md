# Inbox format

An inbox file is how observations collected outside the Python collectors -- by
the daily agent session using web tools, or by hand -- enter the radar. The
loader (`airadar.inbox`) rejects a file that breaks any rule below, naming the
offending observation, rather than silently dropping or defaulting fields.

```json
{
  "collected_at": "2026-10-19",
  "collector": "agent:github-trending",
  "observations": [
    {
      "app": {
        "name": "owner/repo or product name",
        "url": "https://github.com/owner/repo",
        "kind": "repo",
        "description": "one sentence: what it does",
        "tags": ["coding-agent", "mcp"]
      },
      "metric": "stars_gained",
      "value": 4774,
      "period_days": 7,
      "observed_on": "2026-10-19",
      "source_url": "https://github.com/trending?since=weekly",
      "method": "page_extract",
      "note": "optional free text"
    }
  ]
}
```

## Rules

- `app.url` is required: it is the identity key. GitHub repos, Hugging Face
  Spaces/models and npm/PyPI packages are canonicalised from it; anything else
  is keyed by host + first path segment.
- `app.kind`: `repo`, `space`, `model`, `package`, `product`, `consumer_app`.
- `metric` must be one of the vocabulary below. Level metrics are cumulative
  counts at a point in time; period metrics are amounts over `period_days`,
  which they must carry.

  | metric | type | meaning |
  | --- | --- | --- |
  | `stars` | level | GitHub stars |
  | `likes` | level | Hugging Face likes |
  | `downloads` | level | cumulative downloads |
  | `users` | level | reported users (state MAU/WAU in `note`) |
  | `visits_monthly` | level | monthly web visits |
  | `revenue_annualized_usd` | level | reported ARR / run-rate |
  | `points` | level | Hacker News / Product Hunt points or upvotes |
  | `stars_gained` | period | stars gained over the period |
  | `downloads_period` | period | downloads over the period |
  | `growth_pct` | period | reported % growth over the period (e.g. 40 = +40%) |

- `method` states how the number was obtained, which sets its provenance:
  `api` (read verbatim from a structured API response -> observed),
  `page_extract` (read off a rendered web page, possibly via a model -> sourced,
  needs review), `reported` (stated by an article, filing or company -> sourced,
  needs review).
- `source_url` is required for every observation. A number with no URL is not
  accepted, however plausible.
- Never estimate, interpolate or fill in a value. If a page does not state a
  number, do not record the observation.
