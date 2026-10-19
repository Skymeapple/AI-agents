"""The live collectors: GitHub, Hugging Face, Hacker News, npm and PyPI."""

from __future__ import annotations

import datetime as dt

import requests

from ..models import App, AppKind, Method, Observation, canonical_id
from .base import Collected, get_json, today

GITHUB_API = "https://api.github.com/search/repositories"
HF_API = "https://huggingface.co/api"
HN_API = "https://hn.algolia.com/api/v1/search_by_date"
NPM_API = "https://api.npmjs.org/downloads/point/last-week"
PYPI_API = "https://pypistats.org/api/packages"


def _obs(app: App, metric: str, value: float, url: str, collector: str, *,
         on: dt.date, period: float | None = None, via: str | None = None, note: str = "") -> Observation:
    return Observation(app.id, metric, float(value), on, url, Method.API, collector, period, via, note)


def collect_github(watch: dict, session: requests.Session, token: str | None = None,
                   per_query: int = 50) -> Collected:
    """Stars on repos under the watched topics and terms.

    Two queries per term. Established repos (``stars:>200``, recently pushed)
    give a stars level that becomes a measured delta on the next run. Repos
    created in the last 30 days also give an immediate signal: they started at
    zero, so their whole star count is a gain over their age -- a stated period
    gain, computed from ``created_at``.
    """
    out = Collected("api:github")
    on = today()
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    recent = (on - dt.timedelta(days=30)).isoformat()
    pushed = (on - dt.timedelta(days=14)).isoformat()
    terms = [("topic:" + t, t) for t in watch["github_topics"]] + [
        (f'"{t}" in:name,description', t) for t in watch.get("auto_terms", [])
    ]
    for query, via in terms:
        for qualifier in (f"stars:>200 pushed:>{pushed}", f"created:>{recent} stars:>50"):
            try:
                payload = get_json(session, GITHUB_API, {
                    "q": f"{query} {qualifier}", "sort": "stars", "order": "desc", "per_page": per_query,
                }, headers)
            except (requests.RequestException, ValueError) as exc:
                out.errors.append(f"github {query!r}: {exc}")
                continue
            for repo in payload.get("items", []):
                url = repo["html_url"]
                app = App(canonical_id(url), repo["full_name"], url, AppKind.REPO,
                          repo.get("description") or "", list(repo.get("topics", []))[:5], on)
                source = f"{GITHUB_API}?q={query} {qualifier}"
                out.add(app, _obs(app, "stars", repo["stargazers_count"], source, out.collector, on=on, via=via))
                created = dt.date.fromisoformat(repo["created_at"][:10])
                age = (on - created).days
                if created.isoformat() > recent and age >= 1:
                    out.add(app, _obs(app, "stars_gained", repo["stargazers_count"], source, out.collector,
                                      on=on, period=age, via=via, note=f"created {created}"))
    return out


def collect_huggingface(session: requests.Session, limit: int = 100) -> Collected:
    out = Collected("api:huggingface")
    on = today()
    for kind, path in ((AppKind.SPACE, "spaces"), (AppKind.MODEL, "models")):
        url = f"{HF_API}/{path}"
        try:
            items = get_json(session, url, {"sort": "trendingScore", "direction": -1, "limit": limit})
        except (requests.RequestException, ValueError) as exc:
            out.errors.append(f"huggingface {path}: {exc}")
            continue
        for item in items:
            item_id = item.get("id") or item.get("modelId")
            page = f"https://huggingface.co/{'spaces/' if kind is AppKind.SPACE else ''}{item_id}"
            tags = [t for t in item.get("tags", []) if ":" not in t][:4]
            if item.get("pipeline_tag"):
                tags.insert(0, item["pipeline_tag"])
            app = App(canonical_id(page), item_id, page, kind, "", tags, on)
            out.add(app, _obs(app, "likes", item.get("likes", 0), f"{url}?sort=trendingScore", out.collector, on=on))
    return out


def collect_hackernews(watch: dict, session: requests.Session, days: int = 14, min_points: int = 20) -> Collected:
    out = Collected("api:hackernews")
    on = today()
    since = int((dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).timestamp())
    for query in watch["hn_queries"]:
        try:
            payload = get_json(session, HN_API, {
                "tags": "show_hn", "query": query, "hitsPerPage": 100,
                "numericFilters": f"created_at_i>{since},points>={min_points}",
            })
        except (requests.RequestException, ValueError) as exc:
            out.errors.append(f"hackernews {query!r}: {exc}")
            continue
        for hit in payload.get("hits", []):
            item = f"https://news.ycombinator.com/item?id={hit['objectID']}"
            target = hit.get("url") or item
            title = (hit.get("title") or "").removeprefix("Show HN:").strip()
            kind = AppKind.REPO if "github.com/" in target else AppKind.PRODUCT
            try:
                app = App(canonical_id(target), title.split(" – ")[0].split(" - ")[0][:80], target, kind, title, [], on)
            except ValueError:
                continue
            out.add(app, _obs(app, "points", hit.get("points") or 0, item, out.collector, on=on, via=query,
                              note=f"posted {hit.get('created_at', '')[:10]}"))
    return out


def collect_packages(watch: dict, session: requests.Session) -> Collected:
    """Weekly downloads of watched AI SDKs: developer adoption, not hype."""
    out = Collected("api:packages")
    on = today()
    for name in watch["npm"]:
        try:
            payload = get_json(session, f"{NPM_API}/{name}")
        except (requests.RequestException, ValueError) as exc:
            out.errors.append(f"npm {name}: {exc}")
            continue
        page = f"https://www.npmjs.com/package/{name}"
        app = App(canonical_id(page), name, page, AppKind.PACKAGE, "npm package", ["sdk"], on)
        out.add(app, _obs(app, "downloads_period", payload.get("downloads", 0), f"{NPM_API}/{name}",
                          out.collector, on=on, period=7))
    for name in watch["pypi"]:
        try:
            payload = get_json(session, f"{PYPI_API}/{name}/recent")
        except (requests.RequestException, ValueError) as exc:
            out.errors.append(f"pypi {name}: {exc}")
            continue
        page = f"https://pypi.org/project/{name}"
        app = App(canonical_id(page), name, page, AppKind.PACKAGE, "PyPI package", ["sdk"], on)
        weekly = (payload.get("data") or {}).get("last_week", 0)
        out.add(app, _obs(app, "downloads_period", weekly, f"{PYPI_API}/{name}/recent",
                          out.collector, on=on, period=7))
    return out
