"""Domain model: applications, and dated observations of them.

Two things are kept apart, for the same reason sciscout keeps works apart from
tracks:

*An* :class:`App` *is an identity* -- a repository, a Hugging Face Space, a
product. It has a name, a URL and some descriptive tags, and nothing about it
changes from day to day.

*An* :class:`Observation` *is a dated measurement of an app* -- 4,774 stars
gained this week, $500M annualised revenue as of September. Observations are
append-only. Growth is never stored; it is always recomputed from observations,
so a change to the growth model reapplies to the whole history and a backtest
can replay any past day exactly as it looked then.

Every observation says how it was obtained. That ``method`` sets its provenance
grade: a number read verbatim from an API is ``observed``; one read off a
rendered page or stated in an article is ``sourced`` and flagged for review,
because the radar did not measure it and cannot reproduce it.
"""

from __future__ import annotations

import datetime as dt
import enum
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

from sciscout.provenance import Provenance, observed, sourced


class MetricType(enum.Enum):
    LEVEL = "level"  # a cumulative count at a point in time
    PERIOD = "period"  # an amount accrued over ``period_days``


@dataclass(frozen=True)
class MetricSpec:
    name: str
    type: MetricType
    # Metrics in the same family measure the same quantity, so a period metric
    # can be read against its family's level (stars gained against total stars)
    # and absolute velocities are only ever ranked within a family.
    family: str


METRICS: dict[str, MetricSpec] = {
    spec.name: spec
    for spec in (
        MetricSpec("stars", MetricType.LEVEL, "stars"),
        MetricSpec("likes", MetricType.LEVEL, "likes"),
        MetricSpec("downloads", MetricType.LEVEL, "downloads"),
        MetricSpec("users", MetricType.LEVEL, "users"),
        MetricSpec("visits_monthly", MetricType.LEVEL, "visits"),
        MetricSpec("revenue_annualized_usd", MetricType.LEVEL, "revenue"),
        MetricSpec("points", MetricType.LEVEL, "points"),
        MetricSpec("stars_gained", MetricType.PERIOD, "stars"),
        MetricSpec("downloads_period", MetricType.PERIOD, "downloads"),
        MetricSpec("growth_pct", MetricType.PERIOD, "reported_growth"),
    )
}


class Method(enum.Enum):
    """How a number was obtained, which decides how far to trust it."""

    API = "api"
    PAGE_EXTRACT = "page_extract"
    REPORTED = "reported"


class AppKind(enum.Enum):
    REPO = "repo"
    SPACE = "space"
    MODEL = "model"
    PACKAGE = "package"
    PRODUCT = "product"
    CONSUMER_APP = "consumer_app"


_SLUG_RE = re.compile(r"[^a-z0-9._/@-]+")


def canonical_id(url: str) -> str:
    """Stable identity key for an app, derived from its URL.

    The same repository reached from GitHub trending and from a Hacker News
    launch must collapse to one app, or its growth is split across two records
    and neither looks fast. So identity comes from the URL, normalised per host:

    - ``github.com/Owner/Repo/tree/main`` -> ``gh:owner/repo``
    - ``huggingface.co/spaces/a/b`` -> ``hf-space:a/b``; ``huggingface.co/a/b``
      -> ``hf-model:a/b``
    - ``npmjs.com/package/x`` -> ``npm:x``; ``pypi.org/project/x`` -> ``pypi:x``
    - anything else -> ``web:host`` plus the first path segment when the host is
      a shared platform (``apps.apple.com``, ``chromewebstore.google.com``) or
      a company domain hosting several products (``openai.com/codex``), rather
      than the product's own domain.
    """
    raw = url.strip()
    if "://" not in raw:
        raw = "https://" + raw
    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    parts = [p for p in parsed.path.split("/") if p]
    if not host:
        raise ValueError(f"cannot derive an app identity from url {url!r}")

    def slug(*segments: str) -> str:
        return _SLUG_RE.sub("-", "/".join(segments).lower()).strip("-")

    if host == "github.com" and len(parts) >= 2:
        return "gh:" + slug(parts[0], parts[1].removesuffix(".git"))
    if host == "huggingface.co" and len(parts) >= 3 and parts[0] == "spaces":
        return "hf-space:" + slug(parts[1], parts[2])
    if host == "huggingface.co" and len(parts) >= 2 and parts[0] not in {
        "spaces", "datasets", "papers", "blog", "docs", "api",
    }:
        return "hf-model:" + slug(parts[0], parts[1])
    if host == "npmjs.com" and len(parts) >= 2 and parts[0] == "package":
        return "npm:" + slug(*parts[1:3] if parts[1].startswith("@") else parts[1:2])
    if host == "pypi.org" and len(parts) >= 2 and parts[0] == "project":
        return "pypi:" + slug(parts[1])
    shared = {
        "apps.apple.com", "play.google.com", "chromewebstore.google.com",
        "producthunt.com", "news.ycombinator.com", "chatgpt.com",
        # company domains with several products: one product's growth must not
        # merge into another's
        "openai.com", "anthropic.com", "google.com", "blog.google", "microsoft.com",
        "meta.com", "about.meta.com", "amazon.com", "aws.amazon.com", "apple.com",
        "nvidia.com", "x.ai", "mistral.ai", "bytedance.com", "alibabacloud.com",
    }
    if host in shared and parts:
        return "web:" + slug(host, parts[0] if host != "news.ycombinator.com" else parsed.query)
    return "web:" + slug(host)


@dataclass
class App:
    id: str
    name: str
    url: str
    kind: AppKind
    description: str = ""
    tags: list[str] = field(default_factory=list)
    first_seen: dt.date | None = None

    def text(self) -> str:
        """Name, description and tags together, for lexical clustering."""
        name = self.name.replace("/", " ").replace("-", " ").replace("_", " ")
        return f"{name}\n{self.description}\n{' '.join(self.tags)}"

    def merge(self, other: "App") -> None:
        """Fold a later sighting into this record.

        Tags accumulate, since two collectors describing one app from different
        angles are both right. A description is only filled, never overwritten:
        a stable description keeps clustering stable from one day to the next.
        """
        for tag in other.tags:
            if tag not in self.tags:
                self.tags.append(tag)
        if not self.description and other.description:
            self.description = other.description
        if other.first_seen and (not self.first_seen or other.first_seen < self.first_seen):
            self.first_seen = other.first_seen

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "url": self.url,
            "kind": self.kind.value,
            "description": self.description,
            "tags": list(self.tags),
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "App":
        return cls(
            id=raw["id"],
            name=raw["name"],
            url=raw["url"],
            kind=AppKind(raw["kind"]),
            description=raw.get("description", ""),
            tags=list(raw.get("tags", [])),
            first_seen=dt.date.fromisoformat(raw["first_seen"]) if raw.get("first_seen") else None,
        )


@dataclass(frozen=True)
class Observation:
    app_id: str
    metric: str
    value: float
    observed_on: dt.date
    source_url: str
    method: Method
    collector: str
    period_days: float | None = None
    # The search term that surfaced this app, when a collector searched for it.
    # Lets the self-tuner tell productive watch terms from dead ones.
    via: str | None = None
    note: str = ""
    # When the radar learned the number, as distinct from ``observed_on``, the
    # date it describes. A figure for June reported in October was not known in
    # July, and a backtest replaying July must not see it.
    collected_on: dt.date | None = None

    def __post_init__(self) -> None:
        spec = METRICS.get(self.metric)
        if spec is None:
            raise ValueError(f"unknown metric {self.metric!r}; expected one of {sorted(METRICS)}")
        if spec.type is MetricType.PERIOD and not (self.period_days and self.period_days > 0):
            raise ValueError(f"period metric {self.metric!r} needs a positive period_days")
        if spec.type is MetricType.LEVEL and self.period_days is not None:
            raise ValueError(f"level metric {self.metric!r} must not carry period_days")
        if self.metric != "growth_pct" and self.value < 0:
            raise ValueError(f"{self.metric} cannot be negative (got {self.value})")
        if not self.source_url.strip():
            raise ValueError("every observation needs a source_url")
        if self.collected_on is not None and self.collected_on < self.observed_on:
            raise ValueError("collected_on cannot precede observed_on")

    @property
    def known_on(self) -> dt.date:
        return self.collected_on or self.observed_on

    @property
    def spec(self) -> MetricSpec:
        return METRICS[self.metric]

    def key(self) -> tuple:
        """Identity for de-duplication: the same number from the same page on the
        same day is one observation, however many times it is ingested."""
        return (self.app_id, self.metric, self.period_days, self.observed_on, self.source_url)

    def provenance(self) -> Provenance:
        if self.method is Method.API:
            return observed(f"{self.metric} read from {self.collector} API on {self.observed_on}")
        how = "read off a rendered page" if self.method is Method.PAGE_EXTRACT else "reported"
        return sourced(
            f"{self.metric} {how} ({self.collector})",
            reference=self.source_url,
            retrieved=self.observed_on.isoformat(),
            needs_review=True,
        )

    def to_dict(self) -> dict:
        return {
            "app_id": self.app_id,
            "metric": self.metric,
            "value": self.value,
            "period_days": self.period_days,
            "observed_on": self.observed_on.isoformat(),
            "source_url": self.source_url,
            "method": self.method.value,
            "collector": self.collector,
            "via": self.via,
            "note": self.note,
            "collected_on": self.collected_on.isoformat() if self.collected_on else None,
        }

    @classmethod
    def from_dict(cls, raw: dict) -> "Observation":
        return cls(
            app_id=raw["app_id"],
            metric=raw["metric"],
            value=float(raw["value"]),
            observed_on=dt.date.fromisoformat(raw["observed_on"]),
            source_url=raw["source_url"],
            method=Method(raw["method"]),
            collector=raw["collector"],
            period_days=float(raw["period_days"]) if raw.get("period_days") is not None else None,
            via=raw.get("via"),
            note=raw.get("note", ""),
            collected_on=dt.date.fromisoformat(raw["collected_on"]) if raw.get("collected_on") else None,
        )
