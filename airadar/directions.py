"""From fast-growing apps to directions, and from directions to opportunities.

A single app growing fast is news. Several apps in one category growing fast is
a direction, which is the thing worth acting on. Directions are found two ways,
because each catches what the other misses:

**Tags.** Collectors label each app with a few category tags. Cheap and
readable, but a tag only exists once someone thought to write it -- so tags
never surface a category nobody has named yet.

**Lexical clusters.** sciscout's TF-IDF agglomerative clustering, run over app
names and descriptions. Clusters whose distinctive terms match no existing tag
are reported as *emergent groupings*: candidates for a new direction that the
tag vocabulary does not yet cover. That is the part of the radar that can be
surprised.

Opportunity detection is a set of explicit heuristics over the direction table,
and is graded ``assumed`` throughout. The heuristics generate hypotheses worth a
person's (or the daily agent's) research time. They do not establish that a
market gap exists.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from sciscout.discovery import discover
from sciscout.models import Work

from .growth import AppScore
from .models import App, AppKind

OPEN_KINDS = {AppKind.REPO, AppKind.SPACE, AppKind.MODEL, AppKind.PACKAGE}
COMMERCIAL_KINDS = {AppKind.PRODUCT, AppKind.CONSUMER_APP}

# Collapse spelling variants so one direction is not split across three tags.
TAG_ALIASES = {
    "ai-agent": "agent", "agents": "agent", "ai-agents": "agent",
    "coding-agents": "coding-agent", "ai-coding": "coding-agent", "code-agent": "coding-agent",
    "llm-agent": "agent", "voice-ai": "voice", "speech": "voice", "tts": "voice",
    "text-to-video": "video-generation", "video-gen": "video-generation",
    "text-to-image": "image-generation", "image-gen": "image-generation",
    "skills": "agent-skills", "claude-skills": "agent-skills",
    "model-context-protocol": "mcp", "search": "ai-search",
}


# Words common in app descriptions that name no direction. Without these, a
# grouping gets labelled "your / anything / free" and the watch list fills with
# terms that match everything.
GENERIC_TERMS = frozenset(
    """
    your anything everything free tool tools system systems web app apps platform
    open source code per language model models build built make makes across
    first fast faster simple easy local official library framework kit based
    support supports powered help helps run runs like real time way into
    """.split()
)


def normalise_tag(tag: str) -> str:
    tag = tag.strip().lower().replace(" ", "-").replace("_", "-")
    return TAG_ALIASES.get(tag, tag)


@dataclass
class Direction:
    key: str
    members: list[str]
    scored: list[tuple[str, float]]
    top_share: float
    mean_score: float
    open_count: int
    commercial_count: int
    status: str = "new"
    change: float | None = None

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "members": self.members,
            "scored": len(self.scored),
            "top_share": round(self.top_share, 4),
            "mean_score": round(self.mean_score, 4),
            "open_count": self.open_count,
            "commercial_count": self.commercial_count,
        }


@dataclass
class EmergentGroup:
    label: str
    terms: list[str]
    members: list[str]
    cohesion: float


@dataclass
class Opportunity:
    kind: str
    direction: str
    rationale: str
    evidence: list[str] = field(default_factory=list)
    # What research found when it last tested this hypothesis, if it has. A
    # rule keeps firing on the direction table whatever the research said; the
    # verdict travels with it so the report does not re-propose a closed idea
    # as if it were new.
    verdict: dict | None = None


def tag_directions(
    apps: dict[str, App],
    scores: list[AppScore],
    previous: dict | None = None,
    min_members: int = 2,
) -> list[Direction]:
    """Aggregate app growth by tag, and compare with the previous snapshot.

    ``top_share`` -- the fraction of a direction's scored members in the top
    quartile of the whole ranking -- is the headline, rather than the mean
    score, because it is robust to one direction having many slow hangers-on.
    """
    score_of = {s.app_id: s.score for s in scores}
    ranked = [s.app_id for s in scores]
    top = set(ranked[: max(1, len(ranked) // 4)])
    members: dict[str, list[str]] = defaultdict(list)
    for app in apps.values():
        for tag in {normalise_tag(t) for t in app.tags}:
            members[tag].append(app.id)

    previous_by_key = {d["key"]: d for d in (previous or {}).get("directions", [])}
    out = []
    for key, ids in members.items():
        scored = [(i, score_of[i]) for i in ids if i in score_of]
        if len(scored) < min_members:
            continue
        direction = Direction(
            key=key,
            members=sorted(ids),
            scored=sorted(scored, key=lambda x: -x[1]),
            top_share=sum(1 for i, _ in scored if i in top) / len(scored),
            mean_score=sum(s for _, s in scored) / len(scored),
            open_count=sum(1 for i in ids if apps[i].kind in OPEN_KINDS),
            commercial_count=sum(1 for i in ids if apps[i].kind in COMMERCIAL_KINDS),
        )
        before = previous_by_key.get(key)
        if before is not None:
            direction.change = direction.mean_score - before["mean_score"]
            if direction.change >= 0.05:
                direction.status = "accelerating"
            elif direction.change <= -0.05:
                direction.status = "cooling"
            else:
                direction.status = "steady"
        out.append(direction)
    out.sort(key=lambda d: (-d.top_share, -d.mean_score, d.key))
    return out


def emergent_groups(
    apps: dict[str, App], scores: list[AppScore], threshold: float = 0.2
) -> list[EmergentGroup]:
    """Lexical clusters of scored apps that the tag vocabulary does not cover."""
    scored = {s.app_id for s in scores}
    works = [
        Work(id=a.id, title=a.name, abstract=a.text(), published=a.first_seen or dt.date.today())
        for a in apps.values() if a.id in scored
    ]
    if len(works) < 6:
        return []
    known = {normalise_tag(t) for a in apps.values() for t in a.tags}
    known_words = known | {part for tag in known for part in tag.split("-")}
    owners = {a.name.split("/")[0].lower() for a in apps.values() if "/" in a.name}
    result = discover(works, threshold=threshold, min_cluster_size=3, min_df=2, max_df_ratio=0.3)
    groups = []
    for track in result.tracks:
        novel = [
            t for t in track.terms
            if t not in known_words and t not in GENERIC_TERMS and t not in owners and len(t) >= 4
        ]
        if len(novel) < 2:
            continue
        groups.append(EmergentGroup(
            label=" / ".join(novel[:3]),
            terms=novel[:8],
            members=[w.id for w in track.works],
            cohesion=track.cohesion,
        ))
    return groups


def opportunities(directions: list[Direction], apps: dict[str, App]) -> list[Opportunity]:
    """Heuristic innovation hypotheses from the direction table.

    Three patterns, each a hypothesis about where building or investing could
    pay off, not a finding:

    - *fast and uncrowded*: a direction in the top third by momentum with few
      tracked apps. Either the space is genuinely open, or the radar is not
      seeing its incumbents -- both worth knowing.
    - *open source ahead of product*: open-source projects in a direction are
      growing fast and no commercial product is tracked in it. Historically the
      gap a hosted, supported version fills.
    - *product without an open alternative*: commercial apps growing with no open
      project tracked alongside -- the gap an open-source challenger fills.
    """
    if not directions:
        return []
    by_momentum = sorted(directions, key=lambda d: -d.mean_score)
    fast = {d.key for d in by_momentum[: max(1, len(by_momentum) // 3)]}
    out = []
    for d in directions:
        lead = [apps[i].name for i, _ in d.scored[:3]]
        if d.key in fast and len(d.members) <= 3:
            out.append(Opportunity(
                "fast and uncrowded", d.key,
                f"momentum in the top third of directions with only {len(d.members)} tracked apps",
                lead,
            ))
        if d.top_share >= 0.5 and d.open_count >= 2 and d.commercial_count == 0:
            out.append(Opportunity(
                "open source ahead of product", d.key,
                f"{d.open_count} open projects, {d.top_share:.0%} of them in the top quartile, "
                f"and no commercial product tracked",
                lead,
            ))
        if d.top_share >= 0.5 and d.commercial_count >= 2 and d.open_count == 0:
            out.append(Opportunity(
                "product without an open alternative", d.key,
                f"{d.commercial_count} commercial apps growing fast and no open project tracked",
                lead,
            ))
    return out


def rising_pairs(apps: dict[str, App], scores: list[AppScore], min_count: int = 2) -> list[tuple[str, str, int]]:
    """Tag pairs that co-occur on several top-quartile apps: convergence points
    where two directions are being combined into one product."""
    top = [s.app_id for s in scores[: max(1, len(scores) // 4)]]
    counts: Counter[tuple[str, str]] = Counter()
    for app_id in top:
        tags = sorted({normalise_tag(t) for t in apps[app_id].tags})
        for i in range(len(tags)):
            for j in range(i + 1, len(tags)):
                counts[(tags[i], tags[j])] += 1
    return [(a, b, n) for (a, b), n in counts.most_common() if n >= min_count]
