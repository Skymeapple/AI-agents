"""I6 task 6.2: a dual-use screen for agent-proposed protocols, plus a red-team harness.

The controlled list here is PLACEHOLDER names only; a real deployment loads the
institution's controlled-agent and precursor lists (for example the relevant
national export-control schedules). The screen normalises text so trivial
obfuscation (case, spacing, punctuation, look-alike characters, aliases) does
not bypass it. A screen like this is one layer; it does not replace human
biosafety review.
"""
from __future__ import annotations

import random
import re
import unicodedata

CONTROLLED = {
    "controlledagentalpha": ["cagent-a", "alpha agent"],
    "restrictedprecursorbeta": ["rp-beta", "precursor b"],
    "regulatedtoxingamma": ["toxin g", "rtg"],
}
HOMOGLYPHS = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s",
                            "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
                            "х": "x", "і": "i", "α": "a", "ο": "o"})


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold().translate(HOMOGLYPHS)
    return re.sub(r"[^a-z]", "", s)


_TERMS = {norm(k): k for k in CONTROLLED} | {norm(a): k for k, al in CONTROLLED.items() for a in al}


def _strings(o):
    """All string values in a protocol. str(dict) would escape invisible characters
    (zero-width spaces become literal 'u200b'), which the first red-team round exploited."""
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for v in o.values():
            yield from _strings(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _strings(v)


def screen(protocol: dict) -> list[str]:
    strings = list(_strings(protocol))
    text = "|".join(norm(s) for s in strings)
    # short aliases (under 5 letters) match whole fields only, so 'rtg' does not fire on 'start gel'
    fields = {norm(s) for s in strings}
    hits = {k for t, k in _TERMS.items() if (t in text if len(t) >= 5 else t in fields)}
    return [f"dual-use screen: matches controlled item '{k}'" for k in sorted(hits)]


def red_team(n=50, seed=0) -> dict:
    rng = random.Random(seed)
    tricks = [
        lambda s: s.upper(),
        lambda s: " ".join(s),
        lambda s: "-".join(s[i:i + 3] for i in range(0, len(s), 3)),
        lambda s: s.replace("a", "а").replace("o", "о"),  # Cyrillic look-alikes
        lambda s: s.replace("e", "3").replace("a", "4"),
        lambda s: s.replace("o", "0").replace("l", "1"),
        lambda s: "​".join(s),  # zero-width spaces
        lambda s: s[:len(s) // 2] + "_" + s[len(s) // 2:],
        lambda s: unicodedata.normalize("NFD", s),
        lambda s: "".join(c.upper() if rng.random() < .5 else c for c in s),
    ]
    names = list(CONTROLLED) + [a for al in CONTROLLED.values() for a in al]
    bypass = []
    for i in range(n):
        name = rng.choice(names)
        t1, t2 = rng.sample(tricks, 2)
        disguised = t2(t1(name))
        p = {"protocol_id": f"rt{i}", "steps": [{"op": "dispense", "reagent": disguised, "fraction": 1.0}]}
        if not screen(p):
            bypass.append(disguised)
    benign = [{"steps": [{"op": "dispense", "reagent": r, "fraction": 1}]} for r in
              ("A", "B", "C", "sodium chloride", "glycerol", "agar", "ethanol", "tris buffer", "start gel", "alpha-amylase", "beta-lactamase")]
    false_blocks = sum(bool(screen(p)) for p in benign)
    return {"attempts": n, "bypasses": len(bypass), "examples": bypass[:5],
            "benign_tested": len(benign), "benign_false_blocks": false_blocks}


if __name__ == "__main__":
    import json
    print(json.dumps(red_team(), indent=2, ensure_ascii=False))
