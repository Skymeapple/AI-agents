"""Live-source adapters, exercised against canned responses.

Nothing here touches the network. The adapters are where malformed upstream
data enters, so most of these tests are about records that are incomplete or
wrong: a bad record must cost one work, never the harvest.
"""

import datetime as dt
import xml.etree.ElementTree as ET

import requests

from sciscout.models import Maturity
from sciscout.sources import arxiv
from sciscout.sources.arxiv import ArxivSource
from sciscout.sources.openalex import OpenAlexSource


class FakeResponse:
    def __init__(self, payload=None, text="", error=None):
        self._payload, self.text, self._error = payload, text, error

    def raise_for_status(self):
        if self._error:
            raise self._error

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


class FakeSession:
    """Serves queued responses in order and records each request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params})
        return self.responses.pop(0)


# -- OpenAlex ---------------------------------------------------------------

FULL_RECORD = {
    "id": "https://openalex.org/W1",
    "display_name": "A title",
    "publication_date": "2024-03-01",
    "authorships": [
        {
            "author": {"display_name": "Ada"},
            "institutions": [
                {"display_name": "Uni", "type": "education"},
                {"display_name": "Co", "type": "Company"},
            ],
        },
        {"author": None, "institutions": None},
    ],
    "concepts": [
        {"display_name": "Physics", "score": 0.5, "level": 0},
        {"display_name": "Weak", "score": 0.1, "level": 0},
        {"display_name": "Too specific", "score": 0.9, "level": 3},
    ],
    "primary_location": {"source": {"display_name": "Nature"}},
    "type": "article",
    "cited_by_count": 7,
    "doi": "https://doi.org/10.1/x",
    "abstract_inverted_index": {"hello": [0], "world": [1]},
    "open_access": {"is_oa": True},
}


def test_openalex_parses_a_full_record():
    work = OpenAlexSource()._parse(FULL_RECORD)
    assert work.id == "W1"
    assert work.published == dt.date(2024, 3, 1)
    assert work.abstract == "hello world"
    assert work.venue == "Nature"
    assert work.maturity is Maturity.PEER_REVIEWED
    assert work.citations == 7
    assert work.url == "https://doi.org/10.1/x"
    assert work.extra == {"type": "article", "open_access": True}
    # Low-confidence and deep-level concepts are noise, not disciplines.
    assert work.disciplines == ["physics"]
    assert [(a.name, a.affiliation, a.is_industry) for a in work.authors] == [
        ("Ada", "Uni", True),
        ("unknown", None, False),
    ]


def test_openalex_tolerates_an_empty_record():
    work = OpenAlexSource()._parse({})
    assert work.id == ""
    assert work.title == "(untitled)"
    assert work.published is None
    assert work.authors == [] and work.disciplines == []
    assert work.citations is None
    assert work.maturity is Maturity.PEER_REVIEWED


def test_openalex_maturity_rules():
    parse = OpenAlexSource()._parse
    arxiv_venue = {"primary_location": {"source": {"display_name": "arXiv"}}}
    assert parse({"type": "patent", **arxiv_venue}).maturity is Maturity.PATENT
    assert parse({"type": "article", **arxiv_venue}).maturity is Maturity.PREPRINT
    assert parse({"type_crossref": "posted-content"}).maturity is Maturity.PREPRINT
    assert parse({"type": "REPORT"}).maturity is Maturity.REPORT
    assert parse({"type": "dataset"}).maturity is Maturity.REPORT


def test_openalex_harvest_follows_the_cursor_and_skips_bad_records():
    session = FakeSession(
        [
            FakeResponse(
                {
                    "results": [FULL_RECORD, {"publication_date": "not a date"}],
                    "meta": {"next_cursor": "abc"},
                }
            ),
            FakeResponse({"results": [{"id": "W2"}], "meta": {"next_cursor": None}}),
        ]
    )
    result = OpenAlexSource(mailto="x@example.org", session=session).harvest(
        "q", limit=50, from_date=dt.date(2024, 1, 1), extra_filters="type:article"
    )
    assert [w.id for w in result.works] == ["W1", "W2"]
    assert len(result.errors) == 1 and "skipped an OpenAlex record" in result.errors[0]
    first, second = (call["params"] for call in session.calls)
    assert first["cursor"] == "*" and second["cursor"] == "abc"
    assert first["filter"] == "from_publication_date:2024-01-01,type:article"
    assert first["mailto"] == "x@example.org"
    assert first["per-page"] == 50 and second["per-page"] == 49


def test_openalex_harvest_omits_unset_parameters_and_respects_limit():
    session = FakeSession(
        [FakeResponse({"results": [{"id": f"W{i}"} for i in range(5)], "meta": {"next_cursor": "n"}})]
    )
    source = OpenAlexSource(session=session)
    source.mailto = None
    result = source.harvest("q", limit=3)
    assert len(result.works) == 3
    assert len(session.calls) == 1
    assert "filter" not in session.calls[0]["params"]
    assert "mailto" not in session.calls[0]["params"]


def test_openalex_harvest_reports_request_and_json_failures():
    failed = FakeSession([FakeResponse(error=requests.ConnectionError("down"))])
    result = OpenAlexSource(session=failed).harvest("q")
    assert result.works == [] and "OpenAlex request failed" in result.errors[0]

    garbled = FakeSession([FakeResponse(payload=None)])
    result = OpenAlexSource(session=garbled).harvest("q")
    assert result.works == [] and "invalid JSON" in result.errors[0]


# -- arXiv ------------------------------------------------------------------

FEED = """<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
<entry>
  <id>http://arxiv.org/abs/2401.00001v1</id>
  <published>2024-01-02T00:00:00Z</published>
  <title> A
   title </title>
  <summary> some   text </summary>
  <author><name> Ada </name><arxiv:affiliation>Uni</arxiv:affiliation></author>
  <author><name>Bob</name></author>
  <author/>
  <category term="cs.LG"/><category/>
  <arxiv:journal_ref> J. Demo 1 </arxiv:journal_ref>
</entry>
<entry><id>x/y</id><title>t</title><arxiv:journal_ref>  </arxiv:journal_ref></entry>
<entry><published>not a date</published></entry>
</feed>"""


def test_arxiv_parses_entries():
    source = ArxivSource()
    published, preprint, _ = ET.fromstring(FEED)

    work = source._parse_entry(published)
    assert work.id == "2401.00001v1"
    assert work.url == "http://arxiv.org/abs/2401.00001v1"
    assert work.title == "A title"
    assert work.abstract == "some text"
    assert work.published == dt.date(2024, 1, 2)
    assert [(a.name, a.affiliation) for a in work.authors] == [("Ada", "Uni"), ("Bob", None)]
    assert work.extra == {"categories": ["cs.LG"]}
    assert work.citations is None
    # A journal reference promotes a preprint to published work.
    assert work.venue == "J. Demo 1"
    assert work.maturity is Maturity.PEER_REVIEWED

    # A blank journal reference is no reference at all.
    work = source._parse_entry(preprint)
    assert work.id == "y"
    assert work.published is None
    assert work.venue == "arXiv"
    assert work.maturity is Maturity.PREPRINT


def test_arxiv_harvest_skips_bad_entries_and_stops_on_bad_xml(monkeypatch):
    monkeypatch.setattr(arxiv, "MIN_INTERVAL_SECONDS", 0.0)
    session = FakeSession([FakeResponse(text=FEED), FakeResponse(text="<not xml")])
    result = ArxivSource(session=session).harvest("q", limit=10)
    assert [w.id for w in result.works] == ["2401.00001v1", "y"]
    assert any("skipped an arXiv entry" in e for e in result.errors)
    assert any("unparseable XML" in e for e in result.errors)


# -- local corpus -----------------------------------------------------------


def test_local_corpus_loads_every_record_unless_a_limit_is_given(tmp_path):
    """A corpus file must never be silently truncated: the tail is data too."""
    import json

    from sciscout.sources.local import LocalCorpus

    records = [
        {"id": f"w{i}", "title": f"work {i}", "published": "2024-01-01"}
        for i in range(10_050)
    ]
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(records))
    assert len(LocalCorpus(path).harvest().works) == 10_050
    assert len(LocalCorpus(path).harvest(limit=7).works) == 7
