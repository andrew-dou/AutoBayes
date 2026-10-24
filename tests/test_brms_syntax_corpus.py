"""The brms syntax corpus (tests/brms_syntax/, docs/brms-parity/reports/syntax-audit.pdf): what AutoBayes does with each of
its brms formulas must stay what ``corpus.json`` records.

- An accepted formula (outcome ``match``) must keep building a model whose reading equals brms 2.23's ``make_standata()``
  (``audit.compare``: design columns, group-level terms and level labels, addition-term and special-term data, centring,
  family and links).
- A refusal (outcome ``refused_named``) must keep raising the same kind of error with a message naming its construct.

KNOWN GAPS. Every entry carries ``result.gap``, and the test ids below start with it:

- ``defect``: a silent misreading (``misread``), a refusal whose message does not name the construct or misnames it
  (``refused_unclear``) or a crash. ``test_known_gap_is_pinned`` pins today's behaviour, so that a fix shows up as a failure
  there: then re-run ``python -m tests.brms_syntax.audit`` (and review the entry) to record the new behaviour.
- ``capability``: a construct brms takes and AutoBayes refuses by name (not yet offered).
- ``divergence``: a difference a divergence row documents (``differs``).

brms's side is regenerated with ``Rscript tests/brms_syntax/brms_reading.R`` (R and brms 2.23 needed); nothing here needs R.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache

import pytest

from tests.brms_syntax import audit

CORPUS = audit.load_json(audit.CORPUS)
STORE = audit.load_json(audit.STANDATA)
ENTRIES = CORPUS["entries"]


@lru_cache(maxsize=1)
def frame():
    return audit.python_frame()


def cases(*outcomes):
    return [pytest.param(e, id=f"{e['result']['gap'] or 'ok'}:{e['id']}") for e in ENTRIES if e["result"]["outcome"] in outcomes]


@pytest.mark.parametrize("entry", cases("match"))
def test_accepted_formula_matches_brms(entry):
    result = audit.run_entry(entry, frame(), STORE)
    assert result["status"] == "built", f"{entry['id']} is now refused: {result.get('message')}"
    assert result["issues"] == [], f"{entry['id']} no longer reads as brms does: {result['issues']}"


@pytest.mark.parametrize("entry", cases("refused_named"))
def test_refusal_names_its_construct(entry):
    result = audit.run_entry(entry, frame(), STORE)
    recorded = entry["result"]
    assert result["status"] == "error", f"{entry['id']} is now accepted; review it and re-run the audit"
    assert (result["type"], result["mechanism"]) == (recorded["type"], recorded["mechanism"]), result["message"]
    assert any(t.lower() in result["message"].lower() for t in entry["tokens"]), \
        f"{entry['id']}: the message no longer names {entry['tokens']}: {result['message']}"


@pytest.mark.parametrize("entry", cases("misread", "differs", "refused_unclear", "crash"))
def test_known_gap_is_pinned(entry):
    """KNOWN GAPS: today's behaviour of the ``defect`` and ``divergence`` entries, pinned so that a change is noticed."""
    result = audit.run_entry(entry, frame(), STORE)
    recorded = entry["result"]
    if recorded["outcome"] in ("misread", "differs"):
        assert result["status"] == "built", result.get("message")
        assert result["issues"] == recorded["issues"]
    else:
        assert result["status"] == "error"
        assert (result["type"], result["mechanism"]) == (recorded["type"], recorded["mechanism"])
        assert result["message"].startswith(recorded.get("pin", recorded["message"][:80]))


def test_corpus_is_complete_and_reviewed():
    ids = [e["id"] for e in ENTRIES]
    assert len(ids) == len(set(ids))
    assert set(ids) == set(STORE["entries"]), "standata.json and corpus.json list different entries"
    for e in ENTRIES:
        assert e["brms_reading"]["status"] == STORE["entries"][e["id"]]["status"], e["id"]
        assert e["result"]["outcome"] in audit.OUTCOMES, f"{e['id']} is {e['result']['outcome']}: review it"
        assert e["result"]["gap"] == audit.gap(e, e["result"]["outcome"]), e["id"]
        if e["result"]["outcome"] == "refused_named":
            assert e.get("tokens"), e["id"]
        if e["result"]["outcome"] == "differs":
            assert e.get("documented_by"), e["id"]
    assert CORPUS["summary"]["outcomes"] == dict(Counter(e["result"]["outcome"] for e in ENTRIES))
    assert STORE["meta"]["brms"] == "2.23.0"


def test_data_csv_is_the_generators(tmp_path):
    """brms read ``data.csv``; AutoBayes reads the generator's frame: they must be the same numbers."""
    audit.write_csv(audit.make_data(), tmp_path / "data.csv")
    assert (tmp_path / "data.csv").read_bytes() == audit.DATA_CSV.read_bytes()
    assert CORPUS["data"] == audit.data_spec()
