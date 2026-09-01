"""The knowledge base has one job beyond holding text: change without downtime."""
import time

import pytest

SAMPLE = """# KB

## Trend Following
**Tags:** trend, momentum

Buy strength, sell weakness.

## Position Sizing
**Tags:** risk, sizing

Risk a fixed fraction.

## House Rules
**Tags:** process

HOLD is a real answer.
"""


def test_parse_splits_sections_and_strips_the_tag_line():
    from llm_agent.knowledge_base import parse
    kb = parse(SAMPLE, label="t")
    assert [s.title for s in kb.sections] == ["Trend Following", "Position Sizing", "House Rules"]
    assert kb.sections[0].tags == ("trend", "momentum")
    assert "**Tags:**" not in kb.sections[0].body
    assert kb.ok and len(kb.checksum) == 64


def test_parse_rejects_text_with_no_sections():
    from llm_agent.knowledge_base import parse
    with pytest.raises(ValueError):
        parse("just some prose with no headings")
    with pytest.raises(ValueError):
        parse("")


def test_retrieval_ranks_by_tags_and_always_pins_the_rules():
    from llm_agent.knowledge_base import parse
    kb = parse(SAMPLE, label="t")
    picked = kb.retrieve(["momentum breakout"], max_sections=1, max_chars=10_000)
    titles = [s.title for s in picked]
    assert "Trend Following" in titles          # matched on tags
    assert "House Rules" in titles              # pinned, does not consume the budget


def test_hot_reload_swaps_the_snapshot_when_the_file_changes(env, tmp_path):
    from llm_agent.knowledge_base import KnowledgeBaseStore
    path = tmp_path / "kb.md"
    path.write_text(SAMPLE)
    store = KnowledgeBaseStore(path=str(path), refresh_seconds=0)
    first = store.get()
    assert len(first.sections) == 3

    time.sleep(0.01)
    path.write_text(SAMPLE + "\n## New Idea\n**Tags:** regime\n\nRegimes change.\n")
    second = store.get()
    assert len(second.sections) == 4
    assert second.checksum != first.checksum
    assert store.reload_count == 2
    # the old snapshot is untouched: readers mid-prompt keep a consistent view
    assert len(first.sections) == 3


def test_a_broken_edit_keeps_serving_the_last_good_version(env, tmp_path):
    from llm_agent.knowledge_base import KnowledgeBaseStore
    path = tmp_path / "kb.md"
    path.write_text(SAMPLE)
    store = KnowledgeBaseStore(path=str(path), refresh_seconds=0)
    good = store.get()

    time.sleep(0.01)
    path.write_text("oops, no headings at all")
    still = store.get()
    assert still.checksum == good.checksum
    assert "parse failed" in (store.last_error or "")


def test_every_good_version_is_written_to_the_database(env, tmp_path):
    from core.repository import kb_history
    from llm_agent.knowledge_base import KnowledgeBaseStore
    path = tmp_path / "kb.md"
    path.write_text(SAMPLE)
    KnowledgeBaseStore(path=str(path), refresh_seconds=0).get()
    history = kb_history()
    assert history and history[0]["section_count"] == 3 and history[0]["active"]


def test_a_host_with_no_file_loads_the_knowledge_base_from_the_database(env, tmp_path):
    """The migration case: new machine, same Postgres, no local copy."""
    from llm_agent.knowledge_base import KnowledgeBaseStore
    path = tmp_path / "kb.md"
    path.write_text(SAMPLE)
    KnowledgeBaseStore(path=str(path), refresh_seconds=0).get()      # seeds the database
    path.unlink()

    fresh = KnowledgeBaseStore(path=str(path), refresh_seconds=0)
    kb = fresh.get()
    assert kb.source == "database" and len(kb.sections) == 3
