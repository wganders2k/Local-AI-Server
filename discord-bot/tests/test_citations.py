"""Citation refs: raw RAG tags -> [mN] for the model -> links for display."""

import asyncio

from lore.citations import Citations
from lore.session import LoreSession
from lore.tools import ToolContext, execute_tool

GUILD = "111"
URL = "https://discord.com/channels/111/{}/{}"


def test_tag_numbers_messages_stably():
    c = Citations()
    text = "[@5/10] a: hi\n[@5/11] b: yo\n[@5/10] a: again"
    assert c.tag(text) == "[m1] a: hi\n[m2] b: yo\n[m1] a: again"
    # A later result reuses the same refs, so deduplicated chunks stay identical.
    assert c.tag("[@5/11] b: yo") == "[m2] b: yo"
    assert len(c) == 2


def test_link_numbers_by_first_citation_and_suppresses_previews():
    c = Citations()
    c.tag("[@5/10] [@5/11] [@6/12]")
    out = c.link("Second [m2]. First [m1, m3]. Again [m2][m1].", GUILD)
    assert out == (
        f"Second [¹](<{URL.format(5, 11)}>). "
        f"First [²](<{URL.format(5, 10)}>)[³](<{URL.format(6, 12)}>). "
        f"Again [¹](<{URL.format(5, 11)}>)[²](<{URL.format(5, 10)}>)."
    )


def test_link_for_embeds_leaves_urls_bare():
    c = Citations()
    c.tag("[@5/10]")
    assert c.link("x [m1]", GUILD, suppress_previews=False) == f"x [¹]({URL.format(5, 10)})"


def test_unknown_refs_and_missing_guild_are_dropped_cleanly():
    c = Citations()
    c.tag("[@5/10]")
    assert c.link("Made up [m9]. Real [m1].", GUILD) == f"Made up. Real [¹](<{URL.format(5, 10)}>)."
    assert c.link("Real [m1].", "") == "Real."


def test_session_round_trips_citations():
    s = LoreSession(thread_id=1, system_prompt="p", original_question="q", pinned_now="now")
    s.citations.tag("[@5/10] [@6/12]")
    back = LoreSession.from_dict(s.to_dict())
    assert back.citations.to_dict() == {"m1": ["5", "10"], "m2": ["6", "12"]}
    assert back.citations.tag("[@7/1]") == "[m3]"  # numbering continues


def test_old_session_without_citations_loads():
    d = LoreSession(thread_id=1, system_prompt="p", original_question="q", pinned_now="n").to_dict()
    del d["citations"]
    assert len(LoreSession.from_dict(d).citations) == 0


class _FakeRAG:
    async def search_images(self, query, **kw):
        return "[@5/10] a: [image: a cat]", 1


def test_execute_tool_tags_results():
    c = Citations()
    out = asyncio.run(execute_tool("search_images", {"query": "cat"}, ToolContext(rag=_FakeRAG(), citations=c)))
    assert out == "[m1] a: [image: a cat]"


class _SurroundRAG:
    def __init__(self):
        self.calls = []

    async def surrounding(self, channel_id, message_id, direction="after", count=20):
        self.calls.append((channel_id, message_id, direction, count))
        return ">>> ANCHOR — [@5/10] a: hi\n[@5/13] b: the reply", 1


def test_read_surrounding_resolves_ref_and_tags_new_messages():
    c = Citations()
    c.tag("[@5/10] a: hi")
    rag = _SurroundRAG()
    out = asyncio.run(execute_tool(
        "read_surrounding_messages", {"ref": "[m1]", "direction": "both", "count": 5},
        ToolContext(rag=rag, citations=c),
    ))
    assert rag.calls == [("5", "10", "both", 5)]
    # The anchor keeps its ref; the new message gets the next one.
    assert out == ">>> ANCHOR — [m1] a: hi\n[m2] b: the reply"


def test_read_surrounding_rejects_unknown_ref_without_calling_rag():
    rag = _SurroundRAG()
    out = asyncio.run(execute_tool(
        "read_surrounding_messages", {"ref": "m7"}, ToolContext(rag=rag, citations=Citations()),
    ))
    assert out.startswith("Error: 'm7' is not a ref") and rag.calls == []


def test_ranges_link_both_ends():
    c = Citations()
    c.tag("[@5/10] [@5/11]")
    out = c.link("Every girlfriend [m1–m2]. Also [m1-m2].", GUILD)
    assert out == (
        f"Every girlfriend [¹](<{URL.format(5, 10)}>)[²](<{URL.format(5, 11)}>). "
        f"Also [¹](<{URL.format(5, 10)}>)[²](<{URL.format(5, 11)}>)."
    )
