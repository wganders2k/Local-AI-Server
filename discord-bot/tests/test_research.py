"""Rendering tool results as research blocks, and the cross-turn dedup."""

import json

from lore.research import (
    CONDENSED_KIND,
    SEARCH_NOTE_PREFIX,
    drop_seen_chunks,
    _flatten_research_for_synthesis,
    render_research_blocks,
)


def call(tool_call_id, name="search_discord_history", **args):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": tool_call_id,
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    }


def result(tool_call_id, content):
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def test_blocks_are_labelled_with_what_was_searched():
    msgs = [call("a", query="pigs", channel_name="#general"), result("a", "chunk")]
    blocks, dropped = render_research_blocks(msgs)
    assert blocks == ['[Search 1] "pigs" in #general\nchunk']
    assert dropped == 0


def test_unlabelled_search_still_renders():
    msgs = [call("a"), result("a", "chunk")]
    blocks, _ = render_research_blocks(msgs)
    assert blocks[0].startswith("[Search 1] (unlabelled search)")


def test_numbering_continues_across_turns():
    msgs = [call("a", query="q"), result("a", "chunk")]
    blocks, _ = render_research_blocks(msgs, start_index=3)
    assert blocks[0].startswith("[Search 4]")


def test_chunks_already_seen_are_dropped():
    seen = set()
    first = [call("a", query="q"), result("a", "shared\n\n---\n\nunique-1")]
    second = [call("b", query="q2"), result("b", "shared\n\n---\n\nunique-2")]

    render_research_blocks(first, seen_keys=seen)
    blocks, dropped = render_research_blocks(second, seen_keys=seen, start_index=1)

    assert dropped == 1
    assert "shared" not in blocks[0]
    assert "unique-2" in blocks[0]


def test_a_search_whose_every_hit_repeats_adds_no_block():
    seen = set()
    msgs = [call("a", query="q"), result("a", "same")]
    render_research_blocks(msgs, seen_keys=seen)
    blocks, dropped = render_research_blocks(
        [call("b", query="q"), result("b", "same")], seen_keys=seen
    )
    assert blocks == []
    assert dropped == 1


def test_empty_tool_results_are_skipped():
    msgs = [call("a", query="q"), result("a", "   ")]
    blocks, _ = render_research_blocks(msgs)
    assert blocks == []


def test_malformed_tool_arguments_do_not_raise():
    msgs = [
        {"role": "assistant", "tool_calls": [
            {"id": "a", "function": {"name": "x", "arguments": "{not json"}}]},
        result("a", "chunk"),
    ]
    blocks, _ = render_research_blocks(msgs)
    assert blocks[0].startswith("[Search 1] (unlabelled search)")


def test_flattening_produces_a_two_message_exchange():
    msgs = [call("a", query="pigs"), result("a", "chunk")]
    flat = _flatten_research_for_synthesis(msgs, "what about pigs?")
    assert [m["role"] for m in flat] == ["system", "user"]
    assert "what about pigs?" in flat[1]["content"]
    assert "chunk" in flat[1]["content"]
    # No tool-calling pattern left in context for the model to imitate.
    assert not any("tool_calls" in m for m in flat)


def test_flattening_with_no_results_says_so_plainly():
    flat = _flatten_research_for_synthesis([], "q")
    assert "could not find" in flat[1]["content"]


def test_condensed_research_passes_through_in_order():
    condensed = {"role": "user", "content": "[Condensed research 1] gist",
                 "kind": CONDENSED_KIND}
    msgs = [condensed, call("b", query="cows"), result("b", "moo")]
    blocks, _ = render_research_blocks(msgs)
    assert blocks[0] == "[Condensed research 1] gist"
    assert blocks[1].startswith('[Search 2] "cows"')


def test_an_untagged_user_message_is_not_research():
    # The live conversation holds the question and any condensed summary as
    # plain user messages; only the tagged copy counts.
    msgs = [{"role": "user", "content": "the question"}]
    assert render_research_blocks(msgs) == ([], 0)


SEP = "\n\n---\n\n"


def test_a_result_with_nothing_seen_is_returned_unchanged():
    seen: set[str] = set()
    assert drop_seen_chunks("a" + SEP + "b", seen) == "a" + SEP + "b"
    assert len(seen) == 2


def test_seen_chunks_are_dropped_with_a_note():
    seen: set[str] = set()
    drop_seen_chunks("a", seen)
    out = drop_seen_chunks("a" + SEP + "b", seen)
    assert out.split(SEP)[0] == "b"
    assert "1 of 2 excerpt(s) omitted" in out


def test_a_result_with_nothing_new_becomes_only_a_note():
    seen: set[str] = set()
    drop_seen_chunks("a", seen)
    assert drop_seen_chunks("a", seen).startswith(SEARCH_NOTE_PREFIX)


def test_notes_are_not_rendered_as_research():
    note = f"{SEARCH_NOTE_PREFIX} 1 of 2 excerpt(s) omitted"
    msgs = [call("a", query="pigs"), result("a", "oink" + SEP + note),
            call("b", query="pigs"), result("b", f"{SEARCH_NOTE_PREFIX} refused")]
    blocks, _ = render_research_blocks(msgs)
    assert len(blocks) == 1
    assert SEARCH_NOTE_PREFIX not in blocks[0]
