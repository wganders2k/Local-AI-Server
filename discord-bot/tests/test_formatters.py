"""Text post-processing: splitting, disclaimer stripping, embeds."""

from discord_io import split_message
from formatters import build_lore_embeds, find_split_boundary, strip_disclaimers


def test_short_text_is_not_split():
    assert find_split_boundary("hello", 100) == 5


def test_paragraph_breaks_are_preferred():
    buf = "a" * 10 + "\n\n" + "b" * 40
    assert find_split_boundary(buf, 30) == 12  # just past the "\n\n"


def test_line_breaks_are_the_fallback():
    buf = "a" * 10 + "\n" + "b" * 40
    assert find_split_boundary(buf, 30) == 11


def test_hard_split_when_there_is_no_break():
    assert find_split_boundary("a" * 100, 30) == 30


def test_split_message_covers_the_whole_text():
    text = ("paragraph\n\n" * 400).strip()
    chunks = split_message(text, max_length=2000)
    assert all(len(c) <= 2000 for c in chunks)
    assert "".join(c.replace("\n", "") for c in chunks) == text.replace("\n", "")


def test_disclaimers_are_stripped_from_the_tail():
    assert strip_disclaimers("real answer\n\nAs an AI, I must note things.") == "real answer"


def test_body_text_that_merely_mentions_ai_survives():
    text = "As an AI joke, he said something"
    assert strip_disclaimers(text) == text


def test_long_lore_answers_paginate_into_several_embeds():
    embeds = build_lore_embeds("word " * 3000)
    assert len(embeds) > 1
    assert embeds[0].footer.text == "Part 1/%d" % len(embeds)
    assert all(len(e.description) <= 4096 for e in embeds)


def test_empty_lore_answer_still_produces_an_embed():
    assert build_lore_embeds("")[0].description == "(No results)"
