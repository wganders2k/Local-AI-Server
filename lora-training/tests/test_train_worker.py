"""
render_as_served's split of an example into the served prompt and the trained
completion. A wrong split is silent: the run trains, on replies missing their
first characters or on a prompt the server never sends.

Run:  .venv-test/bin/python -m pytest tests -q
"""

import pytest

import train_worker as tw


class GemmaLikeTokenizer:
    """Renders like Gemma 4 with thinking off: the generation prompt carries an
    empty thought channel that a rendered assistant turn does not."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False):
        out = "".join(f"<|turn>{m['role']}\n{m['content']}<turn|>\n" for m in msgs)
        if add_generation_prompt:
            out += "<|turn>model\n<|channel>thought\n<channel|>"
        return out.replace("<|turn>assistant", "<|turn>model")


PROMPT = [{"role": "user", "content": "[bob]: nice one"}]


def test_completion_is_the_reply_and_end_of_turn():
    split = tw.split_as_served(GemmaLikeTokenizer(), PROMPT, [{"role": "assistant", "content": "cheers"}], {})
    assert split["prompt"].endswith("<|channel>thought\n<channel|>")
    assert split["completion"] == "cheers<turn|>\n"


def test_reply_sharing_a_prefix_with_the_thought_marker_keeps_every_character():
    split = tw.split_as_served(GemmaLikeTokenizer(), PROMPT, [{"role": "assistant", "content": "<3 love it"}], {})
    assert split["completion"] == "<3 love it<turn|>\n"


def test_unsplittable_template_fails_loudly():
    class Dropping(GemmaLikeTokenizer):
        def apply_chat_template(self, msgs, **kw):
            return super().apply_chat_template([m for m in msgs if m["role"] != "assistant"], **kw)

    with pytest.raises(ValueError):
        tw.split_as_served(Dropping(), PROMPT, [{"role": "assistant", "content": "hi"}], {})
