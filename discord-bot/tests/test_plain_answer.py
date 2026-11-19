"""/lore answers as plain text: only the first part is a reply, nothing pings."""

import asyncio

import discord

from discord_io import post_plain_answer


class Recorder:
    def __init__(self, refuse=False):
        self.sent: list[tuple[str, str, dict]] = []
        self.refuse = refuse

    async def send(self, content, **kw):
        if self.refuse:
            raise discord.HTTPException(type("R", (), {"status": 403, "reason": "no"})(), "no")
        self.sent.append(("send", content, kw))

    async def reply(self, content, **kw):
        self.sent.append(("reply", content, kw))


def test_only_the_first_part_replies_and_nothing_pings():
    channel, status = Recorder(), Recorder()
    text = "\n\n".join(["para " * 300] * 3)  # well over one message

    asyncio.run(post_plain_answer(channel, text, reply_to=status))

    assert [k for k, _, _ in status.sent] == ["reply"]
    assert status.sent[0][2]["mention_author"] is False
    assert channel.sent and all(k == "send" for k, _, _ in channel.sent)
    for _, content, kw in status.sent + channel.sent:
        assert len(content) <= 2000
        assert kw["allowed_mentions"].everyone is False


def test_a_refused_chunk_goes_to_the_fallback():
    channel = Recorder(refuse=True)
    fallen: list[str] = []

    async def fallback(chunk):
        fallen.append(chunk)

    asyncio.run(post_plain_answer(channel, "short answer", fallback=fallback))

    assert fallen == ["short answer"]
