"""
Message citations for /lore answers.

The RAG service starts every message line it returns with a tag naming the
exact message, ``[@<channel_id>/<message_id>]``. Those are long (about 40
characters of digits) and the model would copy them badly, so before a tool
result reaches the model each one is swapped for a short ref, ``[m12]``,
numbered per conversation. The model cites with those refs, and when the
answer is posted they become links that jump to the message in Discord.

A conversation keeps one registry for its lifetime, so the same message always
carries the same ref. That matters beyond tidiness: research blocks are
deduplicated by their exact text, and a chunk that came back with different
refs on a second search would no longer be recognised as a repeat.
"""

import re
from typing import Optional

_RAW_TAG = re.compile(r"\[@(\d+)/(\d+)\]")
# A ref or a run of them: [m3], [m3, m7], and the ranges the model sometimes
# writes, [m3–m4], which link their two ends. After the first ref the "m" may
# be dropped ([m123-127], [m3, 7]). Adjacent brackets, [m3][m7], [m3] [m7],
# [m3], [m7] or [m3]–[m7], are one match so they render as a single run.
# A dash between two refs (inside or between brackets) marks a range and is
# kept, "(1 – 2)"; any other gap becomes a comma, "(1, 2)".
# The leading space is captured so a dropped ref takes it along, and a
# rendered run always gets exactly one space before its parentheses.
_SEP = r"\s*[,\-–—]\s*"
_GROUP = rf"\[m\d+(?:{_SEP}m?\d+)*\]"
_CITE = re.compile(rf"( ?)({_GROUP}(?:[\s,\-–—]*{_GROUP})*)")
_REF = re.compile(r"m?(\d+)")
# Bracketed labels the model writes in place of a ref, though the answer rules
# forbid them: a tool name for a count ("[count_messages]"), a source that has
# no ref ("[Background]", "[Simulated]"), or a count itself
# ("[alice_e: 8617 messages]"). None can link, so they are dropped rather
# than shown as stray brackets. Deliberately narrow: [sic], [...] and the like
# are left alone.
_NOT_A_REF = re.compile(
    r" ?\[(?:count\w*|search_\w+|summarize_channel|read_surrounding_messages"
    r"|simulate_conversation|background|simulat\w*|[^\[\]\n:]{1,40}:\s*[\d,]+\s+messages?)\]",
    re.I,
)
_DASH = re.compile(r"[\-–—]")


class Citations:
    """Short-ref registry for one /lore conversation."""

    def __init__(self, refs: Optional[dict[str, list[str]]] = None) -> None:
        # ref -> [channel_id, message_id]
        self.refs: dict[str, list[str]] = dict(refs or {})
        self._by_message = {tuple(v): k for k, v in self.refs.items()}

    def __len__(self) -> int:
        return len(self.refs)

    def _ref_for(self, channel_id: str, message_id: str) -> str:
        key = (channel_id, message_id)
        ref = self._by_message.get(key)
        if ref is None:
            ref = f"m{len(self.refs) + 1}"
            self.refs[ref] = [channel_id, message_id]
            self._by_message[key] = ref
        return ref

    def tag(self, text: str) -> str:
        """Swap every raw RAG tag in a tool result for a short ref."""
        return _RAW_TAG.sub(lambda m: f"[{self._ref_for(m.group(1), m.group(2))}]", text)

    def url(self, ref: str, guild_id: str) -> Optional[str]:
        target = self.refs.get(ref)
        if not target or not guild_id:
            return None
        return f"https://discord.com/channels/{guild_id}/{target[0]}/{target[1]}"

    def link(self, answer: str, guild_id: str, *, suppress_previews: bool = True) -> str:
        """
        Turn the refs the model cited into numbered links, for display.

        Numbers run 1, 2, 3 in order of first citation in this answer, and a
        message cited twice keeps its number. A ref the registry does not know
        (the model invented it) is dropped, as are all refs when no guild is
        configured to build links with.

        Args:
            suppress_previews: Wrap URLs in <> so Discord does not unfurl them.
                That syntax is for message content; embeds never unfurl links,
                so pass False for an embed description.
        """
        numbers: dict[str, int] = {}

        def render(match: re.Match) -> str:
            run, seen = match.group(2), set()
            parts: list[str] = []
            last_end, dashed = 0, False
            for m in _REF.finditer(run):
                # The gap since the last ref kept; a dash anywhere in it, even
                # around a dropped ref, makes this a range.
                dashed = dashed or bool(_DASH.search(run, last_end, m.start()))
                last_end = m.end()
                ref = "m" + m.group(1)
                url = self.url(ref, guild_id)
                if url is None or ref in seen:
                    continue
                seen.add(ref)
                n = numbers.setdefault(ref, len(numbers) + 1)
                target = f"<{url}>" if suppress_previews else url
                if parts:
                    parts.append(" – " if dashed else ", ")
                parts.append(f"[{n}]({target})")
                dashed = False
            return f" ({''.join(parts)})" if parts else ""

        return _CITE.sub(render, _NOT_A_REF.sub("", answer))

    def to_dict(self) -> dict[str, list[str]]:
        return dict(self.refs)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "Citations":
        return cls({str(k): [str(v[0]), str(v[1])] for k, v in (d or {}).items()})
