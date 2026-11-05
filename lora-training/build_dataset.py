"""
Tier 3: turn the per-user archive into one member's mimic training set.

The archive (history-service, Tier 2) is one JSONL per *author*, unfiltered.
A mimic adapter needs something else: the member's replies, each paired with
the conversation they were replying to. That conversation was written by other
people and lives in other people's files, so the first thing this does is put
every file back together into per-channel timelines.

    archive/<user>.jsonl x N  ->  channel timelines  ->  turns  ->  examples

Each example is TRL's conversational prompt/completion shape, so loss falls on
the reply alone and never on the context the model is only meant to read:

    {"prompt":     [{"role": "system", ...}, {"role": "user", "content": "[a]: ...\\n[b]: ..."}],
     "completion": [{"role": "assistant", "content": "<the member's reply>"}]}

The user turn labels each line `[name]: text`, which is how /mimic labels the
messages it sends (discord-bot/cogs/mimic.py). The system prompt is rendered
from the same template the bot renders, so training and inference agree on it.

Captions never enter. history-service marks every one
`caption_excluded_from_training`: a synthetic description is not anyone's
voice. Attachments appear in context only as a bare `[attachment]` marker.

CPU only, no torch — it runs in the trainer image without a GPU lease, and its
tests run in the torch-free test venv.

    python build_dataset.py --user-id 123 --persona mimic_user1 --display-name Alice
"""

import argparse
import glob
import json
import logging
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("build-dataset")

HERE = os.path.dirname(os.path.abspath(__file__))

# -- text normalisation --

_URL_RE = re.compile(r"https?://\S+")

# Links are rewritten to what they point at, never kept. Discord's own URLs are
# nothing but snowflakes (cdn.discordapp.com/attachments/<channel>/<message>/…,
# discord.com/channels/<guild>/<channel>/<message>), and most others carry one
# too (twitter.com/x/status/<id>). With ~1.5k of them in one member's context,
# the adapter learned that a long run of digits is something people write, and
# started writing its own. The domain stays: "they posted a tweet" is voice.
_DISCORD_FILE_RE = re.compile(r"https?://(?:cdn\.discordapp\.com|media\.discordapp\.net)/\S+")
_DISCORD_LINK_RE = re.compile(r"https?://(?:\w+\.)?discord(?:app)?\.com/channels/\S+")
_URL_DOMAIN_RE = re.compile(r"https?://(?:www\.)?([^/\s?#:]+)\S*")
# What the three above leave behind, so the drop checks can still see a link.
_LINK_MARKER_RE = re.compile(r"\[(?:attachment|message link|link: [^\]\s]+)\]")
# A typed run this long is an id someone pasted or keyboard-mash, never a number
# anyone reads. 15 digits is below the shortest snowflake (17).
_LONG_NUMBER_RE = re.compile(r"\d{15,}")
LONG_NUMBER = "[long number]"

_USER_MENTION_RE = re.compile(r"<@!?(\d+)>")
_CHANNEL_MENTION_RE = re.compile(r"<#(\d+)>")
_ROLE_MENTION_RE = re.compile(r"<@&\d+>")
_CUSTOM_EMOJI_RE = re.compile(r"<a?:(\w+):\d+>")

# PII. Deliberately conservative patterns: a false positive costs one odd token
# in one example, a false negative puts someone's phone number in the weights.
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Separators required, so a bare run of digits (a Discord id, a score) survives.
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d{1,2}[\s.-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)")
_ADDRESS_RE = re.compile(
    r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s+){1,3}"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Ct|Court|Way|Pl|Place|Cres|Crescent)\b\.?"
)

# Bot invocations are talking to a bot, not to people. They are dropped from the
# timeline entirely — as targets they teach nothing, and as context they are noise.
# The prefixes are the ones this server's bots actually use (counted from the
# archive): mostly music bots. `-`/`--` are also ordinary punctuation, so those
# count only when followed by a music-bot verb.
_COMMAND_RE = re.compile(
    r"^(?:[/!]|;;|[mtp]!)\w"
    r"|^--?(?:p|play|skip|s|leave|np|queue|q|remove|rewind|loop|vol|volume|shuffle|fs|ps)\b",
    re.IGNORECASE,
)

# The archive keeps Discord's own events (pins, thread creation, poll results)
# as messages from whoever triggered them, tagged with their type. Nobody typed
# them: "Pinned a message." was the single most common target in the first
# mimic dataset (225 of 20,887 replies). Only conversation is kept. DCE writes a
# named type by name and an unnamed one as Discord's number; records archived
# before history-service recorded a type have none, and are conversation.
_CONVERSATION_TYPES = {None, "Default", "Reply", "0", "19"}

# history-service stores mentions already rendered as "@Display Name", and
# nicknames hold spaces, quotes and emoji, so no pattern can say where one ends.
# Instead a whole message that is only "@something" and recurs this often across
# the archive is taken to be a mention. A bare ping teaches the adapter to answer
# with a name — "@Deleted User" alone was a target 46 times.
MENTION_MIN_COUNT = 3


def mention_tokens(records: list[dict]) -> list[str]:
    """
    Single mentions as they appear in the archive. A recurring "@Name terraria?"
    is a ping with something to say, not a longer name, so a candidate that
    starts with a shorter one is rejected — as is anything with a second "@"
    (several pings, removed one name at a time) or a space after the "@".
    """
    counts = Counter((r.get("content") or "").strip() for r in records)
    candidates = sorted(
        (t for t, n in counts.items()
         if n >= MENTION_MIN_COUNT and t.startswith("@") and t.count("@") == 1
         and len(t) > 1 and not t[1].isspace() and "\n" not in t),
        key=len,
    )
    tokens: list[str] = []
    for t in candidates:
        if not any(t.startswith(shorter) for shorter in tokens):
            tokens.append(t)
    return tokens[::-1]  # longest first, so no name is cut short by a shorter one inside it


def without_mentions(text: str, tokens: list[str]) -> str:
    for token in tokens:
        text = text.replace(token, "")
    return text.strip()


def scrub_pii(text: str) -> str:
    text = _EMAIL_RE.sub("[email]", text)
    text = _PHONE_RE.sub("[phone]", text)
    return _ADDRESS_RE.sub("[address]", text)


def resolve_mentions(record: dict) -> str:
    """
    `content` with each "@Nickname" history-service could identify turned back
    into Discord's "<@id>", so normalise renders it as "@username" — the same
    name the speaker labels use, which lets the model connect a mention to a
    speaker. A nickname never could: "@Sir Lags-a-Lot (AFK)" and "[pcow]"
    are one person. Records archived before history-service kept `mentions`
    pass through unchanged.
    """
    text = record.get("content") or ""
    for m in sorted(record.get("mentions") or [], key=lambda m: len(m["display"]), reverse=True):
        text = text.replace("@" + m["display"], f"<@{m['user_id']}>")
    return text


def replace_links(text: str) -> str:
    """Every URL -> a marker naming what it was."""
    text = _DISCORD_FILE_RE.sub("[attachment]", text)
    text = _DISCORD_LINK_RE.sub("[message link]", text)
    return _URL_DOMAIN_RE.sub(lambda m: f"[link: {m.group(1).lower()}]", text)


def normalise(text: str, usernames: dict[str, str], channels: dict[str, str]) -> str:
    """Raw Discord markup -> what a person reading the channel would see. No id survives."""
    # Links first: an email pattern would otherwise bite into a URL's userinfo.
    text = replace_links(text)
    text = _USER_MENTION_RE.sub(lambda m: "@" + usernames.get(m.group(1), "someone"), text)
    text = _CHANNEL_MENTION_RE.sub(lambda m: "#" + channels.get(m.group(1), "channel"), text)
    text = _ROLE_MENTION_RE.sub("@role", text)
    text = _CUSTOM_EMOJI_RE.sub(r":\1:", text)
    # Last: mention and emoji markup carry snowflakes that the lines above resolve.
    text = _LONG_NUMBER_RE.sub(LONG_NUMBER, text)
    return scrub_pii(text).strip()


def without_urls(text: str) -> str:
    """Raw URLs and the markers replace_links leaves in their place, for the drop checks."""
    return _LINK_MARKER_RE.sub("", _URL_RE.sub("", text)).strip()


def parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


# -- loading --

def load_archive(archive_dir: str) -> list[dict]:
    """Every record from every per-user file. Order is restored later, per channel."""
    records = []
    for path in sorted(glob.glob(os.path.join(archive_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} records from {archive_dir}")
    return records


def load_aliases(path: str | None) -> dict[str, str]:
    """
    {user_id: alias} from the hand-filled names file. Keys starting with "_" are
    notes, and an empty alias means "keep their handle". The file lives beside
    the training data, not in the repo, because it holds members' real names.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        entries = json.load(fh)
    return {
        uid: entry["alias"].strip()
        for uid, entry in entries.items()
        if not uid.startswith("_") and (entry.get("alias") or "").strip()
    }


def render_system_prompt(template_path: str, persona: str, display_name: str) -> str:
    """
    The bot's own template, rendered the bot's way (str.format, trailing newline
    stripped — see discord-bot/prompt_loader.py). A drift here means the adapter
    trained on one system prompt and is served another.
    """
    with open(template_path, encoding="utf-8") as fh:
        template = fh.read().rstrip("\n")
    return template.format(persona=persona, display_name=display_name)


# -- building --

@dataclass
class Options:
    # Every account the member has posted from: someone whose account was
    # deleted and who rejoined has two ids, and both are their voice.
    user_ids: list[str]
    context_n: int = 12
    gap_minutes: float = 30.0
    merge_seconds: float = 120.0
    min_chars: int = 8
    exclude_channels: list[str] = field(default_factory=list)
    # user id -> the name to use for them in place of their Discord handle
    aliases: dict[str, str] = field(default_factory=dict)


@dataclass
class Turn:
    """Consecutive messages from one author, close enough together to be one utterance."""
    user_id: str
    username: str
    start: datetime
    end: datetime
    texts: list[str]
    attachment_only: bool  # every message in the turn was an attachment with no text

    @property
    def text(self) -> str:
        return "\n".join(t for t in self.texts if t)


def _channel_timelines(records: list[dict], opts: Options, stats: Counter) -> dict[str, list[dict]]:
    excluded = set(opts.exclude_channels)
    by_channel: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        if r.get("is_bot"):
            stats["dropped_bot_message"] += 1
            continue
        if r.get("channel_id") in excluded or r.get("channel_name") in excluded:
            stats["dropped_excluded_channel"] += 1
            continue
        content = (r.get("content") or "").strip()
        if _COMMAND_RE.match(content):
            stats["dropped_command"] += 1
            continue
        if r.get("type") not in _CONVERSATION_TYPES:
            stats["dropped_system_message"] += 1
            continue
        if not content and not r.get("attachments"):
            stats["dropped_empty"] += 1
            continue
        by_channel[r["channel_id"]].append(r)
    for msgs in by_channel.values():
        msgs.sort(key=lambda r: (r["timestamp"], r["message_id"]))
    return by_channel


def _turns(msgs: list[dict], usernames: dict, channels: dict, merge_seconds: float) -> list[Turn]:
    turns: list[Turn] = []
    for r in msgs:
        ts = parse_ts(r["timestamp"])
        text = normalise(resolve_mentions(r), usernames, channels)
        att_only = not text and bool(r.get("attachments"))
        prev = turns[-1] if turns else None
        if (
            prev is not None
            and prev.user_id == r["user_id"]
            and (ts - prev.end).total_seconds() <= merge_seconds
        ):
            prev.texts.append(text)
            prev.end = ts
            prev.attachment_only = prev.attachment_only and att_only
        else:
            turns.append(Turn(r["user_id"], usernames.get(r["user_id"]) or r["user_id"], ts, ts, [text], att_only))
    return turns


def _context_line(turn: Turn) -> str:
    # The only trace an attachment leaves. Never its caption — see module docstring.
    body = turn.text or "[attachment]"
    return f"[{turn.username}]: {body}"


def build_examples(records: list[dict], system_prompt: str | None, opts: Options) -> tuple[list[dict], Counter]:
    """
    (examples sorted oldest first, counts of what was kept and why things were dropped).

    A target turn becomes an example only if it has at least one turn of context
    before it inside the gap. A message that opens a conversation was not a
    reply to anything, and an empty prompt teaches the model to talk unprompted.
    """
    stats: Counter = Counter()
    usernames = {m["user_id"]: m["username"] for r in records for m in r.get("mentions") or [] if m.get("username")}
    usernames.update({r["user_id"]: r.get("username") or r["user_id"] for r in records})
    # One name per person everywhere: speaker labels and resolved mentions both
    # come from this map, so "[will]:" and "@will" are recognisably the same.
    usernames.update(opts.aliases)
    channels = {r["channel_id"]: r.get("channel_name") or r["channel_id"] for r in records}
    gap = opts.gap_minutes * 60.0
    # Resolved mentions render as "@username"; the mined tokens cover records
    # archived before history-service kept `mentions`.
    mentions = sorted(
        {"@" + u for u in usernames.values()} | {"@here", "@everyone", "@role"} | set(mention_tokens(records)),
        key=len, reverse=True,
    )

    examples = []
    for msgs in _channel_timelines(records, opts, stats).values():
        turns = _turns(msgs, usernames, channels, opts.merge_seconds)
        for i, turn in enumerate(turns):
            if turn.user_id not in opts.user_ids:
                continue
            stats["target_turns"] += 1
            if turn.attachment_only:
                stats["dropped_attachment_only"] += 1
                continue
            reply = turn.text
            if not without_urls(reply):
                stats["dropped_link_only"] += 1
                continue
            if len(without_urls(reply)) < opts.min_chars:
                stats["dropped_too_short"] += 1
                continue
            if len(without_urls(without_mentions(reply, mentions))) < opts.min_chars:
                stats["dropped_mention_only"] += 1
                continue
            # Fine to read in context, but as a target it teaches the adapter to
            # emit a placeholder — or, before replace_links, the digits themselves.
            if LONG_NUMBER in reply:
                stats["dropped_long_number"] += 1
                continue

            context: list[Turn] = []
            later = turn
            for prev in reversed(turns[max(0, i - opts.context_n):i]):
                if (later.start - prev.end).total_seconds() > gap:
                    break
                context.append(prev)
                later = prev
            if not context:
                stats["dropped_no_context"] += 1
                continue
            context.reverse()

            prompt = []
            if system_prompt is not None:
                prompt.append({"role": "system", "content": system_prompt})
            prompt.append({"role": "user", "content": "\n".join(_context_line(t) for t in context)})
            examples.append({
                "prompt": prompt,
                "completion": [{"role": "assistant", "content": reply}],
                # Kept for the time split and for tracing an example back; the
                # trainer ignores columns it does not know.
                "timestamp": turn.start.isoformat(),
            })
            stats["examples"] += 1

    examples.sort(key=lambda e: e["timestamp"])
    return examples, stats


def split_by_time(examples: list[dict], eval_fraction: float) -> tuple[list[dict], list[dict]]:
    """
    Newest slice is eval. Random splitting would put a message's neighbours —
    near-duplicates in a fast-moving channel — on both sides, and the eval loss
    would measure memory rather than voice.
    """
    if not examples or eval_fraction <= 0:
        return examples, []
    n_eval = max(1, int(len(examples) * eval_fraction))
    return examples[:-n_eval], examples[-n_eval:]


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build one member's mimic dataset from the archive")
    ap.add_argument(
        "--user-id", required=True,
        help="Discord user id of the member to mimic; comma-separate several for one member's old and new accounts",
    )
    ap.add_argument("--persona", required=True, help="Alias the adapter is served as, e.g. mimic_user1")
    ap.add_argument("--display-name", required=True, help="Name the persona prompt uses for them")
    ap.add_argument("--archive", default=os.environ.get("ARCHIVE_DIR", "/archive/archive"))
    ap.add_argument("--out", default=None, help="Default: /archive/training/<persona>")
    ap.add_argument("--prompt-template", default=os.path.join(HERE, "persona_prompts", "mimic_persona.md"))
    ap.add_argument("--no-system", action="store_true", help="Train without the persona system prompt")
    ap.add_argument("--context-n", type=int, default=12, help="Max context turns before each reply")
    ap.add_argument("--gap-minutes", type=float, default=30.0, help="Silence that ends a conversation")
    ap.add_argument("--merge-seconds", type=float, default=120.0, help="Same-author messages this close are one turn")
    # 24, not Options' 8: below it a reply is mostly "lmao same", and ~30% of
    # targets being that teaches the adapter to answer in two words.
    ap.add_argument("--min-chars", type=int, default=24, help="Shorter replies are dropped as targets")
    ap.add_argument("--eval-fraction", type=float, default=0.05)
    ap.add_argument(
        "--names",
        default="/archive/training/names.json",
        help="Hand-filled {user_id: {alias}} file; members without an alias keep their handle",
    )
    ap.add_argument(
        "--exclude-channels",
        default=os.environ.get("EXCLUDED_CHANNELS", ""),
        help="Comma-separated channel ids or names (same rule as history-service)",
    )
    args = ap.parse_args()

    opts = Options(
        user_ids=[u.strip() for u in args.user_id.split(",") if u.strip()],
        context_n=args.context_n,
        gap_minutes=args.gap_minutes,
        merge_seconds=args.merge_seconds,
        min_chars=args.min_chars,
        exclude_channels=[c.strip() for c in args.exclude_channels.split(",") if c.strip()],
        aliases=load_aliases(args.names),
    )
    logger.info(f"Aliases: {len(opts.aliases)} member(s) renamed from {args.names}")
    system_prompt = None if args.no_system else render_system_prompt(
        args.prompt_template, args.persona, args.display_name
    )

    examples, stats = build_examples(load_archive(args.archive), system_prompt, opts)
    if not examples:
        logger.error(f"No examples for user {args.user_id}: {dict(stats)}")
        return 1
    train, evaluation = split_by_time(examples, args.eval_fraction)

    # Keyed on the alias, not the member's id, so the committed configs that
    # point here never name a real person.
    out = args.out or os.path.join("/archive/training", args.persona)
    os.makedirs(out, exist_ok=True)
    write_jsonl(os.path.join(out, "train.jsonl"), train)
    write_jsonl(os.path.join(out, "eval.jsonl"), evaluation)
    with open(os.path.join(out, "stats.json"), "w") as fh:
        json.dump({"args": vars(args), "counts": dict(stats), "train": len(train), "eval": len(evaluation)}, fh, indent=2)

    for key, value in sorted(stats.items()):
        logger.info(f"  {key:28s} {value}")
    logger.info(f"Wrote {len(train)} train / {len(evaluation)} eval examples to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
