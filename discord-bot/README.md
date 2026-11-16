# Discord Bot

`discord.py` bot exposing `/mimic`, `/chat` and `/lore`, routed to models through
the orchestration proxy. Handles typing indicators, rate limiting, conversation
history, disclaimer stripping, and the agentic RAG loop behind `/lore`.

## Commands

| Command | Parameters | Description |
|---|---|---|
| `/mimic` | `persona` (autocomplete) | The persona replies to the channel's recent conversation, posted through a webhook under the member's nickname and avatar |
| `/mimic-chat` | `persona1`–`persona7` (autocomplete), `length` (2–20, optional) | Creates a thread where the personas chat with each other and with anyone who joins in. It opens by asking the starter to say something, and the personas answer once someone has. A controller picks who speaks next and pauses at natural breaks; `length` makes the first burst run that many turns without pausing |
| `/chat` | `model` (autocomplete) | Creates a thread bound to one model; the bot answers every message posted in it |
| `/lore` | `question`, `rounds` (1–25, default 10) | Agentic RAG over the Discord archive, then an optional follow-up thread |
| `/imagine` | `prompt` and `rewrite` (both required), `count` (1–3 images, default 1), `style`, `mood`, `framing` (optional details), `shape` (square / portrait / landscape), `effort` (low = 8 steps, high = 16), `image` (a picture to start from), `strength` (light / medium / heavy), `cfg` (guidance 1–7; above 1 takes ~2x as long) | An image from a text description, or from a picture (~7 s warm). The detail options follow the prompt as labelled sentences ("Art style: …"), which the rewriter keeps exactly. With `rewrite`, the mimic base (no adapter) first expands the prompt into a detailed description (`prompts/imagine_rewrite.md`), shown under the original; any failure draws the original. It adds ~10 s, because the image model needs the whole card and the mimic base has to be swapped in and out. `count` above 1 makes that many images from the one prompt, each with its own seed (~7 s each), attached together. The footer shows how long each step took. With `image` it is image-to-image: the result starts from the picture and moves from it by `strength` (medium by default), the prompt saying what it should become; without a `shape`, the one nearest the picture's is used. The proxy crops and re-encodes the picture. Registered only when `IMAGINE_ENABLED` |

### Disabled Commands

| Command | Status | Reason |
|---|---|---|
| `/admin-clear-history` | **Removed** | No authorization checks — any user could clear history. Re-add only with role/permission verification. |

## How `/lore` works

1. The agent model (`AGENT_MODEL`, currently `brain-dense-heretic`) is given the
   tools in [`lore/tools.py`](lore/tools.py) and loops: call a tool, read the
   result, decide whether to search again — up to `rounds` times.
2. If it answers before the budget runs out, that is the answer. If it doesn't,
   the research is flattened into plain text and a separate synthesis call writes
   one. (Flattened, because leaving tool-call markup in context teaches the model
   to reply with tool-call markup.)
3. The answer posts as embeds, followed by an offer message. React with ❓ and the
   bot opens a thread seeded with the research it already did — so follow-ups
   usually cost no searching at all.
4. Threads are swept after `LORE_THREAD_TTL_SECONDS` of silence, which is what
   actually reclaims the disk: a session stores every excerpt verbatim.

Two searching styles are exposed on purpose, because they fail in opposite ways:
semantic search finds paraphrases but only returns top-k and cannot count or
order; literal search matches exact text across every message in time order and
reports totals. The prompt in [`prompts/lore_agent.md`](prompts/lore_agent.md)
spends most of its length teaching the model which to reach for. A third,
`search_images`, searches only the descriptions of posted images. Conversation
search never looks inside images; it shows them as `[image: …]` once a chunk has
been found (see [`rag/README.md`](../rag/README.md)). `read_surrounding_messages`
reads on from, or back from, any message the agent has seen (by its ref), for
when an excerpt stops in the middle of an exchange.

**Citations.** Every retrieved message carries a ref (`[m12]`) that
[`lore/citations.py`](lore/citations.py) assigns per conversation, so a message
keeps its ref across searches and follow-ups. The model cites refs inline, and
the posted answer turns them into numbered links to the message (`[¹](…)`). The
saved transcript keeps the plain refs. Links point into `LORE_ARCHIVE_GUILD_ID`
and only open for members of that server. With it unset, refs are stripped.

## Privileged intents

Both must be enabled in the Discord Developer Portal or the bot will not start:

| Intent | Why |
|---|---|
| `MESSAGE_CONTENT` | `on_message` reads message text inside `/chat` and `/lore` threads |
| `SERVER_MEMBERS` | Display-name resolution for message labelling |

Slash commands alone would need neither. `/chat` is what reintroduced the requirement.

## Message labelling

User messages are prefixed with the sender's display name — `[Alice]: rate my strats` — in `/chat`. Channels and threads have several participants, and without labels the model sees one undifferentiated user. The labelled form is what is stored in history, so attribution survives across turns.

`/mimic` and `/mimic-chat` go further: they read the channel's recent history and render it exactly as the persona's training data was rendered — aliases from `names.json`, merged turns, and the same window: the persona's trained `context_n` turns (24), cut at the first silence longer than its `gap_minutes` (2 hours), both read from its `stats.json` — though a silence never leaves fewer than 4 turns (`MIMIC_MIN_CONTEXT_TURNS`), so a chat resumed after a break keeps a little of what came before. A wider window made personas cling to the topic that dominated it instead of answering the newest line — using `shared/mimic_format.py`, the same module `lora-training/build_dataset.py` uses (see `mimic/transcript.py`). Persona lines are posted through a per-channel webhook under the mimicked member's current server nickname and avatar (suffixed "(mimic)"), so they read back as the member the persona mimics (`persona_posting.py`). Both need **Manage Webhooks** and **Read Message History** in the channel.

### Resetting the window

A person can cut what the personas see by starting a message with a prefix (`MIMIC_RESET_PREFIXES`):

| Prefix | Persona windows from then on |
|---|---|
| `>>` | that message and the 3 turns before it, then everything after |
| `>>>>` | that message alone, then everything after |

The prefix is stripped before anything reads the message, and the latest one wins. `>>>` does nothing, because it is Discord's multi-line quote: quoting a passage must not reset the chat. In a `/mimic-chat` thread the bot reacts ✂️ to a message it read as a reset (needs **Add Reactions**), and the thread's opening message explains both prefixes. It works in `/mimic-chat` threads and for `/mimic`. Only the persona prompts are cut: the chat controller still reads the whole transcript, so it knows who has been talking. A persona's `>>` is just text, and a message that is nothing but a prefix is dropped as empty and resets nothing.

## Images

`vision.py` describes images posted to the bot, so the models can react to what an image shows rather than to a bare `[attachment]`. A caption is made by the model the feature already has loaded, so it never costs a model swap. Captions are cached by attachment id.

- **Lore threads.** An image posted with or as a follow-up is described by the agent model (`brain-dense-heretic`, which has a vision projector in `models.ini`). The description is appended to the question as `[image: …]`, so sessions stay plain text.
- **Mimic chats.** Images are described by `[mimic-multi]` with no adapter applied, newest first, at most `VISION_MAX_NEW_PER_BURST` new ones per burst. They are captioned only when the controller is about to be asked. The controller sees `[image: …]`. **Persona prompts still show `[attachment]`**, because that is what the adapters were trained on (`shared/mimic_format.context_line`); captions live on `mimic.transcript.Line`, never on the shared `Msg`. Off by default (`VISION_MIMIC_ENABLED`): the 12B base's projector needs a newer llama.cpp than the stack runs; see `[mimic-multi]` in `models.ini`.

The caption prompt is `prompts/caption.md`, identical to history-service's, so a live caption reads like an archived one.

## Personas

Nothing in this directory lists the personas. `personas.PersonaCatalog` assembles them from:

| Fact | Source |
|---|---|
| who exists, display name, Discord accounts | the roster, `personas.json` beside the training data (real names, so never committed) |
| adapter id | position in `[mimic-multi]`'s `lora` list in `models.ini`, read from the proxy's `/v1/models` at startup |
| how context was cut | each persona's `stats.json`, written by `build_dataset.py` |
| system prompt | `shared/mimic_persona.md`, rendered by the same function training used |

A persona is offered only when it is on the roster **and** has an adapter. The adapter order is never guessed: until the proxy has answered, persona commands say the personas are unavailable and retry discovery on the next command, because a wrong id serves one member's voice under another's name. The roster and stats are re-read when they change, so neither needs a restart. Adding a persona is a training-side job — see `lora-training/README.md`.

All personas are served from `[mimic-multi]`: one loaded base with every adapter, picked per request, so changing speaker never reloads the model.

## File structure

```
discord-bot/
├── bot.py               # Entry point: client, setup_hook, thread restore/delete
├── services.py          # Shared state, built once
├── config.py            # Environment reads and tuning constants — nothing else
├── personas.py          # PersonaCatalog: roster ∩ [mimic-multi] adapters
├── threads.py           # ThreadDirectory: which threads the bot answers, persisted
├── persona_posting.py   # Discord side of personas: webhooks, reading posts back
├── prompt_loader.py     # Loads and renders prompts/*.md
├── discord_io.py        # Streaming and posting
├── formatters.py        # Disclaimer stripping, splitting to Discord's limits, embeds
├── history.py           # Rolling per-thread window for /chat
├── rate_limiter.py      # Per-user sliding window
├── proxy_client.py      # httpx client for the proxy (/v1/chat/completions)
├── rag_client.py        # httpx client for the RAG service
├── mimic/               # Multi-persona chat. Imports no discord.
│   ├── transcript.py    #   Line, Cast, the training-shaped persona prompt
│   ├── engine.py        #   the decide/write/post loop, and lore's simulations
│   ├── controller.py    #   who speaks next
│   ├── addressing.py    #   who a human is talking to
│   └── state.py         #   what a /mimic-chat thread remembers
├── lore/                # The /lore agent. Imports no discord — see below.
│   ├── prompts.py       #   assembles prompts from prompts/*.md
│   ├── tools.py         #   tool schemas + the executor behind them
│   ├── agent.py         #   the tool-calling loop
│   ├── streaming.py     #   streaming one completion, with progress
│   ├── compaction.py    #   condensing oldest research, mid-run or between turns
│   ├── research.py      #   rendering tool results as plain text, dedup
│   ├── session.py       #   LoreSession + its JSON store
│   ├── citations.py     #   [mN] refs for retrieved messages, and their links
│   ├── metrics.py       #   timing, token accounting, char->token estimates
│   ├── context_window.py #  the agent model's context size, read from the proxy
│   └── progress.py      #   ProgressReporter protocol
├── cogs/                # Discord presentation layer
│   ├── common.py        #   admission checks and autocompletes every command shares
│   ├── mimic.py  mimic_chat.py  chat.py  lore.py
│   └── lore_status.py   #   the /lore progress message
├── prompts/             # Prompt text, committed, baked into the image
└── tests/
```

`shared/` (one level up) is copied into the image beside these modules: the prompt
format, the persona system prompt and the roster reader that training uses too.

**`lore/`, `mimic/` and `personas.py` must not import discord.** That boundary is
what makes the agent loop and the persona chat testable without a gateway (see
`tests/test_agent.py` and `tests/test_mimic_chat.py`, which drive full runs against
a fake backend); `tests/test_boundaries.py` enforces it. Progress reporting crosses
it through `lore.progress.ProgressReporter`, which `cogs.lore_status.LoreStatus`
implements; Discord history crosses it as `mimic.transcript.Line`s and a `Cast`.

This is the only service in the repo with subpackages; see
[`CONVENTIONS.md`](../CONVENTIONS.md). Past ~4,400 lines a flat layout stopped
distinguishing "the agent" and "the persona chat" from "the Discord client", which
is exactly the split these directories restore.

## Prompts

All prompt text lives in `prompts/*.md` and is rendered by `prompt_loader.render()`.
`prompts/` is **committed code baked into the image** — editing a prompt is a
rebuild, which is correct for text that is part of the program's behaviour. Do not
bind-mount over it: the container would then see the working tree instead of the
image, and a stale checkout would silently change the model's instructions.

### Setup: lore context file

The `/lore` agent injects server-specific background (member alias index, persona
notes) into its system prompt. That content holds real names, so it is **not
committed** and does **not** live in `prompts/`. It sits on the `bot_context`
volume beside the session store:

```bash
cp discord-bot/prompts/lore_context.example.md \
   /mnt/storage/array/DiscordArchive/bot_context/lore/context.md
# then edit it with your server's real members
```

If the file is missing the bot still starts and `/lore` still works; it logs a
warning and answers without the alias index.

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `DISCORD_TOKEN` | ✅ | — | Bot token from Discord Developer Portal |
| `PROXY_URL` | ✅ | `http://proxy:11436` | Proxy base URL |
| `MAX_QUEUE_DEPTH` | ❌ | `3` | Max queued backend requests before the command refuses |
| `RATE_LIMIT_PER_USER` | ❌ | `5` | Max requests per user per minute |
| `RAG_SERVICE_URL` | ❌ | `http://rag-service:8001` | RAG service base URL |
| `RAG_ENABLED` | ❌ | `true` | Set false to run without `/lore` retrieval |
| `LORE_ARCHIVE_GUILD_ID` | ❌ | `DISCORD_GUILD_ID` | Guild citation links point into; unset = no links |
| `VISION_MIMIC_ENABLED` | ❌ | `false` | Live image captions for the mimic chat controller. Needs a vision projector on `[mimic-multi]` first |
| `IMAGINE_ENABLED` | ❌ | `false` | Registers `/imagine`. Needs the proxy's `IMAGEGEN_ENABLED` |
| `IMAGINE_MODEL_LABEL` | ❌ | `Z-Image-Turbo` | Model name in each image's footer; set it to match the proxy's `IMAGEGEN_*` model |
| `THREAD_REGISTRY_PATH` | ❌ | `bot_context/threads.json` | Which threads the bot answers in, and how |
| `LORE_SESSION_PATH` | ❌ | `bot_context/lore/sessions.json` | Lore thread transcripts |
| `LORE_CONTEXT_PATH` | ❌ | `bot_context/lore/context.md` | Host-local server background for the `/lore` prompt |
| `MIMIC_TRAINING_DIR` | ❌ | `/archive/training` | Training data dir: roster, `names.json`, per-persona `stats.json` |
| `PERSONA_ROSTER` | ❌ | `<MIMIC_TRAINING_DIR>/personas.json` | The persona roster |
| `PERSONA_NAMES_PATH` | ❌ | `bot_context/persona_names.json` | Every webhook name a persona has posted under |
| `MIMIC_ADAPTER_SCALE` | ❌ | `0.85` | How strongly a persona's adapter is applied |

Tuning that is not environment-driven — round caps, context thresholds, TTLs —
lives in [`config.py`](config.py) with the reasoning attached.

The agent's context window is **read from the backend at startup**, not
configured: `lore/context_window.py` asks the proxy's `/v1/models` for the argv
llama-server would launch `AGENT_MODEL` with and takes `--ctx-size` from it. So
editing `ctx-size` in [`models.ini`](../models.ini) is enough — nothing in the bot
needs changing to match. `AGENT_CTX_LIMIT_FALLBACK` in `config.py` is only used if
the proxy is unreachable, and a disagreement between the two is logged as a stale
constant.

## Timeouts

Connect 10s, read 300s, total 320s (`PROXY_*_TIMEOUT` in `config.py`). Long because
a request may wait on an 18 GB model swap *and* on the arbiter preempting a junior
GPU tenant before the first token arrives. The read timeout is a gap-between-reads,
not a total deadline — which is why every long generation is streamed rather than
buffered; see `ProxyClient.chat_stream`.

The RAG client has its own 30s timeout and degrades to empty context rather than
raising, so a slow retrieval costs the agent one tool result instead of failing the
command.

## Tests

```bash
make test-venvs          # once: .venv-test/ from requirements-test.txt, what CI installs
cd discord-bot && .venv-test/bin/python -m pytest
```

## Design Reference

See [`docs/design/discord-bot.md`](../docs/design/discord-bot.md) for full detail,
and [`docs/design/system.md`](../docs/design/system.md) §8 for request flow examples.
