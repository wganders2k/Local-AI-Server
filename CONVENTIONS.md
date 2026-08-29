# Conventions

How code in this repo is laid out and written. Most of this is description
rather than prescription — it is what the services already do, written down so
the next one starts in the same place instead of rediscovering it.

Where a service deviates, that is fine, but the deviation is documented in the
service's own README. An undocumented deviation is drift.

## 1. Service layout

One directory per deployable. Inside it:

```
<service>/
├── Dockerfile
├── requirements.txt
├── README.md          # what it does, how to run it, its env vars
├── config.py          # environment reads and tuning constants
├── <entrypoint>.py    # main.py, unless the README says otherwise
├── <modules>.py       # flat
├── pytest.ini
└── tests/
```

**Flat modules by default.** Subpackages only once a service is large enough
that a flat listing stops telling you where anything lives — in practice past
roughly 1,500 lines. `discord-bot/` is currently the only one that qualifies,
and its README says why.

**One entrypoint per service**, named in the README and referenced by the
Dockerfile `CMD`. It is `main.py` for `proxy`, `arbiter`, `rag` and
`history-service`; `bot.py` for `discord-bot`; `supervisor.py` for
`lora-training`. Keeping the name meaningful beats keeping it uniform.

## 2. `config.py` is configuration only

Environment reads and tuning constants. Not prompt text, not regex libraries,
not object factories — those belong with the code that uses them.

Prompt text lives in `prompts/*.md`, loaded and rendered at use. It is committed
code baked into the image: a prompt is part of the program's behaviour, so
changing one is a rebuild, and it is versioned with the module that renders it.
Text that is host-local instead — anything holding real names or server-specific
content — lives on a mounted volume, never in `prompts/`, and never in the repo.

A constant that has to agree with something outside the file says so in a
comment naming the other place. Better still: if that other place can be
*queried* at runtime, query it, and keep the constant only as a fallback for
when the query fails. `discord-bot/lore/context_window.py` reads the agent's
context window from the proxy rather than trusting a number somebody has to
remember to update, and logs a warning when the fallback has gone stale — a
pairing that used to be maintained by hand and checked by nobody.

## 3. Comments explain *why*, with numbers

This is the most valuable habit in the codebase and the easiest to lose. The
bar, from existing code:

```python
# PROXY_READ_TIMEOUT is the gap allowed between reads, not a total deadline.
# ... a tool-calling round that thinks for 3.4k tokens at ~29 tok/s took 121.6s
# and was killed 1.6s short. Nothing cancels the backend when that fires, so the
# work is wasted AND the GPU lock is held to completion — hence the headroom.
```

```python
# The agent model is a hybrid reasoner and deliberates by default even on
# mechanical work — measured at 52.2s and 724 completion tokens with thinking
# on, versus 1.0s and 27 tokens with it off, for the same output.
```

What makes these worth keeping is that they record a *measurement* and a
*failure that was actually observed*. A comment restating the code ("increment
the counter") is noise; a comment saying what broke, what it cost, and what was
tried instead is the only durable record of it. Write those, especially for
values that look arbitrary — every timeout, threshold and retry count should say
where its number came from.

Prefer a docstring that says what a thing is *for* over one that lists its
parameters. Note trade-offs that were made deliberately, and rejected
alternatives with the reason they were rejected.

## 4. Deliberate duplication is documented at the copy

`proxy/arbiter.py` and `lora-training/arbiter.py` are near-identical clients for
the arbiter service. That is intentional, and each file says so:

> Deliberately a copy of the two calls in `proxy/arbiter.py` rather than a shared
> library. The previous design vendored a client into three repos and every
> change to the protocol became a three-repo change; the protocol is now small
> enough that duplicating forty lines is cheaper than coupling two images' build
> contexts.

The rule: vendor small protocol clients rather than coupling build contexts, and
always say so in the file. Duplication that explains itself is a decision;
duplication that doesn't is a bug waiting to be half-fixed.

## 5. Persistence

- JSON stores write to a temp file and `os.replace()` onto the target, so a
  crash mid-write cannot truncate the real file. `ThreadRegistry` and
  `LoreSessionStore` both do this.
- A corrupt or unreadable store logs a warning and starts empty. It never stops
  the service from booting.
- Runtime state lives under a mounted volume outside the repo, never in the
  service directory, and its path is gitignored. Anything holding real user
  content — archives, transcripts, alias indexes — belongs with the archive that
  produced it, not in version control.

## 6. Compose

- Bind mounts use **absolute** host paths. Compose resolves relative binds
  against the invoking directory, so bringing the stack up through dockge
  silently repoints them at an empty directory — with nothing in the logs to say
  the mount moved.
- Never bind-mount over a directory the image is supposed to provide. A
  read-only bind of a source directory shadows the image's copy with whatever is
  in the working tree.
- Every service-tunable environment variable appears in `.env.example` and in
  that service's README table. A variable nothing reads gets deleted from all
  three.
- Services that must never start from a plain `docker compose up` use a
  `profiles:` entry rather than convention. See `lora-trainer`.

## 7. Tests

`pytest.ini` at the service root:

```ini
[pytest]
testpaths = tests
pythonpath = .
asyncio_mode = auto
```

`rag/` is the outstanding gap: it has no tests, and so deliberately has no
`pytest.ini` either — configuring a suite that cannot run is worse than an
obvious absence.

Tests run against a local `.venv-test/` (gitignored), holding only what the
tests need — deliberately not the heavy runtime dependencies, so the suite runs
without CUDA or a GPU.

Test the logic, not the framework. The most useful boundary a service can have
is one that keeps its core reachable without its transport: `discord-bot/lore/`
imports no discord, so the whole agent loop is exercised against a fake backend.
Aim for that shape when a service grows a hard-to-test dependency.

## 8. Style

- Type hints on function signatures. Dataclasses for state that travels together.
- Log through a named logger (`logging.getLogger("<service>.<area>")`), with
  `%s` placeholders rather than f-strings, so filtering and lazy formatting work.
- Catch narrow exceptions where the difference matters, broad ones only at a
  boundary that must not fall over — and log with `exception()` when you do.
- Anything user-facing degrades rather than raises: an unreachable service
  returns empty and says so in the logs.
