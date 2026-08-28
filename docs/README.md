# Documentation

Cross-cutting design lives here. **Per-service operational docs stay with their
service** — `proxy/README.md`, `arbiter/README.md`, `discord-bot/README.md` and so
on are the first thing to read when working on one, and the last thing to update
when changing it.

| Document | What it is | Currency |
|---|---|---|
| [`design/system.md`](design/system.md) | Full system design: hardware budget, proxy state machine, model routing, RAG pipeline, phase plan. The reference for how the pieces fit. | Living |
| [`design/discord-bot.md`](design/discord-bot.md) | Discord bot design parameters in detail. | Living |
| [`../CONVENTIONS.md`](../CONVENTIONS.md) | How code in this repo is laid out and written. Read before adding a service. | Living |
| [`TODO.md`](TODO.md) | Loose ends not worth a plan document. | Living |
| [`plans/`](plans/) | Implementation plans, written before the work and kept afterwards as a record of *why*. Historical — a plan describes intent at the time it was written, not necessarily current behaviour. | Historical |

When a plan is finished, leave it in `plans/`. The reasoning in it is usually the
only surviving record of what was tried and rejected; the code only shows what
won.
