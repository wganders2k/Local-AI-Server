# Local AI Server

Monorepo for a personal AI server stack running on an Ubuntu machine with an RTX 3090. Hosts a Discord bot with mimic personas and a lore assistant, a self-hosted chat UI (LibreChat), and a VS Code coding assistant — all orchestrated through a single FastAPI proxy that manages VRAM allocation across two Ollama instances.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      Ubuntu Server (RTX 3090)                     │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              FastAPI Orchestration Proxy :11436            │  │
│  └───────────────────────────────────────────────────────────┘  │
│               │                          │                       │
│  ┌──────────────────┐       ┌───────────────────────────┐      │
│  │  Ollama :11435   │       │      Ollama :11434         │      │
│  │  (PERMANENT)     │       │      (SWAPPABLE)           │      │
│  │  autocomplete    │       │  brain / mimic / lore /    │      │
│  │  1.5B Q8_0       │       │  librechat (one at a time) │      │
│  └──────────────────┘       └───────────────────────────┘      │
└──────────────────────────────────────────────────────────────────┘
         ▲                    ▲                    ▲
    VS Code /             LibreChat            Discord Bot
    Cursor                :3080
```

For full architecture detail, see [`Design.md`](Design.md).  
For Discord bot specifics, see [`DiscordBot-Design.md`](DiscordBot-Design.md).

## Repository Structure

```
Local-AI-Server/
├── Design.md               # Full system design document
├── DiscordBot-Design.md    # Discord bot design document
├── docker-compose.yml      # All services defined here
├── Makefile                # Common server operations (see below)
├── .env.example            # Environment variable template — copy to .env
├── .gitignore
│
├── proxy/                  # FastAPI orchestration middleware (:11436)
├── discord-bot/            # discord.py bot
├── rag/                    # ChromaDB + embedding pipeline (CPU-only)
├── lora-training/          # Phase 3: Unsloth QLoRA fine-tuning scripts
└── librechat/              # LibreChat config (librechat.yaml)
```

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/wganders2k/Local-AI-Server.git
cd Local-AI-Server
cp .env.example .env
# Edit .env — set DISCORD_TOKEN and optionally ANTHROPIC_API_KEY
```

### 2. Start all services

```bash
make up
```

### 3. Verify GPU access

```bash
make check-gpu
```

### 4. Check service health

```bash
make status
```

## Common Commands

| Command | Description |
|---|---|
| `make up` | Start all services |
| `make down` | Stop all services (volumes preserved) |
| `make restart` | Full restart |
| `make pull` | `git pull` + rebuild changed images + restart |
| `make logs` | Tail all logs |
| `make logs-bot` | Tail Discord bot logs |
| `make logs-proxy` | Tail proxy logs |
| `make status` | Show container health |
| `make check-gpu` | Verify GPU visible in Ollama containers |
| `make ollama-ps` | Show models currently loaded in VRAM |
| `make ollama-list` | List all registered Ollama models |
| `make restart-bot` | Restart Discord bot only |
| `make shell-bot` | Open shell inside Discord bot container |
| `make nuke` | ⚠️ Wipe everything including volumes |

Run `make help` for the full list.

## Services

| Service | Port | Description |
|---|---|---|
| `ollama-permanent` | `:11435` | Permanent autocomplete model slot |
| `ollama-swappable` | `:11434` | Swappable model slot (brain / mimic / lore / librechat) |
| `proxy` | `:11436` | FastAPI orchestration proxy |
| `discord-bot` | — | Discord bot (no exposed port) |
| `librechat` | `:3080` | LibreChat web UI |
| `librechat-mongodb` | — | MongoDB sidecar for LibreChat |
| `rag-service` | — | RAG ingestion service |
| `chromadb` | — | ChromaDB vector store |

## Development Phases

| Phase | Status | Description |
|---|---|---|
| Phase 1 | 🔨 In progress | Core stack: Ollama + proxy + Discord bot + LibreChat |
| Phase 2 | ⏳ Planned | RAG pipeline: Discord history ingestion + lore retrieval |
| Phase 3 | ⏳ Planned | LoRA fine-tuning: per-member persona models |
| Phase 4 | ⏳ Optional | Hardening: rate limits, auth, priority queuing |

See `Design.md` §10 for the full development timeline.

## Secrets

Never commit `.env`. It contains `DISCORD_TOKEN` and optionally `ANTHROPIC_API_KEY`.  
Use `.env.example` as the template.
