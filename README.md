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

## Model Management

Ollama model definitions live in [`modelfiles/`](modelfiles/). Each `.Modelfile` defines the GGUF source (`FROM` line), inference parameters, and system prompt for one named model. These are version-controlled — changing a model means editing its Modelfile and re-registering.

### First-time setup (after `make up`)

```bash
make models-init
```

This registers all core models. Ollama will pull the GGUF weights from HuggingFace or the Ollama registry as defined in each Modelfile. Large models (Brain ~17.8 GB, LibreChat ~9.5 GB) will take time on first download.

### Re-register a model (uses cached weights)

```bash
make model-create MODEL=librechat_chat SLOT=swappable
```

Use this after editing a Modelfile's parameters or system prompt — the GGUF is already cached so it's fast.

### Switch to a different model or quant

1. Edit the `FROM` line in `modelfiles/<name>.Modelfile`
2. Run `make model-redownload MODEL=<name> SLOT=<permanent|swappable>`

Ollama fetches the new GGUF from the updated source. The proxy references models by name only — no proxy changes needed.

### Force a clean re-fetch (corrupt blob, sanity check)

```bash
make model-redownload MODEL=brain SLOT=swappable
```

Removes the registered model and re-creates it with `--no-cache`, forcing Ollama to re-fetch the GGUF even if a blob is cached.

### Post-nuke recovery

```bash
make up
make models-init
```

`make nuke` wipes all volumes including model weights. After bringing services back up, `models-init` re-downloads everything from scratch.

### Adding a mimic persona

```bash
cp modelfiles/mimic.Modelfile modelfiles/mimic_alice.Modelfile
# Edit mimic_alice.Modelfile — replace <member> with alice in the SYSTEM block
make model-create MODEL=mimic_alice SLOT=swappable
```

See [`modelfiles/README.md`](modelfiles/README.md) for full details.

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
