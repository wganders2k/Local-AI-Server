# Local AI Server

(Disclaimer: documentation generated with AI assistance)

Monorepo for a personal AI server stack running on an Ubuntu machine with an RTX 3090. Hosts a Discord bot with mimic personas and a lore assistant, a self-hosted chat UI (Open WebUI), and a VS Code coding assistant — all orchestrated through a single FastAPI proxy that manages VRAM allocation across two llama-server instances.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      Ubuntu Server (RTX 3090)                    │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │              FastAPI Orchestration Proxy :11436           │   │
│  └───────────────────────────────────────────────────────────┘   │
│               │                          │                       │
│  ┌──────────────────┐       ┌───────────────────────────┐        │
│  │ llama-server     │       │    llama-server :11434    │        │
│  │ :11435           │       │    (SWAPPABLE — router)   │        │
│  │ (PERMANENT)      │       │                           │        │
│  │ autocomplete     │       │  brain / mimic / lore /   │        │
│  │ Qwen3.5-2B IQ4_NL│       │  chat (one at a time)     │        │
│  └──────────────────┘       └───────────────────────────┘        │
└──────────────────────────────────────────────────────────────────┘
        ▲                     ▲                    ▲
    VS Code /             Open WebUI           Discord Bot
    Cursor                :3000
```

For full architecture detail, see [`Design.md`](Design.md).  
For Discord bot specifics, see [`DiscordBot-Design.md`](DiscordBot-Design.md).

## Repository Structure

```
local-ai-server/
├── Design.md               # Full system design document
├── DiscordBot-Design.md    # Discord bot design document
├── docker-compose.yml      # All services defined here
├── Makefile                # Common server operations (see below)
├── models.ini              # llama-server model presets (swappable slot)
├── system_prompts.ini      # Per-persona system prompts
├── .env.example            # Environment variable template — copy to .env
├── .gitignore
│
├── scripts/                # Utility scripts
│   ├── download_models.py  # Download all GGUFs from HuggingFace
│   └── requirements.txt    # Python deps for scripts (huggingface_hub)
│
├── models/                 # GGUF model files (gitignored, populated by make models-download)
│   └── <publisher>/<model>/filename.gguf
│
├── proxy/                  # FastAPI orchestration middleware (:11436)
├── discord-bot/            # discord.py bot
├── history-service/        # Background message collection + LoRA retraining trigger
├── rag/                    # ChromaDB + embedding pipeline (CPU-only)
├── lora-training/          # Phase 3: Unsloth QLoRA fine-tuning scripts
├── open-webui/             # Open WebUI config
└── modelfiles/             # Legacy Ollama Modelfiles (historical reference only)
```

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/wganders2k/local-ai-server.git
cd local-ai-server
cp .env.example .env
# Edit .env — set DISCORD_TOKEN and optionally ANTHROPIC_API_KEY and HF_TOKEN
```

### 2. Download model files

```bash
make models-download
```

Downloads all GGUFs from HuggingFace into `/srv/models/<publisher>/<model>/`. Large models (Brain ~17.8 GB, Mimic ~18 GB) will take time on first run. Already-downloaded files are skipped.

### 3. Start all services

```bash
make up
```

### 4. Verify GPU access

```bash
make check-gpu
```

### 5. Check service health

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
| `make logs-history` | Tail history-service logs |
| `make logs-proxy` | Tail proxy logs |
| `make logs-llama` | Tail both llama-server logs |
| `make status` | Show container health |
| `make check-gpu` | Verify GPU visible in llama-server containers |
| `make llama-ps` | Show models available in both llama-server instances |
| `make restart-bot` | Restart Discord bot only |
| `make restart-llama-swappable` | Restart swappable slot (picks up models.ini changes) |
| `make shell-bot` | Open shell inside Discord bot container |
| `make nuke` | ⚠️ Wipe all volumes (model files in /srv/models are preserved) |

Run `make help` for the full list.

## Model Management

Model configuration lives in two places:

- **`models.ini`** — defines all swappable slot models (brain, mimic personas, lore, chat, image-caption). Each `[section]` is a named preset with its GGUF path, inference parameters, and alias.
- **`docker-compose.yml`** — defines the permanent slot model (autocomplete) via the `--model` flag on the `llama-permanent` service.

GGUF files are stored on the host at `/srv/models/<publisher>/<model>/filename.gguf` and bind-mounted read-only into both llama-server containers.

### Adding a mimic persona

1. Add a new `[mimic_<member>]` section to `models.ini` (copy an existing mimic section, change the alias)
2. Add `mimic_<member>` to `SWAPPABLE_MODELS` in `proxy/config.py`
3. Add the persona to `MENTION_TO_MODEL` in the Discord bot's `router.py`
4. Restart the swappable server: `make restart-llama-swappable`

No download needed — all mimic personas share the same GGUF.

### Switching to a different model or quant

1. Edit the `model` path in `models.ini` (or `--model` flag in `docker-compose.yml` for the permanent slot)
2. Update `scripts/download_models.py` with the new `repo_id` and `filename`
3. Run `make models-download`
4. Restart: `make restart-llama-swappable` (or `restart-llama-permanent`)

## Services

| Service | Port | Description |
|---|---|---|
| `llama-permanent` | `:11435` | Permanent autocomplete model slot |
| `llama-swappable` | `:11434` | Swappable model slot (brain / mimic / lore / chat) |
| `proxy` | `:11436` | FastAPI orchestration proxy |
| `discord-bot` | — | Discord bot (no exposed port) |
| `open-webui` | `:3000` | Open WebUI chat interface |
| `history-service` | — | Background message collection + LoRA retraining trigger |
| `rag-service` | — | RAG ingestion service |
| `chromadb` | — | ChromaDB vector store |

## Development Phases

| Phase | Status | Description |
|---|---|---|
| Phase 1 | 🔨 In progress | Core stack: llama-server + proxy + Discord bot + Open WebUI |
| Phase 2 | ⏳ Planned | RAG pipeline: Discord history ingestion + lore retrieval |
| Phase 3 | ⏳ Planned | LoRA fine-tuning: per-member persona models |
| Phase 4 | ⏳ Optional | Hardening: rate limits, auth, priority queuing |

See `Design.md` §10 for the full development timeline.
