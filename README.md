# Local AI Server

(Disclaimer: documentation generated with AI assistance) 

Monorepo for a personal AI server stack running on an Ubuntu machine with an RTX 3090. Hosts a Discord bot with mimic personas and a lore assistant, a self-hosted chat UI (LibreChat), and a VS Code coding assistant — all orchestrated through a single FastAPI proxy that manages VRAM allocation across two llama-server instances.

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
│  │ llama-server      │       │    llama-server :11434     │      │
│  │ :11435            │       │    (SWAPPABLE — router)    │      │
│  │ (PERMANENT)       │       │                            │      │
│  │ autocomplete      │       │  brain / mimic / lore /    │      │
│  │ Qwen3.5-2B IQ4_NL│       │  librechat (one at a time) │      │
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
├── models.ini              # llama-server model presets (swappable slot)
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
├── librechat/              # LibreChat config (librechat.yaml)
└── modelfiles/             # Legacy Ollama Modelfiles (historical reference only)
```

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/wganders2k/Local-AI-Server.git
cd Local-AI-Server
cp .env.example .env
# Edit .env — set DISCORD_TOKEN and optionally ANTHROPIC_API_KEY and HF_TOKEN
```

### 2. Download model files

```bash
make models-download
```

This downloads all GGUF files from HuggingFace into `./models/<publisher>/<model>/`. Large models (Brain ~17.8 GB, LibreChat ~9.5 GB) will take time on first download. Already-downloaded files are skipped on subsequent runs.

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
| `make logs-history` | Tail history-service logs (LoRA training progress) |
| `make logs-proxy` | Tail proxy logs |
| `make logs-llama` | Tail both llama-server logs |
| `make status` | Show container health |
| `make check-gpu` | Verify GPU visible in llama-server containers |
| `make llama-ps` | Show models available in both llama-server instances |
| `make restart-bot` | Restart Discord bot only |
| `make restart-llama-swappable` | Restart swappable slot (picks up models.ini changes) |
| `make shell-bot` | Open shell inside Discord bot container |
| `make nuke` | ⚠️ Wipe all volumes (model files in ./models are preserved) |

Run `make help` for the full list.

## Model Management

Model configuration lives in two places:

- **`models.ini`** — defines all swappable slot models (brain, mimic personas, lore, librechat, image-caption). Each `[section]` is a named model with its GGUF path, inference parameters, and alias.
- **`docker-compose.yml`** — defines the permanent slot model (autocomplete) via the `--model` flag on the `llama-permanent` service.

GGUF files are stored on the host at `./models/<publisher>/<model>/filename.gguf` and bind-mounted read-only into both llama-server containers.

### First-time setup (after `make up`)

```bash
make models-download
```

Downloads all GGUFs defined in `scripts/download_models.py`. Safe to re-run — already-downloaded files are skipped.

### Adding a mimic persona

1. Add a new `[mimic_<member>]` section to `models.ini` (copy an existing mimic section, change the alias)
2. Add `mimic_<member>` to `SWAPPABLE_MODELS` in `proxy/config.py`
3. Add the persona to `MENTION_TO_MODEL` in the Discord bot's `router.py`
4. Restart the swappable server: `make restart-llama-swappable`

No download needed — all mimic personas share the same GGUF.

### Switching to a different model or quant

1. Edit the `model` path in `models.ini` (or `--model` flag in `docker-compose.yml` for permanent slot)
2. Update `scripts/download_models.py` with the new `repo_id` and `filename`
3. Run `make models-download`
4. Restart: `make restart-llama-swappable` (or `restart-llama-permanent`)

See [`modelfiles/README.md`](modelfiles/README.md) for full model management details.

## Services

| Service | Port | Description |
|---|---|---|
| `llama-permanent` | `:11435` | Permanent autocomplete model slot |
| `llama-swappable` | `:11434` | Swappable model slot (brain / mimic / lore / librechat) |
| `proxy` | `:11436` | FastAPI orchestration proxy |
| `discord-bot` | — | Discord bot (no exposed port) |
| `librechat` | `:3080` | LibreChat web UI |
| `librechat-mongodb` | — | MongoDB sidecar for LibreChat |
| `history-service` | — | Background message collection + LoRA retraining trigger |
| `rag-service` | — | RAG ingestion service |
| `chromadb` | — | ChromaDB vector store |

## Development Phases

| Phase | Status | Description |
|---|---|---|
| Phase 1 | 🔨 In progress | Core stack: llama-server + proxy + Discord bot + LibreChat |
| Phase 2 | ⏳ Planned | RAG pipeline: Discord history ingestion + lore retrieval |
| Phase 3 | ⏳ Planned | LoRA fine-tuning: per-member persona models |
| Phase 4 | ⏳ Optional | Hardening: rate limits, auth, priority queuing |

See `Design.md` §10 for the full development timeline.

## Secrets

Never commit `.env`. It contains `DISCORD_TOKEN`, optionally `ANTHROPIC_API_KEY`, and optionally `HF_TOKEN` for private HuggingFace repos.  
Use `.env.example` as the template.
