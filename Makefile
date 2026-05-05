# ============================================================
# Local AI Server — Makefile
# ============================================================
# Common operations for managing the Local AI Server stack.
#
# Usage:  make <target>
#
# All targets run from the repo root alongside docker-compose.yml.
# Requires: docker, docker compose (v2), git, python3, python3-venv
# ============================================================
# Load environment variables from .env if it exists
ifneq (,$(wildcard ./.env))
    include .env
    export
endif

.PHONY: help up down restart build pull \
        logs logs-proxy logs-bot logs-llama logs-openwebui logs-rag logs-history \
        status \
        restart-bot restart-proxy restart-openwebui restart-rag \
        shell-bot shell-proxy shell-rag \
        llama-ps llama-models \
        check-gpu \
        models-download \
        dce-export-full dce-export-channel dce-export-guild dce-help \
        dce-evaluate history-refresh \
        nuke

# Default target — show help
help:
	@echo ""
	@echo "  Local AI Server — available make targets"
	@echo ""
	@echo "  ── Lifecycle ──────────────────────────────────────────"
	@echo "  up                  Start all services (detached)"
	@echo "  down                Stop and remove containers (volumes preserved)"
	@echo "  restart             down + up"
	@echo "  build               Rebuild all custom images without starting"
	@echo "  pull                git pull + rebuild changed services + restart"
	@echo ""
	@echo "  ── Logs ───────────────────────────────────────────────"
	@echo "  logs                Tail logs from all services"
	@echo "  logs-proxy          Tail proxy logs only"
	@echo "  logs-bot            Tail discord-bot logs only"
	@echo "  logs-llama          Tail both llama-server instance logs"
	@echo "  logs-openwebui      Tail Open WebUI logs only"
	@echo "  logs-rag            Tail RAG service logs only"
	@echo "  logs-history        Tail history-service logs only"
	@echo ""
	@echo "  ── Status ─────────────────────────────────────────────"
	@echo "  status              Show running containers and health"
	@echo "  check-gpu           Verify GPU is visible inside llama-server containers"
	@echo "  llama-ps            Show models currently loaded in both llama-server instances"
	@echo "  llama-models        List all available models in both llama-server instances"
	@echo ""
	@echo "  ── Per-service restart ─────────────────────────────────"
	@echo "  restart-bot         Restart discord-bot only"
	@echo "  restart-proxy       Restart proxy only"
	@echo "  restart-openwebui   Restart Open WebUI only"
	@echo "  restart-rag         Restart RAG service only"
	@echo ""
	@echo "  ── Shells ─────────────────────────────────────────────"
	@echo "  shell-bot           Open bash inside discord-bot container"
	@echo "  shell-proxy         Open bash inside proxy container"
	@echo "  shell-rag           Open bash inside rag-service container"
	@echo ""
	@echo "  ── Model Management ───────────────────────────────────"
	@echo "  models-download     Download all GGUF model files from HuggingFace"
	@echo "                        Skips files already present. Run before first 'make up'."
	@echo "                        Usage: make models-download"
	@echo "                               make models-download SLOT=permanent"
	@echo "                               make models-download SLOT=swappable"
	@echo "                               make models-download DRY_RUN=1"
	@echo "                        Set HF_TOKEN env var for private/gated repos."
	@echo ""
	@echo "  To add or change a model:"
	@echo "    1. Edit models.ini (swappable) or docker-compose.yml --model flag (permanent)"
	@echo "    2. Add/update the entry in scripts/download_models.py"
	@echo "    3. Run: make models-download"
	@echo "    4. Run: make restart  (or restart-llama-swappable / restart-llama-permanent)"
	@echo ""
	@echo "  ── DiscordChatExporter ──────────────────────────────────"
	@echo "  dce-help            Show DCE CLI help"
	@echo "  dce-export-full     Export entire guild (first-time setup)"
	@echo "  dce-export-guild    Export guild with date range"
	@echo "                        Usage: make dce-export-guild AFTER=2026-03-29 BEFORE=2026-04-29"
	@echo "  dce-export-channel  Export specific channel"
	@echo "                        Usage: make dce-export-channel CHANNEL_ID=111222..."
	@echo "                        Optional: AFTER=2026-03-29 BEFORE=2026-04-29"
	@echo ""
	@echo "  ── History Service ──────────────────────────────────────"
	@echo "  dce-evaluate      Trigger history-service to evaluate channels and export"
	@echo "  history-refresh   Alias for dce-evaluate"
	@echo ""
	@echo "  ── Destructive ────────────────────────────────────────"
	@echo "  nuke                ⚠️  Stop everything AND remove all volumes"
	@echo "                        Note: GGUF model files in /srv/models are NOT deleted."
	@echo ""

# ── Lifecycle ──────────────────────────────────────────────

## Start all services in detached mode
up:
	docker compose up -d

## Stop and remove containers (named volumes are preserved)
down:
	docker compose down

## Full restart
restart: down up

## Rebuild all custom images (proxy, discord-bot, rag) without starting
build:
	docker compose build

## Pull latest git changes, rebuild changed images, restart affected services
## This is the standard "deploy latest code" command.
pull:
	git pull
	docker compose up -d --build

# ── Logs ───────────────────────────────────────────────────

## Tail logs from all services (Ctrl+C to stop)
logs:
	docker compose logs -f

## Tail proxy logs only
logs-proxy:
	docker compose logs -f proxy

## Tail discord-bot logs only
logs-bot:
	docker compose logs -f discord-bot

## Tail both llama-server instance logs
logs-llama:
	docker compose logs -f llama-permanent llama-swappable

## Tail Open WebUI logs only
logs-openwebui:
	docker compose logs -f open-webui

## Tail RAG service logs only
logs-rag:
	docker compose logs -f rag-service

## Tail history-service logs only (useful for monitoring LoRA training progress)
logs-history:
	docker compose logs -f history-service

# ── Status ─────────────────────────────────────────────────

## Show running containers, ports, and health status
status:
	docker compose ps

## Verify the GPU is visible inside both llama-server containers
check-gpu:
	@echo "=== llama-permanent GPU check ==="
	docker compose exec llama-permanent nvidia-smi
	@echo ""
	@echo "=== llama-swappable GPU check ==="
	docker compose exec llama-swappable nvidia-smi

## Show models currently loaded (in VRAM) in both llama-server instances
## Uses the OpenAI-compatible /v1/models endpoint
llama-ps:
	@echo "=== llama-permanent (:11435) — available models ==="
	curl -s http://localhost:11435/v1/models | python3 -m json.tool 2>/dev/null || echo "(not running)"
	@echo ""
	@echo "=== llama-swappable (:11434) — available models ==="
	curl -s http://localhost:11434/v1/models | python3 -m json.tool 2>/dev/null || echo "(not running)"

## Alias for llama-ps
llama-models: llama-ps

# ── Per-service restart ─────────────────────────────────────

## Restart discord-bot only (useful during bot development)
restart-bot:
	docker compose restart discord-bot

## Restart proxy only
restart-proxy:
	docker compose restart proxy

## Restart Open WebUI only
restart-openwebui:
	docker compose restart open-webui

## Restart RAG service only
restart-rag:
	docker compose restart rag-service

## Restart llama-swappable only (picks up models.ini changes)
restart-llama-swappable:
	docker compose restart llama-swappable

## Restart llama-permanent only
restart-llama-permanent:
	docker compose restart llama-permanent

# ── Shells ─────────────────────────────────────────────────

## Open an interactive bash shell inside the discord-bot container
shell-bot:
	docker compose exec discord-bot bash

## Open an interactive bash shell inside the proxy container
shell-proxy:
	docker compose exec proxy bash

## Open an interactive bash shell inside the rag-service container
shell-rag:
	docker compose exec rag-service bash

# ── Model Management ───────────────────────────────────────

## Download all GGUF model files from HuggingFace.
## Skips files that are already present — safe to re-run.
## Run this before the first `make up`.
## A Python venv is automatically created in scripts/.venv/ on first run.
##
## Options:
##   SLOT=permanent|swappable|all   Download only models for a specific slot (default: all)
##   DRY_RUN=1                      Print what would be downloaded without downloading
##   MODELS_DIR=/path/to/models     Override model storage directory (default: /srv/models)
##
## Set HF_TOKEN environment variable for private or gated HuggingFace repos.
##
## To add or change a model:
##   1. Edit the MODELS list in scripts/download_models.py
##   2. Update models.ini (for swappable slot) or docker-compose.yml (for permanent slot)
##   3. Run: make models-download
##   4. Run: make restart-llama-swappable  (or restart-llama-permanent)
VENV := scripts/.venv
PYTHON := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip

models-download:
	@# Create venv if it doesn't exist
	@if [ ! -d "$(VENV)" ]; then \
		echo "Creating virtual environment..."; \
		python3 -m venv $(VENV); \
	fi
	@# Install requirements inside the venv
	@$(PIP) install -q -r scripts/requirements.txt
	@# Run the script using the venv's python
	$(PYTHON) scripts/download_models.py \
		$(if $(MODELS_DIR),--models-dir $(MODELS_DIR),) \
		$(if $(SLOT),--slot $(SLOT),) \
		$(if $(DRY_RUN),--dry-run,)

# ── Destructive ────────────────────────────────────────────

## ⚠️  DESTRUCTIVE: Stop everything and remove ALL named volumes.
## This wipes Open WebUI data, ChromaDB data, and training state.
## GGUF model files in /srv/models are NOT deleted — they are bind-mounted,
## not stored in named volumes. Re-run `make models-download` is NOT needed.
## You will need to re-ingest lore data afterwards.
nuke:
	@echo "⚠️  WARNING: This will delete ALL named volumes including"
	@echo "   Open WebUI data and ChromaDB lore data."
	@echo "   GGUF model files in /srv/models are preserved."
	@echo "   Press Ctrl+C within 5 seconds to cancel..."
	@sleep 5
	docker compose down -v
	@echo "✓ All containers and volumes removed."

# ── DiscordChatExporter ──────────────────────────────────────
# DCE is a CLI-only tool (not a long-running service).
# Run via `docker compose --profile manual run --rm discord-chat-exporter`.
# Output lands in /mnt/storage_cold/array/DiscordArchive/raw/

## Show DCE CLI help
dce-help:
	docker compose --profile manual run --rm discord-chat-exporter --help

## Export entire guild (first-time full setup)
## Usage: make dce-export-full
dce-export-full:
	@echo "Exporting entire guild $(DISCORD_GUILD_ID)..."
	docker compose --profile manual run --rm discord-chat-exporter \
		exportguild --guild $(DISCORD_GUILD_ID) \
		--format Json --output /out/

## Export guild with date range (incremental pull)
## Usage: make dce-export-guild AFTER=2026-03-29 BEFORE=2026-04-29
dce-export-guild:
	@if [ -z "$(AFTER)" ]; then echo "Error: AFTER is required (e.g. AFTER=2026-03-29)"; exit 1; fi
	@echo "Exporting guild $(DISCORD_GUILD_ID) from $(AFTER)$(if $(BEFORE), to $(BEFORE),)..."
	docker compose --profile manual run --rm discord-chat-exporter \
		exportguild --guild $(DISCORD_GUILD_ID) \
		--format Json --output /out/ \
		--after $(AFTER) $(if $(BEFORE),--before $(BEFORE),)

## Export specific channel
## Usage: make dce-export-channel CHANNEL_ID=111222333444555666
## Optional: AFTER=2026-03-29 BEFORE=2026-04-29
dce-export-channel:
	@if [ -z "$(CHANNEL_ID)" ]; then echo "Error: CHANNEL_ID is required"; exit 1; fi
	@echo "Exporting channel $(CHANNEL_ID)$(if $(AFTER), from $(AFTER),)...)"
	docker compose --profile manual run --rm discord-chat-exporter \
		export --channel $(CHANNEL_ID) \
		--format Json --output /out/ \
		$(if $(AFTER),--after $(AFTER),) $(if $(BEFORE),--before $(BEFORE),)
	
	# ── History Service ──────────────────────────────────────────
	
	## Trigger history-service to evaluate channels and run targeted exports
	## This is the standard pipeline trigger — host cron calls this monthly.
	dce-evaluate:
		@echo "Triggering history-service channel evaluation..."
		curl -s -X POST http://localhost:11437/evaluate | python3 -m json.tool 2>/dev/null || echo "(history-service not running)"
	
	## Alias for dce-evaluate — single-command pipeline trigger
	history-refresh: dce-evaluate
	
