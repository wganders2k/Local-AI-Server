# ============================================================
# Local AI Server — Makefile
# ============================================================
# Common operations for managing the Local AI Server stack.
#
# Usage:  make <target>
#
# All targets run from the repo root alongside docker-compose.yml.
# Requires: docker, docker compose (v2), git, python3
# ============================================================

.PHONY: help up down restart build pull \
        logs logs-proxy logs-bot logs-llama logs-librechat logs-rag logs-history \
        status \
        restart-bot restart-proxy restart-librechat restart-rag \
        shell-bot shell-proxy shell-rag \
        llama-ps llama-models \
        check-gpu \
        models-download \
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
	@echo "  logs-librechat      Tail LibreChat logs only"
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
	@echo "  restart-librechat   Restart LibreChat only"
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
	@echo "  ── Destructive ────────────────────────────────────────"
	@echo "  nuke                ⚠️  Stop everything AND remove all volumes"
	@echo "                        Note: GGUF model files in ./models are NOT deleted."
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

## Tail LibreChat logs only
logs-librechat:
	docker compose logs -f librechat

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

## Restart LibreChat only
restart-librechat:
	docker compose restart librechat

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
##
## Options:
##   SLOT=permanent|swappable|all   Download only models for a specific slot (default: all)
##   DRY_RUN=1                      Print what would be downloaded without downloading
##   MODELS_DIR=/path/to/models     Override model storage directory (default: ./models)
##
## Set HF_TOKEN environment variable for private or gated HuggingFace repos.
##
## To add or change a model:
##   1. Edit the MODELS list in scripts/download_models.py
##   2. Update models.ini (for swappable slot) or docker-compose.yml (for permanent slot)
##   3. Run: make models-download
##   4. Run: make restart-llama-swappable  (or restart-llama-permanent)
models-download:
	@python3 -c "import huggingface_hub" 2>/dev/null || pip3 install -q -r scripts/requirements.txt
	python3 scripts/download_models.py \
		$(if $(MODELS_DIR),--models-dir $(MODELS_DIR),) \
		$(if $(SLOT),--slot $(SLOT),) \
		$(if $(DRY_RUN),--dry-run,)

# ── Destructive ────────────────────────────────────────────

## ⚠️  DESTRUCTIVE: Stop everything and remove ALL named volumes.
## This wipes LibreChat history, ChromaDB data, and training state.
## GGUF model files in ./models are NOT deleted — they are bind-mounted,
## not stored in named volumes. Re-run `make models-download` is NOT needed.
## You will need to re-ingest lore data afterwards.
nuke:
	@echo "⚠️  WARNING: This will delete ALL named volumes including"
	@echo "   LibreChat conversation history and ChromaDB lore data."
	@echo "   GGUF model files in ./models are preserved."
	@echo "   Press Ctrl+C within 5 seconds to cancel..."
	@sleep 5
	docker compose down -v
	@echo "✓ All containers and volumes removed."
