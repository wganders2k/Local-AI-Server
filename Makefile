# ============================================================
# Local AI Server — Makefile
# ============================================================
# Common operations for managing the Local AI Server stack.
#
# Usage:  make <target>
#
# All targets run from the repo root alongside docker-compose.yml.
# Requires: docker, docker compose (v2), git
# ============================================================

.PHONY: help up down restart build pull \
        logs logs-proxy logs-bot logs-ollama logs-librechat logs-rag \
        status \
        restart-bot restart-proxy restart-librechat restart-rag \
        shell-bot shell-proxy shell-rag \
        ollama-ps ollama-list \
        check-gpu \
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
	@echo "  logs-ollama         Tail both Ollama instance logs"
	@echo "  logs-librechat      Tail LibreChat logs only"
	@echo "  logs-rag            Tail RAG service logs only"
	@echo ""
	@echo "  ── Status ─────────────────────────────────────────────"
	@echo "  status              Show running containers and health"
	@echo "  check-gpu           Verify GPU is visible inside Ollama containers"
	@echo "  ollama-ps           Show models currently loaded in both Ollama instances"
	@echo "  ollama-list         List all registered models in both Ollama instances"
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
	@echo "  ── Destructive ────────────────────────────────────────"
	@echo "  nuke                ⚠️  Stop everything AND remove all volumes"
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

## Tail both Ollama instance logs
logs-ollama:
	docker compose logs -f ollama-permanent ollama-swappable

## Tail LibreChat logs only
logs-librechat:
	docker compose logs -f librechat

## Tail RAG service logs only
logs-rag:
	docker compose logs -f rag-service

# ── Status ─────────────────────────────────────────────────

## Show running containers, ports, and health status
status:
	docker compose ps

## Verify the GPU is visible inside both Ollama containers
check-gpu:
	@echo "=== ollama-permanent GPU check ==="
	docker compose exec ollama-permanent nvidia-smi
	@echo ""
	@echo "=== ollama-swappable GPU check ==="
	docker compose exec ollama-swappable nvidia-smi

## Show models currently loaded (in VRAM) in both Ollama instances
ollama-ps:
	@echo "=== ollama-permanent (:11435) — loaded models ==="
	curl -s http://localhost:11435/api/ps | python3 -m json.tool 2>/dev/null || echo "(not running)"
	@echo ""
	@echo "=== ollama-swappable (:11434) — loaded models ==="
	curl -s http://localhost:11434/api/ps | python3 -m json.tool 2>/dev/null || echo "(not running)"

## List all registered models in both Ollama instances
ollama-list:
	@echo "=== ollama-permanent (:11435) — registered models ==="
	curl -s http://localhost:11435/api/tags | python3 -m json.tool 2>/dev/null || echo "(not running)"
	@echo ""
	@echo "=== ollama-swappable (:11434) — registered models ==="
	curl -s http://localhost:11434/api/tags | python3 -m json.tool 2>/dev/null || echo "(not running)"

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

# ── Destructive ────────────────────────────────────────────

## ⚠️  DESTRUCTIVE: Stop everything and remove ALL named volumes.
## This wipes Ollama model weights, LibreChat history, ChromaDB data.
## You will need to re-pull models and re-ingest lore data afterwards.
nuke:
	@echo "⚠️  WARNING: This will delete ALL volumes including Ollama model weights,"
	@echo "   LibreChat conversation history, and ChromaDB lore data."
	@echo "   Press Ctrl+C within 5 seconds to cancel..."
	@sleep 5
	docker compose down -v
	@echo "✓ All containers and volumes removed."
