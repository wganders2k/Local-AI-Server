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
        logs logs-proxy logs-bot logs-ollama logs-librechat logs-rag logs-history \
        status \
        restart-bot restart-proxy restart-librechat restart-rag \
        shell-bot shell-proxy shell-rag \
        ollama-ps ollama-list \
        check-gpu \
        models-init model-create model-remove model-redownload \
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
	@echo "  logs-history        Tail history-service logs only"
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
	@echo "  ── Model Management ───────────────────────────────────"
	@echo "  models-init         Register all models (first-time setup or post-nuke)"
	@echo "  model-create        Register/re-register a model from its Modelfile"
	@echo "                        Usage: make model-create MODEL=<name> SLOT=<permanent|swappable>"
	@echo "                        If MODEL=mimic, automatically triggers full LoRA retraining via history-service."
	@echo "  model-remove        Remove a registered model (GGUF blob stays cached)"
	@echo "                        Usage: make model-remove MODEL=<name> SLOT=<permanent|swappable>"
	@echo "  model-redownload    Force full re-fetch: remove + re-create from Modelfile"
	@echo "                        Usage: make model-redownload MODEL=<name> SLOT=<permanent|swappable>"
	@echo "                        Edit the FROM line in modelfiles/<name>.Modelfile first to switch models."
	@echo "                        If MODEL=mimic, automatically triggers full LoRA retraining via history-service."
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

## Tail history-service logs only (useful for monitoring LoRA training progress)
logs-history:
	docker compose logs -f history-service

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

# ── Model Management ───────────────────────────────────────

## Register all models from scratch.
## Use this after first `make up` or after `make nuke` to restore all models.
## Models are pulled from HuggingFace or the Ollama registry as defined in each Modelfile.
## Note: This registers the mimic base template only. Per-user mimic personas (mimic_<member>)
## must be created individually.
## After registering mimic, a full LoRA retraining cycle is automatically queued for all users
## via history-service. On a fresh stack this is a no-op; on a partial reset it ensures any
## existing per-user adapters are rebuilt against the current base.
models-init:
	@echo "=== Registering permanent models (ollama-permanent :11435) ==="
	docker compose exec ollama-permanent ollama create autocomplete -f /modelfiles/autocomplete.Modelfile
	@echo ""
	@echo "=== Registering swappable models (ollama-swappable :11434) ==="
	docker compose exec ollama-swappable ollama create brain -f /modelfiles/brain.Modelfile
	docker compose exec ollama-swappable ollama create mimic -f /modelfiles/mimic.Modelfile
	docker compose exec ollama-swappable ollama create lore -f /modelfiles/lore.Modelfile
	docker compose exec ollama-swappable ollama create librechat_chat -f /modelfiles/librechat_chat.Modelfile
	@echo ""
	@echo "✓ Core models registered. Run 'make ollama-list' to verify."
	@echo ""
	@echo "=== Queuing full LoRA retraining cycle via history-service ==="
	docker compose exec history-service python training_trigger.py --force-all
	@echo "✓ Retraining queued for all users. Training will begin during the next training window."
	@echo "  Monitor progress: make logs-history"
	@echo ""
	@echo "  Note: Mimic personas (mimic_<member>) must be created individually."
	@echo "  Copy modelfiles/mimic.Modelfile, fill in the member name, then:"
	@echo "  make model-create MODEL=mimic_<member> SLOT=swappable"

## Register or re-register a single model from its Modelfile.
## Uses the cached GGUF blob if already downloaded — no re-fetch.
## Usage: make model-create MODEL=librechat_chat SLOT=swappable
##        make model-create MODEL=autocomplete SLOT=permanent
##
## If MODEL=mimic (the base template), this automatically queues a full LoRA
## retraining cycle for all users via history-service — no separate step needed.
model-create:
	@test -n "$(MODEL)" || (echo "Error: MODEL is required. Usage: make model-create MODEL=<name> SLOT=<permanent|swappable>"; exit 1)
	@test -n "$(SLOT)" || (echo "Error: SLOT is required. Usage: make model-create MODEL=<name> SLOT=<permanent|swappable>"; exit 1)
	@test -f modelfiles/$(MODEL).Modelfile || (echo "Error: modelfiles/$(MODEL).Modelfile not found."; exit 1)
	@echo "=== Creating model '$(MODEL)' on ollama-$(SLOT) ==="
	docker compose exec ollama-$(SLOT) ollama create $(MODEL) -f /modelfiles/$(MODEL).Modelfile
	@echo "✓ Model '$(MODEL)' registered on ollama-$(SLOT)."
	@if [ "$(MODEL)" = "mimic" ]; then \
		echo ""; \
		echo "=== Queuing full LoRA retraining cycle via history-service ==="; \
		docker compose exec history-service python training_trigger.py --force-all; \
		echo "✓ Retraining queued for all users. Training will begin during the next training window."; \
		echo "  Monitor progress: make logs-history"; \
	fi

## Remove a registered model from an Ollama instance.
## The underlying GGUF blob stays cached in the volume — no disk space freed.
## Usage: make model-remove MODEL=librechat_chat SLOT=swappable
model-remove:
	@test -n "$(MODEL)" || (echo "Error: MODEL is required. Usage: make model-remove MODEL=<name> SLOT=<permanent|swappable>"; exit 1)
	@test -n "$(SLOT)" || (echo "Error: SLOT is required. Usage: make model-remove MODEL=<name> SLOT=<permanent|swappable>"; exit 1)
	@echo "=== Removing model '$(MODEL)' from ollama-$(SLOT) ==="
	docker compose exec ollama-$(SLOT) ollama rm $(MODEL)
	@echo "✓ Model '$(MODEL)' removed from ollama-$(SLOT). GGUF blob remains cached."

## Force a full re-download of a model's GGUF weights and re-register it.
## Use this to:
##   - Switch to a different model/quant (edit the FROM line in the Modelfile first)
##   - Force a clean re-fetch if the cached blob is corrupt or incomplete
##   - Test a new model without nuking the entire stack
## Usage: make model-redownload MODEL=librechat_chat SLOT=swappable
##
## If MODEL=mimic (the base template), this automatically queues a full LoRA
## retraining cycle for all users via history-service — no separate step needed.
model-redownload:
	@test -n "$(MODEL)" || (echo "Error: MODEL is required. Usage: make model-redownload MODEL=<name> SLOT=<permanent|swappable>"; exit 1)
	@test -n "$(SLOT)" || (echo "Error: SLOT is required. Usage: make model-redownload MODEL=<name> SLOT=<permanent|swappable>"; exit 1)
	@test -f modelfiles/$(MODEL).Modelfile || (echo "Error: modelfiles/$(MODEL).Modelfile not found."; exit 1)
	@echo "=== Force re-downloading model '$(MODEL)' on ollama-$(SLOT) ==="
	@echo "    FROM source: $$(grep '^FROM' modelfiles/$(MODEL).Modelfile)"
	-docker compose exec ollama-$(SLOT) ollama rm $(MODEL) 2>/dev/null
	docker compose exec ollama-$(SLOT) ollama create --no-cache $(MODEL) -f /modelfiles/$(MODEL).Modelfile
	@echo "✓ Model '$(MODEL)' re-downloaded and registered on ollama-$(SLOT)."
	@if [ "$(MODEL)" = "mimic" ]; then \
		echo ""; \
		echo "=== Queuing full LoRA retraining cycle via history-service ==="; \
		docker compose exec history-service python training_trigger.py --force-all; \
		echo "✓ Retraining queued for all users. Training will begin during the next training window."; \
		echo "  Monitor progress: make logs-history"; \
	fi

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
