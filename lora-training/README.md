# LoRA Training Pipeline

Phase 3 component. Unsloth QLoRA fine-tuning on `Qwen3.5-35B-A3B-Uncensored` per server member, followed by GGUF merge and `models.ini` update. Scripts can be run manually or invoked automatically by the `history-service` training trigger.

## Responsibilities

- Fine-tune per-member LoRA adapters on `Qwen3.5-35B-A3B-Uncensored` base using Unsloth QLoRA
- Merge adapters into full GGUF models: `mimic_<member>_v{n}.gguf`
- Update `models.ini` to point to the new GGUF — zero proxy or bot code changes required

## Design Reference

See `Design.md` §9a (History & Training Pipeline) and §10 Phase 3 (LoRA Persona Refinement).

## Prerequisites

- Per-user JSONL message history collected by `history-service` (minimum ~200 clean messages to trigger; more is better)
- Unsloth installed (supports Qwen3.5 fine-tuning natively as of March 2026)
- RTX 3090 with sufficient VRAM headroom (~14–16 GB required for QLoRA training)
- Swappable llama-server slot must be idle before training begins (history-service handles this automatically)

## Invocation

### Automatic (Phase 3 — via history-service)

When `TRAINING_TRIGGER_ENABLED=true` in the history-service, training is dispatched automatically when a user accumulates ≥ `RETRAIN_THRESHOLD` new clean messages. The history-service calls these scripts as subprocesses:

```bash
# Called by history-service/training_trigger.py
python train.py --user <user_id> --data /app/data/history/<user_id>.jsonl --output /app/lora-outputs/<user_id>/
python merge.py --user <user_id> --checkpoint /app/lora-outputs/<user_id>/checkpoint/ --output /app/lora-outputs/<user_id>/
```

### Manual (Phase 3 — standalone)

```bash
# Fine-tune LoRA adapter for a specific member
python train.py --user user3 --data data/user3_clean.jsonl --output outputs/user3/

# Merge adapter into full GGUF
python merge.py --user user3 --checkpoint outputs/user3/checkpoint/ --output outputs/user3/

# Update models.ini to point to the new GGUF, then restart the swappable server
make restart-llama-swappable
```

## Upgrade Path (zero bot or proxy changes)

1. Training completes → `mimic_<member>_v{n+1}.gguf` saved to outputs directory
2. `merge.py` updates the `model` path in `models.ini` for the relevant `[mimic_<member>]` section
3. `make restart-llama-swappable` — llama-server picks up the new GGUF path from `models.ini`
4. Bot continues using the same model alias — no proxy changes, no bot code changes
5. `history-service` updates `training_state.json`: resets `messages_since_last_train = 0`, increments `model_version`

## Training Notes

**Why train on all clean history (not just recent messages)?**
Using the full clean message history maximises style capture. QLoRA only trains the adapter (not the base weights), so the training corpus can be large without catastrophic forgetting risk. Aggressive filtering (see `history-service` README) ensures only quality messages reach the training dataset.

**Epoch count:** 1–2 epochs recommended. More epochs risk overfitting to specific phrases rather than capturing general style.

**Base model:** Always train from the fresh `Qwen3.5-35B-A3B-Uncensored` checkpoint. Never train on top of a previously merged model — this compounds drift across versions.

**VRAM during training:** QLoRA on the 35B-A3B model requires ~14–16 GB VRAM. The history-service stops the swappable llama-server instance before starting training to free VRAM.

## Planned File Structure

```
lora-training/
├── train.py            # Unsloth QLoRA fine-tuning script
│                       # Args: --user, --data <jsonl_path>, --output <dir>
├── merge.py            # Adapter merge + GGUF export script
│                       # Args: --user, --checkpoint <dir>, --output <dir>
│                       # Also updates models.ini with the new GGUF path
├── data/               # Per-member message datasets (gitignored if large)
├── outputs/            # Merged GGUFs (gitignored — large files)
└── checkpoints/        # Training checkpoints (gitignored — large files)
```

> `outputs/` and `checkpoints/` are excluded from git via `.gitignore`.
> Store trained models on the server filesystem, not in the repo.
> When invoked by history-service, outputs go to the `lora_outputs` Docker volume.

## Relationship to `history-service`

The `history-service` is the automated orchestration layer that:
1. Collects and filters Discord messages into per-user JSONL files
2. Tracks clean message counts and triggers training when the threshold is met
3. Calls `train.py` and `merge.py` as subprocesses
4. Updates `models.ini` and restarts `llama-swappable` after merge completes
5. Updates training state (version, timestamp, status)

These scripts remain independently runnable for manual workflows. The history-service adds the automated scheduling layer on top.
