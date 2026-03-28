# Model Configuration

Model definitions for all models in the stack have moved to **`models.ini`** in the repo root.

## Overview

The inference backend is **llama.cpp** (`llama-server`), which uses a preset config file (`models.ini`) to define all available models for the swappable slot. The permanent slot (autocomplete) is configured directly in `docker-compose.yml`.

| Config location | Slot | Purpose |
|---|---|---|
| `docker-compose.yml` (`--model` flag) | permanent (`:11435`) | Autocomplete model — loaded at startup, never evicted |
| `models.ini` | swappable (`:11434`) | All other models — loaded on demand by the router |

## models.ini Format

Each `[section]` in `models.ini` defines one named model:

```ini
[brain]
model           = /models/unsloth/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-UD-IQ4_NL.gguf
alias           = brain
n_gpu_layers    = -1
n_ctx        = 40960
n_predict       = -1
temperature     = 0.2
top_k           = 10
top_p           = 0.9
reasoning_format = none
```

The `alias` field is the model name used in API requests (`"model": "brain"`). The `model` field is the path to the GGUF file inside the container (mounted from `MODELS_DIR` on the host).

## Model Files

GGUF files live on the host at:
```
./models/<publisher>/<model-name>/filename.gguf
```

Download all required GGUFs with:
```bash
make models-download
```

This runs `scripts/download_models.py`, which fetches each GGUF from HuggingFace and saves it to the correct path. Already-downloaded files are skipped.

## Current Models

| Alias | GGUF | Slot | VRAM |
|---|---|---|---|
| `autocomplete` | `unsloth/Qwen3.5-2B-GGUF` IQ4_NL | permanent | ~1.21 GB |
| `brain` | `unsloth/Qwen3.5-35B-A3B-GGUF` UD-IQ4_NL | swappable | ~17.8 GB |
| `mimic_user1` … `mimic_user6` | `HauhauCS/Qwen3.5-35B-A3B-Uncensored` IQ4_XS | swappable | ~18 GB (shared GGUF) |
| `lore` | `bartowski/gemma-3-12b-it-GGUF` Q6_K | swappable | ~9.5 GB |
| `librechat_chat` | `unsloth/Qwen3.5-14B-GGUF` UD-IQ4_XS | swappable | ~9.5 GB |
| `image-caption` | `HauhauCS/Qwen3.5-35B-A3B-Uncensored` IQ4_XS | swappable | ~18 GB (shared GGUF) |

## Adding a Mimic Persona

Mimic personas share the same GGUF — only the alias and system prompt differ. To add a new persona:

1. Add a new section to `models.ini`:
   ```ini
   [mimic_alice]
   model           = /models/HauhauCS/Qwen3.5-35B-A3B-Uncensored/Qwen3.5-35B-A3B-Uncensored-IQ4_XS.gguf
   alias           = mimic_alice
   n_gpu_layers    = -1
   n_ctx        = 8192
   n_predict       = 512
   temperature     = 0.85
   top_k           = 40
   top_p           = 0.9
   repeat_penalty  = 1.3
   reasoning_format = none
   ```

2. Add `mimic_alice` to `SWAPPABLE_MODELS` in `proxy/config.py`.

3. Add the persona to `MENTION_TO_MODEL` in the Discord bot's `router.py`.

4. Restart the swappable server to pick up the new preset:
   ```bash
   make restart-llama-swappable
   ```

No model download needed — the GGUF is already present from the mimic base.

## Changing a Model or Quant

1. Edit `models.ini` — update the `model` path and/or parameters for the relevant section.
2. Update `scripts/download_models.py` — add or update the entry with the new `repo_id` and `filename`.
3. Download the new GGUF:
   ```bash
   make models-download
   ```
4. Restart the relevant server:
   ```bash
   make restart-llama-swappable   # for swappable slot models
   make restart-llama-permanent   # for the autocomplete model
   ```

The proxy references models by alias only — no proxy changes needed when swapping underlying GGUFs.

## Switching the Permanent (Autocomplete) Model

The permanent slot model is configured in `docker-compose.yml` via the `--model` flag on the `llama-permanent` service. To change it:

1. Update `scripts/download_models.py` with the new model entry (slot: `"permanent"`).
2. Run `make models-download`.
3. Edit the `--model` flag in `docker-compose.yml` to point to the new GGUF path.
4. Run `make restart-llama-permanent`.

## Legacy Modelfiles

The `.Modelfile` files in this directory are **no longer used**. They are retained for historical reference only. The `models.ini` preset config in the repo root is the current source of truth for all model configuration.
