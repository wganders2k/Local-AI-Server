# Modelfiles

Ollama Modelfile definitions for all models in the stack. These files are the source-of-truth for model configuration — parameters, system prompts, and the GGUF source (`FROM` line).

## Files

| File | Ollama Name | Slot | Source |
|---|---|---|---|
| `autocomplete.Modelfile` | `autocomplete` | permanent (`:11435`) | Ollama registry |
| `brain.Modelfile` | `brain` | swappable (`:11434`) | HuggingFace |
| `mimic.Modelfile` | `mimic_<member>` | swappable (`:11434`) | HuggingFace |
| `lore.Modelfile` | `lore` | swappable (`:11434`) | Ollama registry |
| `librechat_chat.Modelfile` | `librechat_chat` | swappable (`:11434`) | HuggingFace |
| `image-caption.Modelfile` | `image-caption` | swappable (`:11434`) | HuggingFace |

## Mimic Personas

`mimic.Modelfile` is a **template**. For each Discord member you want to mimic:

1. Copy the template: `cp modelfiles/mimic.Modelfile modelfiles/mimic_<member>.Modelfile`
2. Replace `<member>` in the filename and inside the `SYSTEM` block with the actual member name
3. Register it: `make model-create MODEL=mimic_<member> SLOT=swappable`

## Model Management

See `Makefile` targets — run `make help` for the full list. Key targets:

```bash
make models-init                                    # Register all models (first-time setup or post-nuke)
make model-create MODEL=librechat_chat SLOT=swappable   # Re-register from Modelfile (uses cached GGUF)
make model-redownload MODEL=librechat_chat SLOT=swappable  # Force full re-fetch of GGUF weights
make model-remove MODEL=brain SLOT=swappable        # Remove a registered model
```

## Switching Models

To try a different model or quant for any slot:

1. Edit the `FROM` line in the relevant Modelfile (e.g. change quant tag or HuggingFace repo)
2. Run `make model-redownload MODEL=<name> SLOT=<permanent|swappable>`

Ollama will fetch the new GGUF from the updated source. The proxy references models by name only — no proxy changes needed.

## How Ollama Resolves the FROM Line

- `FROM qwen2.5-coder:1.5b` → pulls from the Ollama model registry
- `FROM hf.co/owner/repo:tag` → pulls the GGUF directly from HuggingFace Hub
- `FROM /path/to/local.gguf` → loads from a local file path (useful for LoRA-merged models in Phase 3)

The GGUF blob is cached in the Ollama volume (`/root/.ollama/models/blobs/`). `model-create` reuses the cache; `model-redownload` forces a fresh fetch.
