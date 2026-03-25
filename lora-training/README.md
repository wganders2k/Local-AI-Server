# LoRA Training Pipeline

Phase 3 component. Unsloth QLoRA fine-tuning on `Qwen3.5-9B-Uncensored` per server member, followed by GGUF merge and re-registration in Ollama. Not a running service — scripts are run manually on the server when training new persona models.

## Responsibilities

- Fine-tune per-member LoRA adapters on `Qwen3.5-9B-Uncensored` base using Unsloth QLoRA
- Merge adapters into full GGUF models: `deepleffen_<member>_v2.gguf`
- Re-register updated Modelfiles in Ollama — zero proxy or bot code changes required

## Design Reference

See `Design.md` §10 Phase 3 (LoRA Persona Refinement).

## Prerequisites

- 500–1000 messages per member extracted from Discord history
- Unsloth installed (supports Qwen3.5 fine-tuning natively as of March 2026)
- RTX 3090 with sufficient VRAM headroom (training uses the same GPU)

## Upgrade path (zero bot changes)

1. Train LoRA adapter on Qwen3.5-9B-Uncensored base
2. Merge: `deepleffen_<member>_v2.gguf`
3. Update Modelfile `FROM` line to point to merged GGUF
4. `ollama rm deepleffen_<member>` + `ollama create deepleffen_<member> -f deepleffen_<member>_v2.Modelfile`
5. Bot continues using the same model name — no proxy changes, no bot code changes

## Planned file structure

```
lora-training/
├── train.py            # Unsloth QLoRA fine-tuning script
├── merge.py            # Adapter merge + GGUF export script
├── data/               # Per-member message datasets (gitignored if large)
├── outputs/            # Merged GGUFs (gitignored — large files)
└── checkpoints/        # Training checkpoints (gitignored — large files)
```

> `outputs/` and `checkpoints/` are excluded from git via `.gitignore`.
> Store trained models on the server filesystem, not in the repo.
