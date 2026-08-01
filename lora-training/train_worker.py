"""
The GPU half of the trainer. Short-lived and disposable.

It trains, and it knows nothing else. Asking for the GPU, waiting to be granted
it, and hearing that it is wanted back all belong to supervisor.py, which is PID
1 in this container and never touches the card itself.

That split is the whole reason a preempted run can resume. The training process
has to *die* for its CUDA context to be freed, and Docker suppresses a
container's restart policy for any API-initiated stop or kill — so if this were
PID 1, a preemption would end the run permanently. Measured: `restart: always`
and `on-failure` both leave RestartCount=0 after `docker kill`.

This process is expected to be SIGKILLed at any moment: that is how VRAM is
returned to whatever outranks it, because a CUDA context is only really freed
when the process holding it exits. Nothing here traps signals or tries to shut
down cleanly — attempting a graceful save on the way out would spend the seconds
that the whole design exists to protect.

Everything that must survive a kill therefore has to be on disk *before* the kill,
which is what the periodic checkpoint is for, and has to be recognisable as
complete afterwards, which is what checkpoints.mark_complete is for.

The exit code is what the supervisor reads, and each value means one thing:

    0      training finished
    -9     the supervisor killed us for a preemption; nothing is wrong
    other  something genuinely broke

Run directly for a foreground training run, taking the card without asking:

    python train_worker.py --config configs/smoke.yaml
"""

import argparse
import logging
import os
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import checkpoints

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("lora-worker")

EXIT_DONE = 0
EXIT_ERROR = 1


class PeriodicSaveCallback:
    """
    Save on a wall-clock interval, and mark each save complete once it lands.

    Time-based rather than step-based because the quantity to bound is *lost
    work*, and a step is not a fixed amount of time — the same `save_steps` that
    means four minutes on a 0.6B means over an hour on a 35B. With a wall-clock
    interval the worst case is the same number regardless of the model.

    The COMPLETE marker is written from `on_save`, which the Trainer fires only
    after `_save_checkpoint` has returned. Marking any earlier would defeat the
    point: a checkpoint interrupted by a kill would look resumable.
    """

    def __init__(self, interval_minutes: float, keep: int = 2):
        self.interval = interval_minutes * 60.0
        self.keep = keep
        self._last_save = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        if time.monotonic() - self._last_save >= self.interval:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(path):
            checkpoints.mark_complete(path)
        # Prune here rather than leaving it to save_total_limit, which counts
        # directories without knowing which of them are usable.
        checkpoints.prune(args.output_dir, keep=self.keep)
        self._last_save = time.monotonic()
        return control


def _build_callback_class():
    """
    Subclass TrainerCallback at call time.

    Imported lazily so that this module — and its config loader — can be
    exercised without torch installed.
    """
    from transformers import TrainerCallback

    return type("PeriodicSave", (PeriodicSaveCallback, TrainerCallback), {})


def load_config(path: str) -> dict:
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    for required in ("model", "dataset", "output_dir"):
        if required not in cfg:
            raise ValueError(f"{path}: missing required key '{required}'")
    return cfg


def build_dataset(cfg: dict):
    from datasets import load_dataset

    spec = cfg["dataset"]
    if isinstance(spec, str):
        spec = {"path": spec}

    if "path" in spec:
        path = spec["path"]
        if not os.path.isabs(path):
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
        ds = load_dataset("json", data_files=path, split="train")
    elif "hub" in spec:
        ds = load_dataset(spec["hub"], split=spec.get("split", "train"))
    else:
        raise ValueError("dataset needs either 'path' (JSONL) or 'hub' (HF dataset id)")

    limit = spec.get("limit")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    logger.info(f"Dataset: {len(ds)} examples")
    return ds


def build_model(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = cfg["model"]
    load_in_4bit = cfg.get("load_in_4bit", False)

    quant_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    logger.info(f"Loading {name} (4bit={load_in_4bit}) — this is the slow part")
    t0 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Model loaded in {time.monotonic() - t0:.0f}s")

    if load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.get("gradient_checkpointing", True)
        )
    return model, tokenizer


def main() -> int:
    ap = argparse.ArgumentParser(description="Preemptible LoRA training worker")
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Override output_dir from the config (checkpoints live here)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    output_dir = args.output_dir or cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Anything without a COMPLETE marker is debris from a kill. Clear it before
    # the Trainer sees it, since its own checkpoint discovery has no idea which
    # directories finished writing.
    checkpoints.prune(output_dir, keep=cfg.get("keep_checkpoints", 2))
    resume_from = checkpoints.latest(output_dir)
    if resume_from is None:
        logger.info("No complete checkpoint — starting from scratch")

    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    dataset = build_dataset(cfg)
    model, tokenizer = build_model(cfg)

    lora = cfg.get("lora", {})
    peft_config = LoraConfig(
        r=lora.get("r", 16),
        lora_alpha=lora.get("alpha", 32),
        lora_dropout=lora.get("dropout", 0.0),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora.get(
            "target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )

    train = cfg.get("training", {})
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train.get("epochs", 1),
        per_device_train_batch_size=train.get("batch_size", 1),
        gradient_accumulation_steps=train.get("grad_accum", 8),
        learning_rate=train.get("learning_rate", 2e-4),
        lr_scheduler_type=train.get("lr_scheduler", "cosine"),
        warmup_ratio=train.get("warmup_ratio", 0.03),
        logging_steps=train.get("logging_steps", 5),
        max_length=train.get("max_seq_length", 2048),
        bf16=True,
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        optim=train.get("optim", "paged_adamw_8bit" if cfg.get("load_in_4bit") else "adamw_torch"),
        report_to=[],
        # The wall-clock callback owns when to save; `save_steps` is set beyond
        # any reachable step so the step-based cadence never fires on its own.
        # It cannot be "no" — that makes the Trainer ignore `should_save`
        # entirely, and the callback would then never write anything.
        save_strategy="steps",
        save_steps=train.get("save_steps_ceiling", 10**9),
        # Pruning is ours, in checkpoints.prune, because save_total_limit counts
        # directories without knowing which of them finished writing.
        save_total_limit=None,
    )

    PeriodicSave = _build_callback_class()
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[
            PeriodicSave(
                interval_minutes=cfg.get("checkpoint_minutes", 20),
                keep=cfg.get("keep_checkpoints", 2),
            )
        ],
    )

    logger.info(
        f"Training starts (resume={resume_from or 'scratch'}, "
        f"checkpoint every {cfg.get('checkpoint_minutes', 20)} min). "
        f"This process may be killed at any moment; that is expected."
    )
    trainer.train(resume_from_checkpoint=resume_from)

    # A final save, marked like any other, so a completed run leaves a usable
    # adapter rather than only whatever the last interval happened to catch.
    trainer.save_model(os.path.join(output_dir, "final"))
    checkpoints.mark_complete(os.path.join(output_dir, "final"))

    logger.info(f"Training complete — adapter at {output_dir}/final")
    return EXIT_DONE


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(EXIT_ERROR)
