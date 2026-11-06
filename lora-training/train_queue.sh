#!/usr/bin/env bash
# Train personas one after another. Started by `make train-queue`, which runs
# this as the systemd user unit `train-queue` so it outlives the shell (and any
# Claude session) that launched it. Never run two at once: there is one
# lora-trainer container.
#
#   train_queue.sh <job-id> [persona ...]     (no personas = whole roster)
#
# Each run trains into lora-outputs/<persona>-<job-id>, a directory of its own:
# the worker resumes from whatever it finds in its output dir, so reusing a
# finished run's dir would "resume" at the last step and train nothing.
# Re-running with the same job id therefore picks up where it stopped: finished
# personas are skipped and a half-done one resumes from its last checkpoint.
#
# Adapters are NOT exported. Check a run (eval loss in the log, `make
# train-sample`), then `make train-export-latest` to put it live on /mimic.
#
# Stops at the first failed run.
set -u

JOB="${1:?usage: train_queue.sh <job-id> [persona ...]}"
shift
REPO=/home/peacow/local-ai-server
OUTPUTS=/home/peacow/lora-outputs
TRAINING=/mnt/storage_cold/array/DiscordArchive/training
ROSTER="$TRAINING/personas.json"
# lora-outputs is root-owned (the container writes it), so the log lives here.
LOGDIR=/home/peacow/.local/share/train-queue
LOG="$LOGDIR/queue-$JOB.log"
mkdir -p "$LOGDIR"

log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

cd "$REPO" || exit 1

# Default: every roster persona. Either way, largest training set first, so the
# long runs happen while nobody is waiting on the result.
if [ $# -gt 0 ]; then QUEUE="$*"; else
  QUEUE=$(python3 -c 'import json,sys; print(" ".join(k for k in json.load(open(sys.argv[1])) if not k.startswith("_")))' "$ROSTER") || exit 1
fi
QUEUE=$(for p in $QUEUE; do
  n=$({ wc -l < "$TRAINING/$p/train.jsonl"; } 2>/dev/null || echo 0)
  echo "$n $p"
done | sort -rn | awk '{print $2}' | tr '\n' ' ')

for p in $QUEUE; do
  [ -f "lora-training/configs/$p.yaml" ] || { log "FAIL: no lora-training/configs/$p.yaml"; exit 1; }
  [ -s "$TRAINING/$p/train.jsonl" ] || { log "FAIL: no dataset for $p (make train-datasets PERSONA=$p)"; exit 1; }
done

log "QUEUE $JOB: $QUEUE"
for p in $QUEUE; do
  RUN="$OUTPUTS/$p-$JOB"
  if [ -f "$RUN/final/COMPLETE" ]; then log "SKIP $p (already finished in $RUN)"; continue; fi

  log "SUBMIT $p -> $RUN"
  if ! make train-submit CONFIG="configs/$p.yaml" OUTPUT_DIR="$RUN" >>"$LOG" 2>&1; then
    log "FAIL $p: submit failed"; exit 1
  fi
  sleep 20
  while [ "$(docker inspect -f '{{.State.Running}}' lora-trainer 2>/dev/null)" = "true" ]; do sleep 60; done

  code=$(docker inspect -f '{{.State.ExitCode}}' lora-trainer 2>/dev/null)
  if [ "$code" != "0" ] || [ ! -f "$RUN/final/COMPLETE" ]; then
    log "FAIL $p: exit=$code"
    docker logs lora-trainer 2>&1 | grep -v reclaim-notice | tail -30 >>"$LOG"
    exit 1
  fi
  loss=$(docker logs lora-trainer 2>&1 | sed -n "s/.*'eval_loss': '\([0-9.]*\)'.*/\1/p" | tail -1)
  log "DONE $p eval_loss=${loss:-?}"
done
log "QUEUE COMPLETE — nothing exported; run 'make train-export-latest' when happy"
