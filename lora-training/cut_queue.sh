#!/usr/bin/env bash
# Cut every persona's dataset with the judge, train on the cut data, then judge
# each new adapter against the live one. Started by `make train-cut-queue`,
# which runs this as the systemd user unit `train-queue`, so it outlives the
# shell (and any Claude session) that launched it.
#
#   cut_queue.sh <job-id> [persona ...]     (no personas = whole roster)
#
# 1. cut_context.py per persona -> /archive/training/<persona>-cut/. Labels are
#    cached, so a re-run after a failure only labels what is left.
# 2. train_queue.sh <job-id> <persona>-cut ... -> lora-outputs/<persona>-cut-<job-id>.
#    Strictly after every cut: the judge's GPU lease (llm-batch) outranks the
#    trainer, so labelling while a run trains would preempt it on every call.
# 3. Per persona: topic eval of the live adapter, then the new run exported as
#    the candidate (never live) and the same eval of it. Results in
#    lora-outputs/topic_eval/<persona>-{live,cut}-<job-id>.jsonl and the log.
#
# Nothing goes live: `make train-export-latest PERSONA=...` is a human's call.
# Stops at the first failure.
set -u

JOB="${1:?usage: cut_queue.sh <job-id> [persona ...]}"
shift
REPO="${REPO:-/home/peacow/local-ai-server}"
TRAINING="${TRAINING:-/mnt/storage_cold/array/DiscordArchive/training}"
LOGDIR="${LOGDIR:-$HOME/.local/share/train-queue}"
LOG="$LOGDIR/cut-queue-$JOB.log"
mkdir -p "$LOGDIR"
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
RUN="docker compose --profile managed run --rm --no-deps -T lora-trainer"

cd "$REPO" || exit 1
if [ $# -gt 0 ]; then PERSONAS="$*"; else
  PERSONAS=$(python3 shared/persona_roster.py list --roster "$TRAINING/personas.json") || exit 1
fi

# The judge must run as batch work, below the LLM. If the arbiter does not know
# the name it would fall back to priority 0 — still below the LLM, but tied with
# the video job and so stuck behind it — and an old proxy would ask as "llm".
if ! curl -sf http://localhost:11438/gpu/status | grep -q '"llm-batch"'; then
  log "FAIL: the arbiter has no llm-batch job — deploy arbiter and proxy first"; exit 1
fi
for p in $PERSONAS; do
  [ -f "lora-training/configs/$p-cut.yaml" ] || { log "FAIL: no lora-training/configs/$p-cut.yaml"; exit 1; }
done

log "CUT-QUEUE $JOB: $PERSONAS"
docker compose --profile managed build lora-trainer >>"$LOG" 2>&1 || { log "FAIL: trainer image build"; exit 1; }

for p in $PERSONAS; do
  log "CUT $p"
  if ! $RUN python cut_context.py --persona "$p" >>"$LOG" 2>&1; then log "FAIL: cut $p"; exit 1; fi
  tail -1 "$LOG" | sed "s/^/  /"
done

QUEUE=$(for p in $PERSONAS; do printf '%s-cut ' "$p"; done)
log "TRAIN $QUEUE"
if ! lora-training/train_queue.sh "$JOB" $QUEUE; then log "FAIL: training queue (see queue-$JOB.log)"; exit 1; fi

for p in $PERSONAS; do
  if [ ! -s "$TRAINING/topic_eval/$p.jsonl" ]; then
    log "PROBES $p"
    $RUN python topic_eval.py build --persona "$p" >>"$LOG" 2>&1 || { log "FAIL: probes $p"; exit 1; }
  fi
  log "EVAL $p live"
  $RUN python topic_eval.py run --persona "$p" --models "$p" --tag "live-$JOB" >>"$LOG" 2>&1 \
    || { log "FAIL: eval $p live"; exit 1; }
  log "EVAL $p candidate ($p-cut-$JOB)"
  $RUN python export_latest.py --personas "$p" --run "/home/peacow/lora-outputs/$p-cut-$JOB" --candidate >>"$LOG" 2>&1 \
    || { log "FAIL: candidate export $p"; exit 1; }
  $RUN python topic_eval.py run --persona "$p" --models candidate --tag "cut-$JOB" >>"$LOG" 2>&1 \
    || { log "FAIL: eval $p candidate"; exit 1; }
done
log "CUT-QUEUE COMPLETE — nothing is live; compare lora-outputs/topic_eval/*-{live,cut}-$JOB.jsonl, then make train-export-latest"
