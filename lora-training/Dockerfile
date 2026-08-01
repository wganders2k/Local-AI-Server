# Two processes. supervisor.py is PID 1 and owns the GPU lease; train_worker.py
# is its child and owns the card.
#
# The split is forced by how CUDA and Docker actually behave. A CUDA context is
# freed when its process exits, so preemption means killing the training
# process — and if that were PID 1, the container would die with it. Docker
# suppresses a container's restart policy for any API-initiated stop or kill
# (measured: RestartCount=0 for both `always` and `on-failure`), so a preempted
# run would never come back.
#
# A human submits the run (`make train-submit`). The container's exit code says
# how it ended: 0 for a finished run, anything else for a genuine failure.
# Preemption is not visible in it at all — the supervisor absorbs it.

FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# torch and its CUDA stack come from the base image; installing the pinned torch
# from requirements.txt on top would pull a second copy. Everything else is
# version-sensitive against it — see the note in requirements.txt.
COPY requirements.txt .
RUN grep -vE '^(torch|pytest)==' requirements.txt > /tmp/reqs.txt \
    && pip install --no-cache-dir -r /tmp/reqs.txt \
    && rm /tmp/reqs.txt

COPY . .

# Unbuffered so a preempted run's logs are not lost with it — SIGKILL gives
# Python no chance to flush.
ENV PYTHONUNBUFFERED=1

# Which run this is, chosen at submit time. configs/ is bind-mounted over the
# copy above, so editing a YAML or a JSONL does not mean rebuilding an image
# with a multi-GB CUDA base layer.
ENV TRAIN_CONFIG=configs/smoke.yaml

CMD ["python", "supervisor.py"]
