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

# A C compiler, which the runtime base image lacks. TRL computes the loss with a
# fused LM-head Triton kernel (it never materialises the full logits — the
# biggest buffer with Gemma's 262k vocabulary), and Triton builds a small
# CUDA-utils shim with gcc the first time it runs in each process.
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# torch and its CUDA stack come from the base image; installing the pinned torch
# from requirements.txt on top would pull a second copy. Everything else is
# version-sensitive against it — see the note in requirements.txt.
COPY requirements.txt .
RUN grep -vE '^torch==' requirements.txt > /tmp/reqs.txt \
    && pip install --no-cache-dir -r /tmp/reqs.txt \
    && rm /tmp/reqs.txt

# llama.cpp's adapter converter, for export_adapter.py. Pinned to a release tag:
# the converter must know the base's architecture (Gemma 4 unified needs a 2026
# tag), and its `gguf` package is installed from the same tree rather than PyPI
# so the two can never disagree about tensor names. Only the Python half is
# kept — nothing here is compiled.
ARG LLAMA_CPP_TAG=b11627
ADD https://github.com/ggml-org/llama.cpp/archive/refs/tags/${LLAMA_CPP_TAG}.tar.gz /tmp/llama.cpp.tar.gz
RUN mkdir -p /opt/llama.cpp \
    && tar -xzf /tmp/llama.cpp.tar.gz -C /opt/llama.cpp --strip-components=1 \
        llama.cpp-${LLAMA_CPP_TAG}/convert_lora_to_gguf.py \
        llama.cpp-${LLAMA_CPP_TAG}/conversion \
        llama.cpp-${LLAMA_CPP_TAG}/gguf-py \
    && pip install --no-cache-dir /opt/llama.cpp/gguf-py \
    && rm /tmp/llama.cpp.tar.gz
ENV LLAMA_CPP_DIR=/opt/llama.cpp

COPY . .

# Unbuffered so a preempted run's logs are not lost with it — SIGKILL gives
# Python no chance to flush.
ENV PYTHONUNBUFFERED=1

# Which run this is, chosen at submit time. configs/ is bind-mounted over the
# copy above, so editing a YAML or a JSONL does not mean rebuilding an image
# with a multi-GB CUDA base layer.
ENV TRAIN_CONFIG=configs/smoke.yaml

CMD ["python", "supervisor.py"]
