# GPU Arbiter

The only component that decides who holds the GPU, and the only one that can see
it.

## Why it exists

One RTX 3090, three tenants: llama-server, a video processing job, and a LoRA
trainer. CUDA has no memory preemption, so something has to arbitrate.

The previous design did it by asking. Each job was told "please release" over
HTTP and replied "done", and the proxy believed it — because a containerised
proxy has no way to check. That produced ~2,900 lines: a three-state handover
machine, stale-registration sweeps, a 180s timeout, a 409 refusal path, 1/sec
polling from every client, and a client library vendored into three repos. All of
it to work around one missing capability.

What replaced it is a lease. A job asks; it is granted or refused; when something
that outranks it asks, it is told and gives the card back. Most of that machinery
turned out to be working around the absence of that one idea rather than the
absence of a capability — there is no registration table, no parking, no polling
and no client library shared between repos.

## What it is, and what it deliberately is not

An **admission controller**. Every job runs itself, asks for the card, and is
granted or refused:

```
acquire(job)   reclaim from everything the caller outranks, wait for the driver,
               grant the lease
release(job)   the caller is done; the lease is free
reap()         housekeeping — clear a lease whose holder died without releasing.
               Nothing waits on it.
```

**It is not a scheduler, and it starts nothing.** It used to: `tick()` picked up
the highest-priority job that fitted and `docker start`ed it. That meant a
training run began because free VRAM happened to appear, with no human anywhere
in the chain — which was never a decision anyone made. Work is *submitted* by
whoever owns it. The arbiter only says who may hold the card while it runs, which
is the one question that genuinely needs a central answer.

The whole scheduling half is gone with it: backoff, failure counters, exit-code
classification, and the idle-VRAM trigger. `test_nothing_is_ever_started` asserts
the negative, and the test double has no `start` method at all — a regression is
an `AttributeError`, not a quietly passing test.

## There is no privileged tenant

The LLM is a row in [`jobs.yaml`](jobs.yaml) like everything else. It wins because
its `priority` is 100; nothing in this codebase contains the string `llm`. Invert
two numbers in that file and the trainer wins instead — `test_priority_alone_
decides_who_wins` asserts exactly that, and is the test that stops the special
case growing back.

The tempting implementation is a boolean: *the LLM holds the card*. It is smaller,
and it is wrong in a specific way. It fuses two independent facts:

| | why |
|---|---|
| the LLM **outranks** the others | it is interactive; a user is waiting |
| the arbiter **does not stop** it | the proxy already starts and stops llama-server to swap models, and two services driving one container would race |

Those have nothing to do with each other. The first is `priority`, the second is
`kind: none`, and both are config. A boolean would have made a second privileged
tenant unrepresentable, and left "how important" and "how controllable" impossible
to set independently.

Callers send a name and nothing else. Not a priority, not a VRAM figure, not what
should be stopped for them — a job that could state its own importance is grading
its own homework, and the LLM gets no exemption from that.

The one fallback: a name absent from `jobs.yaml` is granted the top rank and
logged loudly on every call. Permissive on purpose. The callers are the
interactive services, and the alternative is a YAML typo taking the LLM offline
with 503s.

## Two things that are not obvious

**An empty lease table is not headroom.** `release()` means no request is in
flight, not that there is room — the model stays resident until the proxy's idle
evictor unloads it, so a 16 GB job that trusted the lease alone would OOM against
an 18 GB model. Every grant is checked against a real NVML reading, which is why
the arbiter is the only component that needs to see the card.

**A running container is not a tenant.** Holding is a lease the arbiter granted,
not a state inferred from Docker. The trainer is up and alive while it waits for
permission; reading "container is running" as "holds the card" would kill it on
every LLM request and it would spend its life restarting.

**A lease has one answer, and that is why "is it on the card" stopped being a
separate question.** The previous design asked Docker whether a container was up
and tried to read intent out of the reply, and a registered-but-parked job made
"on the card" and "wants the card" diverge. Reading either for the other is a bug
in both directions — treating parked as free grants a junior job that the
resuming one then OOMs; treating it as occupied stops it once per tick forever.
Both happened, in that order, on the first deploy. Holding is now something the
arbiter wrote down, so there is one answer and nothing to reconcile.

## Policy

All of it in [`jobs.yaml`](jobs.yaml) — priority and memory requirements are
operator decisions, and a job declaring its own would be self-assessment. Adding a
fourth GPU workload is one entry there and nothing anywhere else.

Every job asks for the card the same way, so `kind` no longer says how a job is
run. It says one thing only: **how the card is taken back from it.**

| kind | reclaim | used by |
|---|---|---|
| `none` | drop the lease; nothing to tear down | `llm` |
| `cooperative` | tell the job; it stops its own GPU process | `video-processing`, `lora-trainer` |
| `container` | `docker kill` — cgroup teardown | nothing, currently |

`none` is correct only because nothing outranks the job that uses it, so its
unenforceable reclaim is never actually exercised.

`cooperative` is what both real tenants ended up as, for the same reason in each
case: **the process holding the CUDA context is not the process the arbiter can
reach.** For the video job it is a child of a host daemon. For the trainer it is
a child of the container's PID 1 — and killing the container instead does not
work, because Docker suppresses restart policies for any API-initiated stop or
kill, so a preempted run would never come back. Measured: `RestartCount=0` after
`docker kill` for both `always` and `on-failure`.

What that costs is verification. A cooperative job asserts it let go and nothing
here can check. It is bounded by a timeout, and a timeout is a refusal — making
the LLM wait is safe; loading a model over memory that may still be held is not.

`container` is the enforceable one and currently has no user. Worth knowing: if
it never gains one, the arbiter does not need Docker at all — and with it goes
the privilege discussed below.

## Privilege

Reaches Docker through `docker-socket-proxy`, not `/var/run/docker.sock`. Be
clear about what that buys, because the comment here used to overstate it:

```
POST /containers/create  -> 404   (reached the daemon)
POST /containers/*/exec  -> 404   (reached the daemon)
POST /build              -> 403   (blocked)
```

`CONTAINERS` + `POST` still admits `/containers/create`, and a container created
with `Binds: ["/:/host"]` and `Privileged: true` is root on the host. So the
proxy **narrows the surface, not the privilege** — this service is
root-equivalent either way, and must stay on the internal compose network.

Two things follow. Tightening the proxy's allowance would be a real improvement.
And since `kind: container` currently has no user, dropping the Docker dependency
entirely would remove this exposure outright.

GPU access is `NVIDIA_DRIVER_CAPABILITIES=utility`: NVML and `nvidia-smi`, no
CUDA. It can measure the card and is structurally incapable of allocating on it,
which matters for the thing deciding whether anyone *else* has room.

## Cooperative jobs, and what they trade away

The video processing job is not a container, so it cannot be stopped — only
asked.
Reaching the process that holds the CUDA context would mean handing the arbiter
systemd's user manager — the authority to run anything at all as that user, to
stop one process the job already supervises. Not worth it. So a cooperative job
holds a blocking `GET /gpu/reclaim-notice` for as long as it holds the card, and
reclaiming is waking that call and waiting for the release.

That connection is also the liveness signal. A job holding the card is a job
holding one of these; if it stops, its process is gone and the reaper takes the
lease back. Registration, heartbeats and staleness sweeps were all doing this job
badly.

Measured on hardware, before and after: a handover that took **2–4s** through
register/park/confirm with 1/sec polling now takes **0.9–2.0s**, which is what
`docker kill` cost. The polling was most of the difference.

What is not recovered is verification: the job asserts it let go, and nothing
here can check. Bounded by a timeout, and a timeout is a refusal.

## Endpoints

```
POST /gpu/acquire        {"name": ...} take the card, preempting what you
                         outrank; 503 if it cannot be guaranteed
POST /gpu/release        {"name": ...} done with it
GET  /gpu/status         holder, reason, free VRAM, the job table
GET  /metrics            Prometheus
POST /external-job/*     transitional; see above
```

## Tests

```bash
.venv-test/bin/python -m pytest tests -q
```

`test_scheduler.py` pins the guarantees the design rests on: nothing is ever
started, acquire never reports success unless everything junior is actually gone,
a lease never outlives its holder, and every decision is `priority` alone —
invert the ranking and the LLM loses.
