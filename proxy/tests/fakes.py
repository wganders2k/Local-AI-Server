"""
Test doubles shared by the proxy suites.

FakeArbiterService models the arbiter's admission rules closely enough to test
handovers against: a priority per job name, no preemption on a tie, `none`
jobs reclaimed by bookkeeping only, `cooperative` jobs asked through the
reclaim notice and waited on until they release.
"""

import asyncio

import httpx

import main

PRIORITY = {"llm": 100, "profile-segment": 50, "llm-batch": 25, "video-processing": 0}
KIND = {"llm": "none", "profile-segment": "none", "llm-batch": "cooperative", "video-processing": "cooperative"}


class FakeArbiterService:
    def __init__(self):
        self.holder: str | None = None
        self.calls: list[tuple[str, str]] = []
        self.refusals: dict[str, str] = {}  # name -> reason, refused regardless of rank
        self.gate = asyncio.Event()
        self.gate.set()
        self._wanted = asyncio.Event()
        self._let_go = asyncio.Event()

    def client(self, name: str) -> "FakeArbiterClient":
        return FakeArbiterClient(self, name)

    def block(self):
        """Hold every acquire open so a test can observe the window before it returns."""
        self.gate.clear()

    def unblock(self):
        self.gate.set()

    def names(self, action: str) -> list[str]:
        return [n for a, n in self.calls if a == action]

    async def acquire(self, name: str) -> tuple[bool, str]:
        self.calls.append(("acquire", name))
        await self.gate.wait()
        if name in self.refusals:
            return False, self.refusals[name]
        if self.holder == name:
            return True, "already held"
        if self.holder is not None:
            if PRIORITY[self.holder] >= PRIORITY[name]:
                return False, f"{self.holder} holds the GPU and {name} does not outrank it"
            if KIND[self.holder] == "cooperative":
                self._let_go.clear()
                self._wanted.set()
                try:
                    await asyncio.wait_for(self._let_go.wait(), 5.0)
                except asyncio.TimeoutError:
                    return False, f"could not reclaim the GPU from {self.holder}"
            self.holder = None
        self._wanted.clear()
        self.holder = name
        return True, "granted"

    async def release(self, name: str) -> None:
        self.calls.append(("release", name))
        if self.holder == name:
            self.holder = None
            self._let_go.set()

    async def reclaim_notice(self, name: str, timeout: float = 0.2) -> bool:
        if self.holder != name:
            return True  # the real arbiter's answer to a holder it has no lease for
        try:
            await asyncio.wait_for(self._wanted.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False


class FakeArbiterClient:
    def __init__(self, service: FakeArbiterService, name: str):
        self.service = service
        self.job_name = name

    async def acquire(self):
        return await self.service.acquire(self.job_name)

    async def release(self):
        await self.service.release(self.job_name)

    async def reclaim_notice(self):
        return await self.service.reclaim_notice(self.job_name)

    async def aclose(self):
        pass


class Backend:
    """Stub forwarder recording every attempt to reach the model backend."""

    def __init__(self):
        self.calls: list[str] = []
        self.hold: dict[str, asyncio.Event] = {}  # model -> released when set

    def hold_model(self, model: str) -> asyncio.Event:
        ev = asyncio.Event()
        self.hold[model] = ev
        return ev

    async def forward(self, target_base_url, request, model=None, on_complete=None):
        self.calls.append(model)
        try:
            if model in self.hold:
                await self.hold[model].wait()
            return httpx.Response(200, json={"ok": True, "model": model})
        finally:
            if on_complete:
                on_complete()


def install_dispatcher(monkeypatch, batch_yield=0.0, retry=0.05, free_gpu=None):
    """A fresh dispatcher on a fake arbiter, installed into main. Returns (service, dispatcher)."""
    service = FakeArbiterService()
    d = main.build_dispatcher(service.client("llm"), service.client("llm-batch"))
    d.batch_yield_seconds = batch_yield
    d.batch_retry_seconds = retry
    d.poll_seconds = 0.01
    d.lease._notice_retry = 0.01
    if free_gpu is not None:
        d.free_gpu = free_gpu
    monkeypatch.setattr(main, "dispatcher", d)
    return service, d


async def settle(seconds: float = 0.05):
    """Let scheduled tasks (finish -> release) run."""
    await asyncio.sleep(seconds)
