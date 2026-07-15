import asyncio
import json
import signal

import pytest

import services.latentsync_server as server


@pytest.fixture(autouse=True)
def _clear_job_registry():
    server._ACTIVE_PROCESSES.clear()
    server._CANCELLED_JOBS.clear()
    yield
    server._ACTIVE_PROCESSES.clear()
    server._CANCELLED_JOBS.clear()


class _HangingProcess:
    def __init__(self) -> None:
        self.pid = 4242
        self.returncode = None
        self.started = asyncio.Event()
        self._never = asyncio.Event()
        self.terminate_called = False
        self.kill_called = False

    async def communicate(self):
        self.started.set()
        await self._never.wait()

    async def wait(self):
        self.returncode = -signal.SIGTERM
        return self.returncode

    def terminate(self) -> None:
        self.terminate_called = True

    def kill(self) -> None:
        self.kill_called = True


@pytest.mark.asyncio
async def test_run_subprocess_cancellation_terminates_process_group(monkeypatch) -> None:
    process = _HangingProcess()
    create_calls: list[dict] = []
    signals: list[tuple[int, signal.Signals]] = []

    async def fake_create(*_cmd, **kwargs):
        create_calls.append(kwargs)
        return process

    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_create)
    monkeypatch.setattr(server.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    task = asyncio.create_task(server._run_subprocess(["worker"], timeout=60))
    await process.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert create_calls[0]["start_new_session"] is True
    assert signals == [(process.pid, signal.SIGTERM)]
    assert process.returncode == -signal.SIGTERM


class _StubbornProcess(_HangingProcess):
    def __init__(self) -> None:
        super().__init__()
        self.wait_calls = 0

    async def wait(self):
        self.wait_calls += 1
        if self.wait_calls == 1:
            await self._never.wait()
        self.returncode = -signal.SIGKILL
        return self.returncode


@pytest.mark.asyncio
async def test_terminate_process_group_escalates_to_sigkill(monkeypatch) -> None:
    process = _StubbornProcess()
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(server, "_PROCESS_TERMINATE_GRACE_SEC", 0.001)
    monkeypatch.setattr(server.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    await server._terminate_process_group(process)

    assert signals == [
        (process.pid, signal.SIGTERM),
        (process.pid, signal.SIGKILL),
    ]
    assert process.returncode == -signal.SIGKILL


class _EndpointCancelledProcess(_HangingProcess):
    async def communicate(self):
        self.started.set()
        await self._never.wait()
        return b"", b"cancelled"

    async def wait(self):
        self.returncode = -signal.SIGTERM
        self._never.set()
        return self.returncode


@pytest.mark.asyncio
async def test_cancel_endpoint_stops_only_registered_job_process(monkeypatch) -> None:
    process = _EndpointCancelledProcess()
    signals: list[tuple[int, signal.Signals]] = []

    async def fake_create(*_cmd, **_kwargs):
        return process

    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", fake_create)
    monkeypatch.setattr(server.os, "killpg", lambda pid, sig: signals.append((pid, sig)))

    task = asyncio.create_task(
        server._run_subprocess(["worker"], timeout=60, job_id="job-1")
    )
    await process.started.wait()
    assert process in server._ACTIVE_PROCESSES["job-1"]

    response = await server.cancel_job("job-1")
    with pytest.raises(server.JobCancelledError):
        await task

    payload = json.loads(response.body)
    assert payload["cancelled_processes"] == 1
    assert signals == [(process.pid, signal.SIGTERM)]
    assert "job-1" not in server._ACTIVE_PROCESSES


@pytest.mark.asyncio
async def test_cancelled_job_does_not_start_new_subprocess(monkeypatch) -> None:
    server._CANCELLED_JOBS.add("job-1")
    called = False

    async def should_not_start(*_cmd, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(server.asyncio, "create_subprocess_exec", should_not_start)

    with pytest.raises(server.JobCancelledError):
        await server._run_subprocess(["worker"], timeout=60, job_id="job-1")
    assert called is False
