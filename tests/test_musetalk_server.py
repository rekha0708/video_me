import asyncio
import json
from io import BytesIO
from pathlib import Path
import signal

import pytest
from fastapi import UploadFile

import services.musetalk_server as server


@pytest.fixture(autouse=True)
def _clear_job_registry():
    server._ACTIVE_PROCESSES.clear()
    server._CANCELLED_JOBS.clear()
    yield
    server._ACTIVE_PROCESSES.clear()
    server._CANCELLED_JOBS.clear()


class _HangingProcess:
    def __init__(self) -> None:
        self.pid = 5252
        self.returncode = None
        self.started = asyncio.Event()
        self._never = asyncio.Event()

    async def communicate(self):
        self.started.set()
        await self._never.wait()

    async def wait(self):
        self.returncode = -signal.SIGTERM
        return self.returncode

    def terminate(self) -> None:
        pass

    def kill(self) -> None:
        pass


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
async def test_cancel_endpoint_stops_registered_job_process(monkeypatch) -> None:
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


def _upload(filename: str, content: bytes) -> UploadFile:
    return UploadFile(file=BytesIO(content), filename=filename)


def _prepare_repo(tmp_path: Path, monkeypatch) -> Path:
    repo = tmp_path / "MuseTalk"
    (repo / server._UNET_MODEL).parent.mkdir(parents=True)
    (repo / server._UNET_MODEL).write_bytes(b"weights")
    monkeypatch.setattr(server, "MUSETALK_DIR", repo)
    return repo


@pytest.mark.asyncio
async def test_lipsync_marks_generated_video_as_applied(tmp_path: Path, monkeypatch) -> None:
    _prepare_repo(tmp_path, monkeypatch)

    async def fake_run(cmd, **_kwargs):
        result_dir = Path(cmd[cmd.index("--result_dir") + 1])
        output = result_dir / "v15" / "input.mp4"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"synced-video")
        return 0, b"", b""

    monkeypatch.setattr(server, "_run_subprocess", fake_run)

    response = await server.lipsync(
        _upload("input.mp4", b"source-video"),
        _upload("input.wav", b"source-audio"),
        "shot-1",
        "job-1",
    )

    assert response.body == b"synced-video"
    assert response.headers["X-Video-Me-Lipsync"] == "applied"


@pytest.mark.asyncio
async def test_lipsync_marks_missing_face_output_as_passthrough(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _prepare_repo(tmp_path, monkeypatch)

    async def fake_run(_cmd, **_kwargs):
        return 0, b"", b""

    monkeypatch.setattr(server, "_run_subprocess", fake_run)

    response = await server.lipsync(
        _upload("input.mp4", b"source-video"),
        _upload("input.wav", b"source-audio"),
        "shot-2",
        "job-2",
    )

    assert response.body == b"source-video"
    assert response.headers["X-Video-Me-Lipsync"] == "passthrough"
