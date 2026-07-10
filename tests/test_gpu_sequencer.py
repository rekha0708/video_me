import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from core.gpu_sequencer import (
    VIDEO_MODEL_LOAD_STAGE,
    VOICE_MODEL_LOAD_STAGE,
    ensure_fish_s2_process_running,
    ensure_video_model_unloaded,
    free_comfyui,
    prepare_video_model,
    prepare_voice_model,
    stop_fish_s2_process,
    unload_wan,
)


def _settings(**overrides) -> SimpleNamespace:
    base = dict(
        llm_base_url="http://localhost:11434/v1",
        llm_model="qwen3.6:35b",
        image_critique_base_url="http://localhost:11434/v1",
        image_critique_model="qwen3.6:35b",
        wan_load_gap_sec=30,
        wan_load_timeout_sec=1800,
        fish_s2_load_gap_sec=5,
        fish_s2_load_timeout_sec=120,
        fish_s2_base_url="http://localhost:8025",
        fish_s2_venv_python="/workspace/.venv_fish_s2/bin/uvicorn",
        fish_s2_speech_dir="/workspace/fish-speech",
        fish_s2_log_path="/workspace/logs/fish_s2.log",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _managed_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.managed_vram = True
    adapter.load = AsyncMock()
    adapter.unload = AsyncMock(return_value=True)
    adapter.wait_until_loaded = AsyncMock()
    return adapter


def _unmanaged_adapter() -> MagicMock:
    adapter = MagicMock(spec=[])  # no managed_vram attribute (like musubi_flux)
    return adapter


# ------------------------------------------------------------------ no-op path

async def test_unload_is_noop_for_unmanaged_adapter() -> None:
    adapter = _unmanaged_adapter()
    await ensure_video_model_unloaded(adapter)
    # spec=[] would raise on any attribute access — reaching here means no calls


async def test_prepare_is_noop_for_unmanaged_adapter() -> None:
    adapter = _unmanaged_adapter()
    sleep = AsyncMock()
    with patch("core.gpu_sequencer.unload_ollama_model") as unload_llm:
        await prepare_video_model(adapter, _settings(), sleep=sleep)
    unload_llm.assert_not_called()
    sleep.assert_not_called()


# ------------------------------------------------------------------ unload

async def test_unload_calls_adapter_unload() -> None:
    adapter = _managed_adapter()
    await ensure_video_model_unloaded(adapter)
    adapter.unload.assert_awaited_once()


# ------------------------------------------------------------------ prepare ordering

async def test_prepare_orders_unload_gap_load_wait() -> None:
    calls: list[str] = []
    adapter = _managed_adapter()
    adapter.load.side_effect = lambda: calls.append("load")
    adapter.wait_until_loaded.side_effect = lambda *a, **k: calls.append("wait")

    async def sleep(sec: float) -> None:
        calls.append(f"sleep:{sec}")

    with patch(
        "core.gpu_sequencer.unload_ollama_model",
        side_effect=lambda *a, **k: calls.append("unload_ollama"),
    ):
        await prepare_video_model(adapter, _settings(wan_load_gap_sec=30), sleep=sleep)

    assert calls == ["unload_ollama", "sleep:30", "load", "wait"]


async def test_prepare_uses_gap_and_timeout_from_settings() -> None:
    adapter = _managed_adapter()
    sleep = AsyncMock()
    with patch("core.gpu_sequencer.unload_ollama_model"):
        await prepare_video_model(
            adapter, _settings(wan_load_gap_sec=7, wan_load_timeout_sec=99), sleep=sleep
        )
    sleep.assert_awaited_once_with(7)
    adapter.wait_until_loaded.assert_awaited_once_with(99)


async def test_prepare_unloads_critique_model_when_different() -> None:
    adapter = _managed_adapter()
    settings = _settings(
        image_critique_base_url="http://other:11434/v1", image_critique_model="other-vlm"
    )
    with patch("core.gpu_sequencer.unload_ollama_model") as unload_llm:
        await prepare_video_model(adapter, settings, sleep=AsyncMock())
    assert unload_llm.call_count == 2


async def test_prepare_unloads_ollama_once_when_models_match() -> None:
    adapter = _managed_adapter()
    with patch("core.gpu_sequencer.unload_ollama_model") as unload_llm:
        await prepare_video_model(adapter, _settings(), sleep=AsyncMock())
    assert unload_llm.call_count == 1


# ------------------------------------------------------------------ notify events

async def test_prepare_emits_video_model_load_events() -> None:
    adapter = _managed_adapter()
    notify = MagicMock()
    with patch("core.gpu_sequencer.unload_ollama_model"):
        await prepare_video_model(adapter, _settings(), sleep=AsyncMock(), notify=notify)
    notify.assert_any_call(VIDEO_MODEL_LOAD_STAGE, "stage_started")
    notify.assert_any_call(VIDEO_MODEL_LOAD_STAGE, "stage_completed")


async def test_prepare_no_events_for_unmanaged_adapter() -> None:
    notify = MagicMock()
    await prepare_video_model(_unmanaged_adapter(), _settings(), sleep=AsyncMock(), notify=notify)
    notify.assert_not_called()


# ------------------------------------------------------------------ prepare_voice_model
# Same contract as prepare_video_model (shared _prepare_managed_adapter helper),
# just reading the fish_s2_* gap/timeout settings and emitting VOICE_MODEL_LOAD_STAGE.

async def test_prepare_voice_is_noop_for_unmanaged_adapter() -> None:
    adapter = _unmanaged_adapter()
    sleep = AsyncMock()
    with patch("core.gpu_sequencer.unload_ollama_model") as unload_llm:
        await prepare_voice_model(adapter, _settings(), sleep=sleep)
    unload_llm.assert_not_called()
    sleep.assert_not_called()


async def test_prepare_voice_orders_unload_gap_load_wait() -> None:
    calls: list[str] = []
    adapter = _managed_adapter()
    adapter.load.side_effect = lambda: calls.append("load")
    adapter.wait_until_loaded.side_effect = lambda *a, **k: calls.append("wait")

    async def sleep(sec: float) -> None:
        calls.append(f"sleep:{sec}")

    with (
        patch(
            "core.gpu_sequencer.unload_ollama_model",
            side_effect=lambda *a, **k: calls.append("unload_ollama"),
        ),
        patch("core.gpu_sequencer.ensure_fish_s2_process_running", new=AsyncMock()),
    ):
        await prepare_voice_model(adapter, _settings(fish_s2_load_gap_sec=5), sleep=sleep)

    assert calls == ["unload_ollama", "sleep:5", "load", "wait"]


async def test_prepare_voice_uses_gap_and_timeout_from_settings() -> None:
    adapter = _managed_adapter()
    sleep = AsyncMock()
    with (
        patch("core.gpu_sequencer.unload_ollama_model"),
        patch("core.gpu_sequencer.ensure_fish_s2_process_running", new=AsyncMock()),
    ):
        await prepare_voice_model(
            adapter,
            _settings(fish_s2_load_gap_sec=3, fish_s2_load_timeout_sec=42),
            sleep=sleep,
        )
    sleep.assert_awaited_once_with(3)
    adapter.wait_until_loaded.assert_awaited_once_with(42)


async def test_prepare_voice_ensures_process_running_only_when_managed() -> None:
    ensure_process = AsyncMock()
    with (
        patch("core.gpu_sequencer.unload_ollama_model"),
        patch("core.gpu_sequencer.ensure_fish_s2_process_running", new=ensure_process),
    ):
        await prepare_voice_model(_managed_adapter(), _settings(), sleep=AsyncMock())
    ensure_process.assert_awaited_once()

    ensure_process.reset_mock()
    await prepare_voice_model(_unmanaged_adapter(), _settings(), sleep=AsyncMock())
    ensure_process.assert_not_called()


async def test_prepare_voice_emits_voice_model_load_events() -> None:
    adapter = _managed_adapter()
    notify = MagicMock()
    with (
        patch("core.gpu_sequencer.unload_ollama_model"),
        patch("core.gpu_sequencer.ensure_fish_s2_process_running", new=AsyncMock()),
    ):
        await prepare_voice_model(adapter, _settings(), sleep=AsyncMock(), notify=notify)
    notify.assert_any_call(VOICE_MODEL_LOAD_STAGE, "stage_started")
    notify.assert_any_call(VOICE_MODEL_LOAD_STAGE, "stage_completed")


async def test_prepare_voice_no_events_for_unmanaged_adapter() -> None:
    notify = MagicMock()
    await prepare_voice_model(_unmanaged_adapter(), _settings(), sleep=AsyncMock(), notify=notify)
    notify.assert_not_called()


# ------------------------------------------------------------------ free_comfyui
# ComfyUI is invisible to ensure_video_model_unloaded/prepare_*_model unless it's
# THIS job's own render/video adapter — free_comfyui talks to it directly by URL
# instead, so a model left resident by a *different* job's adapter choice still
# gets freed. See ComfyUIUnloadMixin.unload() for the identical adapter-bound
# version of this same call.

def _mock_httpx(*, post_error: Exception | None = None, status_code: int = 200):
    mock_post_resp = MagicMock()
    mock_post_resp.status_code = status_code
    mock_post_resp.text = "error body"

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    mock_client.post = AsyncMock(
        side_effect=post_error if post_error else None,
        return_value=mock_post_resp,
    )

    fake_httpx = MagicMock()
    fake_httpx.AsyncClient = MagicMock(return_value=mock_client)
    return fake_httpx, mock_client


async def test_free_comfyui_posts_to_free_endpoint() -> None:
    fake_httpx, mock_client = _mock_httpx()
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        result = await free_comfyui("http://localhost:8188")
    assert result is True
    assert mock_client.post.call_args.args[0] == "http://localhost:8188/free"
    assert mock_client.post.call_args.kwargs["json"] == {
        "unload_models": True,
        "free_memory": True,
    }


async def test_free_comfyui_strips_trailing_slash() -> None:
    fake_httpx, mock_client = _mock_httpx()
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        await free_comfyui("http://localhost:8188/")
    assert mock_client.post.call_args.args[0] == "http://localhost:8188/free"


async def test_free_comfyui_returns_false_when_unreachable() -> None:
    class _FakeConnectError(Exception):
        pass

    fake_httpx, _ = _mock_httpx(post_error=_FakeConnectError("refused"))
    fake_httpx.ConnectError = _FakeConnectError
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        result = await free_comfyui("http://localhost:8188")
    assert result is False


async def test_free_comfyui_returns_false_on_http_error_without_raising() -> None:
    # Deliberately not the same contract as ComfyUIUnloadMixin.unload() (which
    # raises) — free_comfyui is a best-effort safety net called unconditionally
    # regardless of whether ComfyUI is even in use, so a hiccup here must never
    # fail an otherwise-healthy job.
    fake_httpx, _ = _mock_httpx(status_code=500)
    fake_httpx.ConnectError = ConnectionError
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        result = await free_comfyui("http://localhost:8188")
    assert result is False


# ------------------------------------------------------------------ Fish S2 process lifecycle
# POST /unload's torch.cuda.empty_cache() only reclaims a fraction of what
# Fish S2 accumulates across many synthesis calls within one long-running
# process (observed ~63GB resident vs ~20GB fresh-process baseline). The
# worker kills the whole process after every job and respawns it on demand.

def _fake_proc(returncode: int = 0) -> AsyncMock:
    proc = AsyncMock()
    proc.returncode = returncode
    proc.wait = AsyncMock(return_value=returncode)
    return proc


async def test_stop_fish_s2_process_pkills_matching_pattern() -> None:
    fake_exec = AsyncMock(return_value=_fake_proc(returncode=0))
    with patch("core.gpu_sequencer.asyncio.create_subprocess_exec", new=fake_exec):
        result = await stop_fish_s2_process()
    assert result is True
    args = fake_exec.call_args.args
    assert args[0] == "pkill"
    assert "services.fish_s2_server:app" in args


async def test_stop_fish_s2_process_returns_false_when_nothing_to_kill() -> None:
    # pkill exits 1 when no process matched the pattern — a normal no-op.
    fake_exec = AsyncMock(return_value=_fake_proc(returncode=1))
    with patch("core.gpu_sequencer.asyncio.create_subprocess_exec", new=fake_exec):
        result = await stop_fish_s2_process()
    assert result is False


async def test_ensure_fish_s2_process_running_noop_when_already_reachable() -> None:
    fake_exec = AsyncMock()
    with (
        patch("core.gpu_sequencer._fish_s2_reachable", new=AsyncMock(return_value=True)),
        patch("core.gpu_sequencer.asyncio.create_subprocess_exec", new=fake_exec),
    ):
        await ensure_fish_s2_process_running(_settings())
    fake_exec.assert_not_called()


async def test_ensure_fish_s2_process_running_spawns_when_unreachable(tmp_path) -> None:
    reachable = AsyncMock(side_effect=[False, False, True])  # unreachable → spawn → up
    fake_exec = AsyncMock(return_value=_fake_proc())
    sleep = AsyncMock()
    settings = _settings(
        fish_s2_base_url="http://localhost:8025",
        fish_s2_venv_python="/fake/venv/bin/uvicorn",
        fish_s2_speech_dir="/fake/fish-speech",
        fish_s2_log_path=str(tmp_path / "fish_s2.log"),
    )
    with (
        patch("core.gpu_sequencer._fish_s2_reachable", new=reachable),
        patch("core.gpu_sequencer.asyncio.create_subprocess_exec", new=fake_exec),
    ):
        await ensure_fish_s2_process_running(settings, sleep=sleep)

    fake_exec.assert_awaited_once()
    args = fake_exec.call_args.args
    assert args[0] == "/fake/venv/bin/uvicorn"
    assert "services.fish_s2_server:app" in args
    assert "8025" in args
    kwargs = fake_exec.call_args.kwargs
    assert kwargs["env"]["FISH_SPEECH_DIR"] == "/fake/fish-speech"
    assert kwargs["cwd"]  # spawned with an explicit repo-root cwd
    sleep.assert_awaited_once_with(2.0)  # default poll_sec


async def test_ensure_fish_s2_process_running_raises_timeout_if_never_reachable(tmp_path) -> None:
    # fish_s2_load_timeout_sec is the ONE shared budget (see core/config.py) —
    # no separate "process startup" timeout to independently get wrong.
    settings = _settings(
        fish_s2_log_path=str(tmp_path / "fish_s2.log"),
        fish_s2_load_timeout_sec=2,
    )
    with (
        patch("core.gpu_sequencer._fish_s2_reachable", new=AsyncMock(return_value=False)),
        patch("core.gpu_sequencer.asyncio.create_subprocess_exec", new=AsyncMock(return_value=_fake_proc())),
    ):
        try:
            await ensure_fish_s2_process_running(settings, sleep=AsyncMock())
            assert False, "expected TimeoutError"
        except TimeoutError as exc:
            assert "2s" in str(exc)


# ------------------------------------------------------------------ unload_wan
# Nothing in the mainline pipeline ever unloads Wan after Phase B loads it —
# the only ensure_video_model_unloaded(adapters.video) call runs once, before
# Phase A. unload_wan() is called unconditionally from dashboard_worker's
# per-job `finally` instead, independent of adapter instances (same URL-based
# shape as free_comfyui, since dashboard_worker never constructs adapters).

async def test_unload_wan_posts_to_unload_endpoint() -> None:
    fake_httpx, mock_client = _mock_httpx()
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        result = await unload_wan("http://localhost:8030")
    assert result is True
    assert mock_client.post.call_args.args[0] == "http://localhost:8030/unload"


async def test_unload_wan_strips_trailing_slash() -> None:
    fake_httpx, mock_client = _mock_httpx()
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        await unload_wan("http://localhost:8030/")
    assert mock_client.post.call_args.args[0] == "http://localhost:8030/unload"


async def test_unload_wan_returns_false_when_unreachable() -> None:
    class _FakeConnectError(Exception):
        pass

    fake_httpx, _ = _mock_httpx(post_error=_FakeConnectError("refused"))
    fake_httpx.ConnectError = _FakeConnectError
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        result = await unload_wan("http://localhost:8030")
    assert result is False


async def test_unload_wan_returns_false_on_http_error_without_raising() -> None:
    # Best-effort, unlike WanAdapter.unload() (which raises) — this is called
    # unconditionally after every job regardless of which video_adapter it
    # used, so a hiccup here must never fail an otherwise-healthy job.
    fake_httpx, _ = _mock_httpx(status_code=500)
    fake_httpx.ConnectError = ConnectionError
    with patch.dict(sys.modules, {"httpx": fake_httpx}):
        result = await unload_wan("http://localhost:8030")
    assert result is False
