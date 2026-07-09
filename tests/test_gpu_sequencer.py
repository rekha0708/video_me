from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from core.gpu_sequencer import (
    VIDEO_MODEL_LOAD_STAGE,
    VOICE_MODEL_LOAD_STAGE,
    ensure_video_model_unloaded,
    prepare_video_model,
    prepare_voice_model,
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

    with patch(
        "core.gpu_sequencer.unload_ollama_model",
        side_effect=lambda *a, **k: calls.append("unload_ollama"),
    ):
        await prepare_voice_model(adapter, _settings(fish_s2_load_gap_sec=5), sleep=sleep)

    assert calls == ["unload_ollama", "sleep:5", "load", "wait"]


async def test_prepare_voice_uses_gap_and_timeout_from_settings() -> None:
    adapter = _managed_adapter()
    sleep = AsyncMock()
    with patch("core.gpu_sequencer.unload_ollama_model"):
        await prepare_voice_model(
            adapter,
            _settings(fish_s2_load_gap_sec=3, fish_s2_load_timeout_sec=42),
            sleep=sleep,
        )
    sleep.assert_awaited_once_with(3)
    adapter.wait_until_loaded.assert_awaited_once_with(42)


async def test_prepare_voice_emits_voice_model_load_events() -> None:
    adapter = _managed_adapter()
    notify = MagicMock()
    with patch("core.gpu_sequencer.unload_ollama_model"):
        await prepare_voice_model(adapter, _settings(), sleep=AsyncMock(), notify=notify)
    notify.assert_any_call(VOICE_MODEL_LOAD_STAGE, "stage_started")
    notify.assert_any_call(VOICE_MODEL_LOAD_STAGE, "stage_completed")


async def test_prepare_voice_no_events_for_unmanaged_adapter() -> None:
    notify = MagicMock()
    await prepare_voice_model(_unmanaged_adapter(), _settings(), sleep=AsyncMock(), notify=notify)
    notify.assert_not_called()
