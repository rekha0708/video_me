"""
GPU model sequencing between pipeline phases.

The Wan2.2 video model is too large to share the GPU with the render-phase
models (Flux subprocess + Ollama critique VLM) — loading both caused the
render_character OOM diagnosed in commit 58ce9d8. This module keeps each
VRAM-managed adapter (video, voice/TTS, and ComfyUI-backed render/video)
unloaded during phases that don't need it and brings it up only when its
phase begins:

    unload Ollama → wait gap → POST /load → poll until ready

Adapters opt in by setting a truthy ``managed_vram`` class attribute and
implementing ``load()``, ``unload()`` and ``wait_until_loaded()``
(see ``adapters/generate_video/wan_adapter.py`` and
``adapters/synthesize_voice/fish_s2_adapter.py``). Adapters without this
attribute (e.g. the default musubi-tuner render adapter, which is a one-shot
subprocess and already self-releases its VRAM) are no-ops here.

``core/workflow.py`` calls ``ensure_video_model_unloaded``/``prepare_video_model``
and ``ensure_video_model_unloaded``/``prepare_voice_model`` (the latter pair
reused for the voice/TTS slot — both functions are already adapter-agnostic)
at each phase boundary: before render_character, after image approval, and
before Phase-2 frame critique. ComfyUI-backed adapters
(``ComfyUIFluxAdapter``, ``LtxAdapter``) opt in via ``ComfyUIUnloadMixin``
below, whose ``unload()`` calls ComfyUI's own (otherwise unused) ``POST
/free`` — its ``load()`` is a no-op since ComfyUI lazy-loads whatever
checkpoint the next submitted workflow needs.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from core.observability import log_event

logger = logging.getLogger(__name__)

# Synthetic stage names emitted through the stage_hook so the dashboard events
# feed shows the model load instead of going silent.
VIDEO_MODEL_LOAD_STAGE = "video_model_load"
VOICE_MODEL_LOAD_STAGE = "voice_model_load"


def is_managed(adapter: Any) -> bool:
    # Strict `is True` so mock adapters (whose attributes are truthy MagicMocks)
    # and accidental truthy values don't opt in.
    return getattr(adapter, "managed_vram", False) is True


class ComfyUIUnloadMixin:
    """
    Shared managed-VRAM contract for adapters backed by a ComfyUI server
    (``ComfyUIFluxAdapter``, ``LtxAdapter``). ComfyUI already lazy-loads
    whatever checkpoint the next ``/prompt`` needs, so there's nothing to
    proactively "load" — only ``unload()`` does real work, via ComfyUI's
    real-but-otherwise-unused ``POST /free`` endpoint. Safe to call even when
    nothing is resident, and safe to call once per adapter instance even
    though two adapters (render's comfyui_flux, video's ltx) may share the
    same underlying ComfyUI process — ``/free`` is idempotent.
    """

    managed_vram = True

    async def load(self) -> None:
        return None

    async def wait_until_loaded(self, timeout_sec: float, poll_sec: float = 1.0) -> None:
        return None

    async def unload(self) -> bool:
        import httpx

        base_url = self._base_url  # type: ignore[attr-defined]
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{base_url}/free",
                    json={"unload_models": True, "free_memory": True},
                )
        except httpx.ConnectError:
            logger.warning(
                "ComfyUI unreachable at %s — assuming no model resident", base_url
            )
            return False
        if resp.status_code >= 400:
            raise RuntimeError(
                f"ComfyUI at {base_url} refused to free VRAM "
                f"({resp.status_code}): {resp.text[:500]}"
            )
        return True


def unload_ollama_model(base_url: str, model: str) -> None:
    """Tell Ollama to evict the model from VRAM (keep_alive=0) before GPU-heavy stages."""
    import urllib.request, json as _json
    ollama_base = base_url.replace("/v1", "").rstrip("/")
    try:
        data = _json.dumps({"model": model, "keep_alive": 0}).encode()
        req = urllib.request.Request(f"{ollama_base}/api/generate", data=data,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=10)
        logger.info("Unloaded %s from VRAM", model)
    except Exception as exc:
        logger.warning("Could not unload Ollama model (non-fatal): %s", exc)


async def ensure_video_model_unloaded(video_adapter: Any) -> None:
    """Make sure a VRAM-managed video model is out of VRAM before the render phase."""
    if not is_managed(video_adapter):
        return
    unloaded = await video_adapter.unload()
    log_event(
        logger,
        "video_model_unloaded",
        adapter=type(video_adapter).__name__,
        service_reachable=unloaded,
    )


async def _prepare_managed_adapter(
    adapter: Any,
    settings: Any,
    *,
    gap_sec: float,
    timeout_sec: float,
    stage_name: str,
    sleep: Callable[[float], Any],
    notify: Callable[[str, str], Any] | None,
) -> None:
    """
    Bring a VRAM-managed adapter's model into memory.

    Unloads the Ollama LLM/VLM first, waits ``gap_sec`` for VRAM to settle,
    then loads the adapter's model and polls until ready (``timeout_sec``
    ceiling). ``notify`` is the workflow stage_hook; it receives synthetic
    (stage_name, "stage_started"/"stage_completed") events so the dashboard
    shows progress during the load.
    """
    if not is_managed(adapter):
        return

    if notify:
        notify(stage_name, "stage_started")

    unload_ollama_model(settings.llm_base_url, settings.llm_model)
    if (settings.image_critique_base_url, settings.image_critique_model) != (
        settings.llm_base_url, settings.llm_model
    ):
        unload_ollama_model(settings.image_critique_base_url, settings.image_critique_model)

    logger.info("Waiting %ss for VRAM to settle before loading %s", gap_sec, type(adapter).__name__)
    await sleep(gap_sec)

    await adapter.load()
    await adapter.wait_until_loaded(timeout_sec)

    log_event(logger, "video_model_loaded", adapter=type(adapter).__name__)
    if notify:
        notify(stage_name, "stage_completed")


async def prepare_video_model(
    video_adapter: Any,
    settings: Any,
    *,
    sleep: Callable[[float], Any] = asyncio.sleep,
    notify: Callable[[str, str], Any] | None = None,
) -> None:
    """
    Bring a VRAM-managed video model into memory after the render phase.

    Unloads the Ollama LLM/VLM first, waits ``wan_load_gap_sec`` for VRAM to
    settle, then loads the video model and polls until it is ready
    (``wan_load_timeout_sec`` ceiling — the Wan load takes minutes).

    ``notify`` is the workflow stage_hook; it receives synthetic
    (VIDEO_MODEL_LOAD_STAGE, "stage_started"/"stage_completed") events so the
    dashboard shows progress during the load.
    """
    await _prepare_managed_adapter(
        video_adapter,
        settings,
        gap_sec=settings.wan_load_gap_sec,
        timeout_sec=settings.wan_load_timeout_sec,
        stage_name=VIDEO_MODEL_LOAD_STAGE,
        sleep=sleep,
        notify=notify,
    )


async def prepare_voice_model(
    voice_adapter: Any,
    settings: Any,
    *,
    sleep: Callable[[float], Any] = asyncio.sleep,
    notify: Callable[[str, str], Any] | None = None,
) -> None:
    """
    Bring a VRAM-managed voice (TTS) model into memory, e.g. Fish Audio S2.

    Same contract as ``prepare_video_model`` but reads ``fish_s2_load_gap_sec``
    / ``fish_s2_load_timeout_sec`` — Fish S2's load is a weights-from-disk load,
    not a multi-minute diffusion pipeline build, so both are much shorter.
    """
    await _prepare_managed_adapter(
        voice_adapter,
        settings,
        gap_sec=settings.fish_s2_load_gap_sec,
        timeout_sec=settings.fish_s2_load_timeout_sec,
        stage_name=VOICE_MODEL_LOAD_STAGE,
        sleep=sleep,
        notify=notify,
    )
