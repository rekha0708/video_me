"""
GPU model sequencing between the render and video phases.

The Wan2.2 video model is too large to share the GPU with the render-phase
models (Flux subprocess + Ollama critique VLM) — loading both caused the
render_character OOM diagnosed in commit 58ce9d8. This module keeps a
VRAM-managed video adapter unloaded during the render phase and brings it up
only after image approval:

    unload Ollama → wait gap (default 30 s) → POST /load → poll until ready

Adapters opt in by setting a truthy ``managed_vram`` class attribute and
implementing ``load()``, ``unload()`` and ``wait_until_loaded()``
(see ``adapters/generate_video/wan_adapter.py``). The default LTX adapter has
no such attribute, so both entry points are no-ops on the default stack.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from core.observability import log_event

logger = logging.getLogger(__name__)

# Synthetic stage name emitted through the stage_hook so the dashboard events
# feed shows the multi-minute model load instead of going silent.
VIDEO_MODEL_LOAD_STAGE = "video_model_load"


def _is_managed(video_adapter: Any) -> bool:
    # Strict `is True` so mock adapters (whose attributes are truthy MagicMocks)
    # and accidental truthy values don't opt in.
    return getattr(video_adapter, "managed_vram", False) is True


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
    if not _is_managed(video_adapter):
        return
    unloaded = await video_adapter.unload()
    log_event(
        logger,
        "video_model_unloaded",
        adapter=type(video_adapter).__name__,
        service_reachable=unloaded,
    )


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
    if not _is_managed(video_adapter):
        return

    if notify:
        notify(VIDEO_MODEL_LOAD_STAGE, "stage_started")

    unload_ollama_model(settings.llm_base_url, settings.llm_model)
    if (settings.image_critique_base_url, settings.image_critique_model) != (
        settings.llm_base_url, settings.llm_model
    ):
        unload_ollama_model(settings.image_critique_base_url, settings.image_critique_model)

    gap = settings.wan_load_gap_sec
    logger.info("Waiting %ss for VRAM to settle before loading the video model", gap)
    await sleep(gap)

    await video_adapter.load()
    await video_adapter.wait_until_loaded(settings.wan_load_timeout_sec)

    log_event(logger, "video_model_loaded", adapter=type(video_adapter).__name__)
    if notify:
        notify(VIDEO_MODEL_LOAD_STAGE, "stage_completed")
