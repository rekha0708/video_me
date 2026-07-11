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
import time
from typing import Any, Callable

from core.observability import log_event

logger = logging.getLogger(__name__)

# Synthetic stage names emitted through the stage_hook so the dashboard events
# feed shows the model load instead of going silent.
VIDEO_MODEL_LOAD_STAGE = "video_model_load"
VOICE_MODEL_LOAD_STAGE = "voice_model_load"
VIDEO_MODEL_UNLOAD_STAGE = "video_model_unload"
VOICE_MODEL_UNLOAD_STAGE = "voice_model_unload"
FISH_S2_PROCESS_START_STAGE = "fish_s2_process_start"


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


async def free_comfyui(comfyui_base_url: str) -> bool:
    """Free ComfyUI's resident model unconditionally, independent of whether
    ComfyUI is even one of THIS job's own selected adapters.

    ensure_video_model_unloaded/prepare_*_model below only manage the current
    job's own render/video/voice adapter slots — a job running musubi_flux +
    wan never touches ComfyUI at all, so ComfyUI's resident LTX/Flux model
    left over from a *different, earlier* job's render_adapter=comfyui_flux or
    video_adapter=ltx choice is invisible to those hooks and never gets freed.
    That exact gap stranded ~50GB and OOM'd a later job's render. This talks
    to ComfyUI directly by URL, so it always sees it regardless of adapter
    selection. Best-effort: unreachable or already-idle is a no-op, not an
    error — call sites don't need to know or care whether ComfyUI is in use.
    """
    try:
        import httpx
    except ImportError:
        logger.info("httpx not installed — skipping best-effort ComfyUI free")
        return False
    url = f"{comfyui_base_url.rstrip('/')}/free"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json={"unload_models": True, "free_memory": True})
        if resp.status_code >= 400:
            logger.warning("ComfyUI refused to free VRAM (%s): %s", resp.status_code, resp.text[:300])
            return False
        return True
    except httpx.ConnectError:
        logger.info("ComfyUI unreachable at %s — assuming nothing resident", url)
        return False
    except Exception as exc:
        logger.warning("Could not free ComfyUI (non-fatal): %s", exc)
        return False


async def unload_wan(wan_base_url: str) -> bool:
    """Unload Wan's resident video model unconditionally, after every job.

    Nothing in the mainline pipeline ever unloads Wan after Phase B — the only
    call to ensure_video_model_unloaded(adapters.video) runs once, *before*
    Phase A, to keep Wan out of VRAM during image rendering. Once Phase B
    loads it, it stays resident (56GB+) through assemble_video/publish and
    beyond, for every job outcome (completed, failed, or cancelled) — the
    only thing that ever evicts it is the *next* job's own pre-Phase-A hook.
    Called from dashboard_worker's per-job `finally`, alongside
    stop_fish_s2_process(), so no job leaves Wan loaded for an indefinite
    number of unrelated jobs after it. Unlike Fish S2, Wan hasn't shown
    retention that /unload can't reclaim, so a plain HTTP call — not a
    process kill — is enough here. Best-effort: unreachable or already-idle
    is a no-op, not an error.
    """
    try:
        import httpx
    except ImportError:
        logger.info("httpx not installed — skipping best-effort Wan unload")
        return False
    url = f"{wan_base_url.rstrip('/')}/unload"
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url)
        if resp.status_code >= 400:
            logger.warning("Wan refused to unload (%s): %s", resp.status_code, resp.text[:300])
            return False
        return True
    except httpx.ConnectError:
        logger.info("Wan unreachable at %s — assuming nothing resident", url)
        return False
    except Exception as exc:
        logger.warning("Could not unload Wan (non-fatal): %s", exc)
        return False


FISH_S2_PROCESS_MATCH = "services.fish_s2_server:app"  # unique pgrep -f pattern


async def stop_fish_s2_process() -> bool:
    """Kill the Fish S2 server process entirely, not just POST /unload.

    Fish S2's own CUDA allocator retains memory across synthesis calls that
    /unload's torch.cuda.empty_cache() cannot reclaim — observed ~63GB
    resident after one job's worth of TTS calls vs ~20GB for a fresh process.
    A full process kill is the only reliable way to reset it to zero. Called
    unconditionally after every job (any outcome) so the next job — whichever
    adapters it uses — starts from a clean slate. Best-effort: no matching
    process is a normal no-op, not an error.
    """
    proc = await asyncio.create_subprocess_exec(
        "pkill", "-f", FISH_S2_PROCESS_MATCH,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()
    killed = proc.returncode == 0
    log_event(logger, "fish_s2_process_stopped", killed=killed)
    return killed


async def _fish_s2_reachable(base_url: str) -> bool:
    import httpx
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{base_url.rstrip('/')}/health")
        return resp.status_code == 200
    except Exception:
        return False


async def ensure_fish_s2_process_running(
    settings: Any,
    *,
    sleep: Callable[[float], Any] = asyncio.sleep,
    poll_sec: float = 2.0,
    notify: Callable[..., Any] | None = None,
) -> None:
    """Start the Fish S2 server process if it isn't already running, and wait
    for it to accept connections before returning.

    Pairs with stop_fish_s2_process(): the worker kills the process after
    every job, so the next job that needs voice synthesis must (re)spawn it
    on demand. Silent no-op (no notify) when a process is already listening —
    that's the common case and firing an event for it would just be noise;
    the dashboard only needs to know about an actual spawn-and-wait, which
    can take 120-150s cold (see fish_s2_load_timeout_sec below).

    Uses fish_s2_load_timeout_sec — the SAME setting _prepare_managed_adapter
    uses below for wait_until_loaded — deliberately, not a separate value.
    fish_s2_server.py's lifespan() loads the model eagerly at startup and
    blocks the whole ASGI app on it, so "process accepting connections" and
    "model fully loaded" are the same real-world event for this service, not
    two independent waits (see core/config.py's fish_s2_load_timeout_sec for
    the verified timing data this budget is based on).
    """
    if await _fish_s2_reachable(settings.fish_s2_base_url):
        return

    import os
    from pathlib import Path
    from urllib.parse import urlparse

    logger.info("Fish Audio S2 process not running — starting it")
    if notify:
        notify(
            FISH_S2_PROCESS_START_STAGE, "stage_started",
            message="Fish Audio S2 process not running — spawning it and waiting for it to accept connections…",
        )
    started = time.monotonic()
    repo_root = Path(__file__).resolve().parent.parent
    log_path = Path(settings.fish_s2_log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    port = urlparse(settings.fish_s2_base_url).port or 8025

    with open(log_path, "ab") as log_file:
        await asyncio.create_subprocess_exec(
            settings.fish_s2_venv_python,
            "services.fish_s2_server:app",
            "--host", "0.0.0.0",
            "--port", str(port),
            cwd=str(repo_root),
            env={**os.environ, "FISH_SPEECH_DIR": settings.fish_s2_speech_dir},
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
        )
        # create_subprocess_exec's fork+exec has already happened by the time
        # it returns, and the child holds its own dup'd fd — closing our copy
        # here (end of `with`) doesn't affect the detached child's logging,
        # the same fire-and-forget pattern `nohup ... > log 2>&1 &` uses.

    deadline_tries = max(1, int(settings.fish_s2_load_timeout_sec / poll_sec))
    for _ in range(deadline_tries):
        if await _fish_s2_reachable(settings.fish_s2_base_url):
            elapsed = time.monotonic() - started
            log_event(logger, "fish_s2_process_started")
            if notify:
                notify(
                    FISH_S2_PROCESS_START_STAGE, "stage_completed",
                    message=f"Fish Audio S2 process up and accepting connections (took {elapsed:.1f}s)",
                )
            return
        await sleep(poll_sec)
    raise TimeoutError(
        f"Fish Audio S2 process did not start within "
        f"{settings.fish_s2_load_timeout_sec}s. Check {log_path}."
    )


async def ensure_video_model_unloaded(
    video_adapter: Any,
    *,
    notify: Callable[..., Any] | None = None,
    stage_name: str = VIDEO_MODEL_UNLOAD_STAGE,
) -> None:
    """Make sure a VRAM-managed model is out of VRAM before the render phase.

    Despite the name, this is called for both the video and voice adapter
    slots (see core/workflow.py) — pass a distinct ``stage_name`` per call
    site (VIDEO_MODEL_UNLOAD_STAGE / VOICE_MODEL_UNLOAD_STAGE) so the
    dashboard timeline and cost summary can tell them apart. Previously this
    only logged to the process log, so the dashboard event feed never showed
    when a resident video/voice model was actually freed.
    """
    if not is_managed(video_adapter):
        return
    adapter_name = type(video_adapter).__name__
    if notify:
        notify(stage_name, "stage_started", message=f"Unloading {adapter_name} to free VRAM…")
    started = time.monotonic()
    unloaded = await video_adapter.unload()
    elapsed = time.monotonic() - started
    log_event(
        logger,
        "video_model_unloaded",
        adapter=adapter_name,
        service_reachable=unloaded,
    )
    if notify:
        status = "freed" if unloaded else "was already idle/unreachable (nothing to free)"
        notify(
            stage_name, "stage_completed",
            message=f"{adapter_name} {status} (took {elapsed:.1f}s)",
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

    adapter_name = type(adapter).__name__
    if notify:
        notify(
            stage_name, "stage_started",
            message=f"Loading {adapter_name} — freeing LLM VRAM, waiting {gap_sec:.0f}s, then loading model…",
        )
    started = time.monotonic()

    unload_ollama_model(settings.llm_base_url, settings.llm_model)
    if (settings.image_critique_base_url, settings.image_critique_model) != (
        settings.llm_base_url, settings.llm_model
    ):
        unload_ollama_model(settings.image_critique_base_url, settings.image_critique_model)

    logger.info("Waiting %ss for VRAM to settle before loading %s", gap_sec, adapter_name)
    await sleep(gap_sec)

    await adapter.load()
    await adapter.wait_until_loaded(timeout_sec)

    elapsed = time.monotonic() - started
    log_event(logger, "video_model_loaded", adapter=adapter_name)
    if notify:
        notify(
            stage_name, "stage_completed",
            message=f"{adapter_name} model loaded and ready (took {elapsed:.1f}s)",
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
    / ``fish_s2_load_timeout_sec`` — a plain weights-from-disk load rather than
    Wan's multi-minute diffusion pipeline build, so both are shorter, but not
    as short as "weights load" suggests: LLAMA + DAC decoder + warmup has been
    observed taking ~120-150s cold, hence the 240s default (not e.g. 60s).

    The worker kills the Fish S2 *process* (not just its model) after every
    job to reclaim VRAM the process itself can't release — see
    stop_fish_s2_process(). So before this job can load the model at all, the
    process needs to exist; ensure_fish_s2_process_running respawns it if the
    previous job's cleanup left nothing listening. Fish S2 is currently the
    only managed voice adapter, hence the direct coupling here rather than a
    generic per-adapter process hook.
    """
    if is_managed(voice_adapter):
        await ensure_fish_s2_process_running(settings, sleep=sleep, notify=notify)
    await _prepare_managed_adapter(
        voice_adapter,
        settings,
        gap_sec=settings.fish_s2_load_gap_sec,
        timeout_sec=settings.fish_s2_load_timeout_sec,
        stage_name=VOICE_MODEL_LOAD_STAGE,
        sleep=sleep,
        notify=notify,
    )
