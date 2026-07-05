import logging
from pathlib import Path

from core.capabilities.base import GenerateVideo
from core.models.capabilities import VideoClip, VideoRequest
from core.models.common import CostEstimate, HealthStatus
from core.observability import log_event

logger = logging.getLogger(__name__)

# Style wrapper applied around the LLM-authored action description.
_PROMPT_PREFIX = "children's cartoon animation, vibrant colors"
_PROMPT_SUFFIX = "smooth motion, expressive, high quality"

# Default frames-per-second for generated clips.
_DEFAULT_FPS: int = 16


class WanAdapter(GenerateVideo):
    """
    generate_video adapter: per-shot image-to-video via a Wan 2.7-compatible HTTP API.

    The service must expose:
      GET  /health    → 200 {"status": "ok", "model_loaded": bool, ...}
      POST /load      → 200/202, kicks off a background model load
      POST /unload    → 200, releases the model's VRAM (409 while a load is running)
      POST /generate  → multipart/form-data:
                          image         (file, PNG),
                          prompt        (str),
                          duration_sec  (float),
                          fps           (int)
                        → raw MP4 bytes

    The Wan model is huge, so ``core.gpu_sequencer`` keeps it unloaded during the
    render_character phase (``managed_vram = True`` opts this adapter in) and loads
    it only right before the video phase.

    One instance per job — ``work_dir`` should be job-scoped.

    Args:
        work_dir:  Output directory. Each shot is saved to ``work_dir/<shot_id>/clip.mp4``.
        base_url:  Wan HTTP service root (e.g. http://localhost:8030).
        fps:       Requested output frame rate. Wan generates in multiples of 8 frames;
                   the service is responsible for rounding. 16 fps is a good quality/speed
                   trade-off for short children's clips.
    """

    version = "1.0.0"

    # The GPU sequencer loads/unloads this adapter's model around the render phase.
    managed_vram = True

    def __init__(
        self,
        work_dir: Path,
        base_url: str = "http://localhost:8030",
        fps: int = _DEFAULT_FPS,
    ) -> None:
        self.work_dir = work_dir
        self._base_url = base_url.rstrip("/")
        self._fps = fps

    async def health(self) -> HealthStatus:
        try:
            import httpx
        except ImportError:
            return HealthStatus(
                status="down",
                reason="httpx not installed. Run: pip install httpx",
            )
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/health")
                resp.raise_for_status()
            return HealthStatus(status="ok")
        except Exception as exc:
            return HealthStatus(
                status="down",
                reason=f"Wan service unreachable at {self._base_url}: {exc}",
            )

    # ------------------------------------------------------------------
    # VRAM lifecycle (called by core.gpu_sequencer around the render phase)
    # ------------------------------------------------------------------

    async def load(self) -> None:
        """Ask the service to start loading the model (non-blocking on the server)."""
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{self._base_url}/load")
            resp.raise_for_status()

    async def unload(self) -> bool:
        """
        Release the model's VRAM. Returns False if the service is unreachable
        (nothing resident, safe to continue). Raises if the service is up but
        refused to unload — proceeding would OOM the render phase.
        """
        import httpx

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(f"{self._base_url}/unload")
        except httpx.ConnectError:
            logger.warning(
                "Wan service unreachable at %s — assuming no model resident", self._base_url
            )
            return False
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Wan service at {self._base_url} refused to unload "
                f"({resp.status_code}): {resp.text[:500]}"
            )
        return True

    async def wait_until_loaded(self, timeout_sec: float, poll_sec: float = 10.0) -> None:
        """Poll /health until the model is loaded. Raises on load error or timeout."""
        import asyncio
        import time

        import httpx

        deadline = time.monotonic() + timeout_sec
        async with httpx.AsyncClient(timeout=10.0) as client:
            while time.monotonic() < deadline:
                resp = await client.get(f"{self._base_url}/health")
                resp.raise_for_status()
                body = resp.json()
                if body.get("model_loaded"):
                    return
                if body.get("error"):
                    raise RuntimeError(f"Wan model failed to load: {body['error']}")
                await asyncio.sleep(poll_sec)
        raise TimeoutError(
            f"Wan model not loaded after {timeout_sec:.0f}s. Check the wan_server logs; "
            "raise VIDEO_ME_WAN_LOAD_TIMEOUT_SEC if the load is just slow."
        )

    async def estimate_cost(self, req: VideoRequest) -> CostEstimate:
        return CostEstimate(
            amount=0.0,
            notes="Self-hosted Wan 2.7; GPU compute cost via rented hardware.",
        )

    async def run(self, req: VideoRequest) -> VideoClip:
        image_path = Path(req.image_uri)
        if not image_path.exists():
            raise FileNotFoundError(
                f"Render image not found for shot {req.shot_id}: {image_path}. "
                "render_character must run before generate_video."
            )

        out_dir = self.work_dir / req.shot_id
        out_dir.mkdir(parents=True, exist_ok=True)

        prompt = self._build_prompt(req.action, req.setting)

        log_event(
            logger,
            "generate_video_started",
            shot_id=req.shot_id,
            image=str(image_path),
            duration_sec=req.duration_sec,
            prompt=prompt,
        )

        mp4_bytes = await self._call_wan(image_path, prompt, req.duration_sec)
        clip_path = self._save_clip(mp4_bytes, out_dir)

        log_event(
            logger,
            "generate_video_completed",
            shot_id=req.shot_id,
            clip=str(clip_path),
        )

        return VideoClip(
            uri=str(clip_path),
            duration_sec=req.duration_sec,
            shot_id=req.shot_id,
        )

    # ------------------------------------------------------------------
    # Private helpers (mockable in tests)
    # ------------------------------------------------------------------

    def _build_prompt(self, action: str, setting: str = "") -> str:
        # Order: style → subject/action → environment → quality. The setting
        # steers lighting/scene mood; the first-frame image still dominates.
        parts = [_PROMPT_PREFIX, action]
        if setting.strip():
            parts.append(setting.strip())
        parts.append(_PROMPT_SUFFIX)
        return ", ".join(parts)

    async def _call_wan(
        self, image_path: Path, prompt: str, duration_sec: float
    ) -> bytes:
        """POST to the Wan service; return raw MP4 bytes."""
        import httpx

        async with httpx.AsyncClient(timeout=1900.0) as client:
            with image_path.open("rb") as img_file:
                resp = await client.post(
                    f"{self._base_url}/generate",
                    data={
                        "prompt": prompt,
                        "duration_sec": str(duration_sec),
                        "fps": str(self._fps),
                    },
                    files={"image": (image_path.name, img_file, "image/png")},
                )
            if resp.status_code >= 400:
                logger.error(
                    "Wan service error %s: %s",
                    resp.status_code,
                    resp.text[:1000],
                )
            resp.raise_for_status()
        return resp.content

    def _save_clip(self, mp4_bytes: bytes, out_dir: Path) -> Path:
        """Write MP4 bytes to out_dir/clip.mp4 and return the path."""
        path = out_dir / "clip.mp4"
        path.write_bytes(mp4_bytes)
        return path
