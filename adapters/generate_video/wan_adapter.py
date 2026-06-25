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
      GET  /health    → {"status": "ok"}
      POST /generate  → multipart/form-data:
                          image         (file, PNG),
                          prompt        (str),
                          duration_sec  (float),
                          fps           (int)
                        → raw MP4 bytes

    One instance per job — ``work_dir`` should be job-scoped.

    Args:
        work_dir:  Output directory. Each shot is saved to ``work_dir/<shot_id>/clip.mp4``.
        base_url:  Wan HTTP service root (e.g. http://localhost:8030).
        fps:       Requested output frame rate. Wan generates in multiples of 8 frames;
                   the service is responsible for rounding. 16 fps is a good quality/speed
                   trade-off for short children's clips.
    """

    version = "1.0.0"

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

        prompt = self._build_prompt(req.action)

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

    def _build_prompt(self, action: str) -> str:
        return f"{_PROMPT_PREFIX}, {action}, {_PROMPT_SUFFIX}"

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
