import logging
from pathlib import Path

from core.capabilities.base import GenerateVideo
from core.models.capabilities import VideoClip, VideoRequest
from core.models.common import CostEstimate, HealthStatus
from core.observability import log_event

logger = logging.getLogger(__name__)

_PROMPT_PREFIX = "cinematic image-to-video animation, natural motion"
_PROMPT_SUFFIX = "stable identity, high quality, clean background, no subtitles"
_DEFAULT_FPS = 16


def _infer_frames_for_duration(duration_sec: float, fps: int) -> int:
    if duration_sec <= 0 or fps <= 0:
        return 81
    return 4 * max(1, round(duration_sec * fps / 4)) + 1


class WanLightX2VAdapter(GenerateVideo):
    """
    Experimental Wan2.2 I2V adapter backed by LightX2V's 4-step distill LoRA path.

    This is a visual-motion backend, not native lip-sync. Pair it with
    lipsync_adapter=none for maximum throughput, or LatentSync/MuseTalk when
    mouth alignment still matters.
    """

    version = "0.1.0"
    managed_vram = True

    def __init__(
        self,
        work_dir: Path,
        base_url: str = "http://localhost:8032",
        fps: int = _DEFAULT_FPS,
    ) -> None:
        self.work_dir = work_dir
        self._base_url = base_url.rstrip("/")
        self._fps = fps

    async def health(self) -> HealthStatus:
        try:
            import httpx
        except ImportError:
            return HealthStatus(status="down", reason="httpx not installed. Run: pip install httpx")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/health")
                resp.raise_for_status()
            return HealthStatus(status="ok")
        except Exception as exc:
            return HealthStatus(
                status="down",
                reason=f"LightX2V Wan I2V service unreachable at {self._base_url}: {exc}",
            )

    async def load(self) -> None:
        import httpx

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{self._base_url}/load")
            resp.raise_for_status()

    async def unload(self) -> bool:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(f"{self._base_url}/unload")
        except httpx.ConnectError:
            logger.warning(
                "LightX2V Wan I2V service unreachable at %s; assuming no model resident",
                self._base_url,
            )
            return False
        if resp.status_code >= 400:
            raise RuntimeError(
                f"LightX2V Wan I2V service at {self._base_url} refused to unload "
                f"({resp.status_code}): {resp.text[:500]}"
            )
        return True

    async def wait_until_loaded(self, timeout_sec: float, poll_sec: float = 10.0) -> None:
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
                    raise RuntimeError(f"LightX2V Wan I2V failed to load: {body['error']}")
                await asyncio.sleep(poll_sec)
        raise TimeoutError(
            f"LightX2V Wan I2V not loaded after {timeout_sec:.0f}s. "
            "Check lightx2v_wan.log or raise VIDEO_ME_WAN_LOAD_TIMEOUT_SEC."
        )

    async def estimate_cost(self, req: VideoRequest) -> CostEstimate:
        return CostEstimate(
            amount=0.0,
            notes="Self-hosted LightX2V Wan2.2 I2V; GPU compute cost via rented hardware.",
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
        prompt = self._build_prompt(req.action, req.setting, req.style_suffix)
        infer_frames = _infer_frames_for_duration(req.duration_sec, self._fps)

        log_event(
            logger,
            "generate_video_started",
            adapter="wan_lightx2v",
            shot_id=req.shot_id,
            image=str(image_path),
            duration_sec=req.duration_sec,
            fps=self._fps,
            infer_frames=infer_frames,
            prompt=prompt,
        )

        mp4_bytes = await self._call_lightx2v(
            image_path=image_path,
            prompt=prompt,
            duration_sec=req.duration_sec,
            fps=self._fps,
            infer_frames=infer_frames,
            shot_id=req.shot_id,
        )
        clip_path = self._save_clip(mp4_bytes, out_dir)

        log_event(
            logger,
            "generate_video_completed",
            adapter="wan_lightx2v",
            shot_id=req.shot_id,
            clip=str(clip_path),
        )
        return VideoClip(uri=str(clip_path), duration_sec=req.duration_sec, shot_id=req.shot_id)

    def _build_prompt(self, action: str, setting: str = "", style_suffix: str = "") -> str:
        prefix = _PROMPT_PREFIX
        if style_suffix.strip():
            prefix = f"{style_suffix.strip()}, {prefix}"
        parts = [prefix, action]
        if setting.strip():
            parts.append(setting.strip())
        parts.append(_PROMPT_SUFFIX)
        return ", ".join(parts)

    async def _call_lightx2v(
        self,
        *,
        image_path: Path,
        prompt: str,
        duration_sec: float,
        fps: int,
        infer_frames: int,
        shot_id: str,
    ) -> bytes:
        import httpx

        async with httpx.AsyncClient(timeout=1800.0) as client:
            with image_path.open("rb") as img_file:
                resp = await client.post(
                    f"{self._base_url}/generate",
                    data={
                        "prompt": prompt,
                        "duration_sec": str(duration_sec),
                        "fps": str(fps),
                        "infer_frames": str(infer_frames),
                        "shot_id": shot_id,
                    },
                    files={"image": (image_path.name, img_file, "image/png")},
                )
            if resp.status_code >= 400:
                logger.error("LightX2V Wan service error %s: %s", resp.status_code, resp.text[:1000])
            resp.raise_for_status()
        return resp.content

    def _save_clip(self, mp4_bytes: bytes, out_dir: Path) -> Path:
        path = out_dir / "clip.mp4"
        path.write_bytes(mp4_bytes)
        return path
