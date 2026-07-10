import logging
from pathlib import Path

from core.capabilities.base import GenerateVideo
from core.models.capabilities import VideoClip, VideoRequest
from core.models.common import CostEstimate, HealthStatus
from core.observability import log_event

logger = logging.getLogger(__name__)

_PROMPT_PREFIX = "audio-synchronized singing performance video, natural mouth articulation"
_PROMPT_SUFFIX = "stable identity, expressive performance, high quality, cinematic lighting"
_DEFAULT_FPS = 16


def _infer_frames_for_duration(duration_sec: float, fps: int) -> int:
    """Wan-style frame counts use 4n+1, matching services/wan_server.py."""
    if duration_sec <= 0 or fps <= 0:
        return 0
    return 4 * max(1, round(duration_sec * fps / 4)) + 1


class WanS2VAdapter(GenerateVideo):
    """
    generate_video adapter: Wan2.2 Speech-to-Video via a thin local HTTP wrapper.

    Wan S2V is audio-conditioned, so the workflow must synthesize/slice audio
    before this adapter runs. The generated clip already contains synced mouth
    motion and the separate lip_sync stage is skipped.
    """

    version = "1.0.0"
    native_lipsync: bool = True
    requires_voice_unloaded: bool = True

    def __init__(
        self,
        work_dir: Path,
        base_url: str = "http://localhost:8031",
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
                reason=f"Wan S2V service unreachable at {self._base_url}: {exc}",
            )

    async def estimate_cost(self, req: VideoRequest) -> CostEstimate:
        return CostEstimate(
            amount=0.0,
            notes="Self-hosted Wan2.2-S2V; GPU compute cost via rented hardware.",
        )

    async def run(self, req: VideoRequest) -> VideoClip:
        image_path = Path(req.image_uri)
        audio_path = Path(req.audio_uri or "")
        self._check_inputs(image_path, audio_path, req.shot_id)

        out_dir = self.work_dir / req.shot_id
        out_dir.mkdir(parents=True, exist_ok=True)
        prompt = self._build_prompt(req.action, req.setting, req.style_suffix)

        log_event(
            logger,
            "generate_video_started",
            adapter="wan_s2v",
            shot_id=req.shot_id,
            image=str(image_path),
            audio=str(audio_path),
            duration_sec=req.duration_sec,
            prompt=prompt,
        )

        mp4_bytes = await self._call_wan_s2v(
            image_path=image_path,
            audio_path=audio_path,
            prompt=prompt,
            duration_sec=req.duration_sec,
            fps=self._fps,
            shot_id=req.shot_id,
        )
        clip_path = self._save_clip(mp4_bytes, out_dir)

        log_event(
            logger,
            "generate_video_completed",
            adapter="wan_s2v",
            shot_id=req.shot_id,
            clip=str(clip_path),
        )
        return VideoClip(uri=str(clip_path), duration_sec=req.duration_sec, shot_id=req.shot_id)

    def _check_inputs(self, image_path: Path, audio_path: Path, shot_id: str) -> None:
        if not image_path.exists():
            raise FileNotFoundError(
                f"Render image not found for shot {shot_id}: {image_path}. "
                "render_character must run before generate_video."
            )
        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio track not found for shot {shot_id}: {audio_path}. "
                "Wan S2V requires synthesize_voice/source audio before generate_video."
            )

    def _build_prompt(self, action: str, setting: str = "", style_suffix: str = "") -> str:
        prefix = _PROMPT_PREFIX
        if style_suffix.strip():
            prefix = f"{style_suffix.strip()}, {_PROMPT_PREFIX}"
        parts = [prefix, action]
        if setting.strip():
            parts.append(setting.strip())
        parts.append(_PROMPT_SUFFIX)
        return ", ".join(parts)

    async def _call_wan_s2v(
        self,
        *,
        image_path: Path,
        audio_path: Path,
        prompt: str,
        duration_sec: float,
        fps: int,
        shot_id: str,
    ) -> bytes:
        import httpx

        infer_frames = _infer_frames_for_duration(duration_sec, fps)

        async with httpx.AsyncClient(timeout=3600.0) as client:
            with image_path.open("rb") as img_file, audio_path.open("rb") as audio_file:
                resp = await client.post(
                    f"{self._base_url}/generate",
                    data={
                        "prompt": prompt,
                        "duration_sec": str(duration_sec),
                        "fps": str(fps),
                        "infer_frames": str(infer_frames) if infer_frames > 0 else "",
                        "shot_id": shot_id,
                    },
                    files={
                        "image": (image_path.name, img_file, "image/png"),
                        "audio": (audio_path.name, audio_file, "audio/wav"),
                    },
                )
            if resp.status_code >= 400:
                logger.error("Wan S2V service error %s: %s", resp.status_code, resp.text[:1000])
            resp.raise_for_status()
        return resp.content

    def _save_clip(self, mp4_bytes: bytes, out_dir: Path) -> Path:
        path = out_dir / "clip.mp4"
        path.write_bytes(mp4_bytes)
        return path
