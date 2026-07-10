import logging
import wave
from pathlib import Path

from core.capabilities.base import LipSync
from core.models.capabilities import LipSyncRequest, VideoClip
from core.models.common import CostEstimate, HealthStatus
from core.observability import log_event

logger = logging.getLogger(__name__)


class LatentSyncAdapter(LipSync):
    """
    lip_sync adapter: audio-conditioned LatentSync repair for Wan I2V clips.

    This is used only when the selected video backend does not generate
    audio-synced motion natively. It is more expensive than MuseTalk and benefits
    from freeing any resident TTS model before inference.
    """

    version = "1.0.0"
    requires_voice_unloaded: bool = True

    def __init__(
        self,
        work_dir: Path,
        base_url: str = "http://localhost:8041",
        inference_steps: int = 20,
        guidance_scale: float = 1.5,
    ) -> None:
        self.work_dir = work_dir
        self._base_url = base_url.rstrip("/")
        self._inference_steps = inference_steps
        self._guidance_scale = guidance_scale

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
                reason=f"LatentSync service unreachable at {self._base_url}: {exc}",
            )

    async def estimate_cost(self, req: LipSyncRequest) -> CostEstimate:
        return CostEstimate(
            amount=0.0,
            notes="Self-hosted LatentSync; GPU compute cost via rented hardware.",
        )

    async def run(self, req: LipSyncRequest) -> VideoClip:
        video_path = Path(req.video_uri)
        audio_path = Path(req.audio_uri)
        self._check_inputs(video_path, audio_path, req.shot_id)

        out_dir = self.work_dir / req.shot_id
        out_dir.mkdir(parents=True, exist_ok=True)

        log_event(
            logger,
            "lip_sync_started",
            adapter="latentsync",
            shot_id=req.shot_id,
            video=str(video_path),
            audio=str(audio_path),
            inference_steps=self._inference_steps,
            guidance_scale=self._guidance_scale,
        )

        mp4_bytes = await self._call_latentsync(video_path, audio_path, req.shot_id)
        synced_path = self._save_clip(mp4_bytes, out_dir)
        duration = self._audio_duration(audio_path)

        log_event(
            logger,
            "lip_sync_completed",
            adapter="latentsync",
            shot_id=req.shot_id,
            synced=str(synced_path),
            duration_sec=duration,
        )
        return VideoClip(uri=str(synced_path), duration_sec=duration, shot_id=req.shot_id)

    def _check_inputs(self, video_path: Path, audio_path: Path, shot_id: str) -> None:
        if not video_path.exists():
            raise FileNotFoundError(
                f"Video clip not found for shot {shot_id}: {video_path}. "
                "generate_video must run before lip_sync."
            )
        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio track not found for shot {shot_id}: {audio_path}. "
                "synthesize_voice must run before lip_sync."
            )

    async def _call_latentsync(
        self, video_path: Path, audio_path: Path, shot_id: str
    ) -> bytes:
        import httpx

        async with httpx.AsyncClient(timeout=1200.0) as client:
            with video_path.open("rb") as vf, audio_path.open("rb") as af:
                resp = await client.post(
                    f"{self._base_url}/lipsync",
                    data={
                        "shot_id": shot_id,
                        "inference_steps": str(self._inference_steps),
                        "guidance_scale": str(self._guidance_scale),
                    },
                    files={
                        "video": (video_path.name, vf, "video/mp4"),
                        "audio": (audio_path.name, af, "audio/wav"),
                    },
                )
            if resp.status_code >= 400:
                logger.error("LatentSync service error %s: %s", resp.status_code, resp.text[:1000])
            resp.raise_for_status()
        return resp.content

    def _save_clip(self, mp4_bytes: bytes, out_dir: Path) -> Path:
        path = out_dir / "synced.mp4"
        path.write_bytes(mp4_bytes)
        return path

    def _audio_duration(self, audio_path: Path) -> float:
        try:
            with wave.open(str(audio_path)) as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            return 0.0
