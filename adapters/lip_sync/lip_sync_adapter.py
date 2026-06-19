import logging
import wave
from pathlib import Path

from core.capabilities.base import LipSync
from core.models.capabilities import LipSyncRequest, VideoClip
from core.models.common import CostEstimate, HealthStatus
from core.observability import log_event

logger = logging.getLogger(__name__)


class LipSyncAdapter(LipSync):
    """
    lip_sync adapter: per-shot mouth alignment via a Wav2Lip-compatible HTTP API.

    Consumes outputs from two upstream stages:
      - generate_video  → video_uri  (MP4, Wan-generated clip)
      - synthesize_voice → audio_uri (WAV, per-speaker TTS output)

    The service must expose:
      GET  /health   → {"status": "ok"}
      POST /lipsync  → multipart/form-data:
                         video  (file, MP4),
                         audio  (file, WAV),
                         shot_id (str)
                       → raw MP4 bytes

    Output duration is read from the audio WAV header — the synced clip is
    trimmed/padded by the service to match the dialogue, so audio length
    is the authoritative duration.

    One instance per job — ``work_dir`` should be job-scoped.

    Args:
        work_dir:  Output directory. Each shot saved to ``work_dir/<shot_id>/synced.mp4``.
        base_url:  Lip-sync service root (e.g. http://localhost:8040).
    """

    version = "1.0.0"

    def __init__(
        self,
        work_dir: Path,
        base_url: str = "http://localhost:8040",
    ) -> None:
        self.work_dir = work_dir
        self._base_url = base_url.rstrip("/")

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
                reason=f"Lip-sync service unreachable at {self._base_url}: {exc}",
            )

    async def estimate_cost(self, req: LipSyncRequest) -> CostEstimate:
        return CostEstimate(
            amount=0.0,
            notes="Self-hosted lip-sync service; GPU compute cost via rented hardware.",
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
            shot_id=req.shot_id,
            video=str(video_path),
            audio=str(audio_path),
        )

        mp4_bytes = await self._call_lipsync(video_path, audio_path, req.shot_id)
        synced_path = self._save_clip(mp4_bytes, out_dir)
        duration = self._audio_duration(audio_path)

        log_event(
            logger,
            "lip_sync_completed",
            shot_id=req.shot_id,
            synced=str(synced_path),
            duration_sec=duration,
        )

        return VideoClip(
            uri=str(synced_path),
            duration_sec=duration,
            shot_id=req.shot_id,
        )

    # ------------------------------------------------------------------
    # Private helpers (mockable in tests)
    # ------------------------------------------------------------------

    def _check_inputs(
        self, video_path: Path, audio_path: Path, shot_id: str
    ) -> None:
        """Raise FileNotFoundError with a clear stage-ordering message if either input is absent."""
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

    async def _call_lipsync(
        self, video_path: Path, audio_path: Path, shot_id: str
    ) -> bytes:
        """POST to the lip-sync service; return raw MP4 bytes."""
        import httpx

        async with httpx.AsyncClient(timeout=300.0) as client:
            with video_path.open("rb") as vf, audio_path.open("rb") as af:
                resp = await client.post(
                    f"{self._base_url}/lipsync",
                    data={"shot_id": shot_id},
                    files={
                        "video": (video_path.name, vf, "video/mp4"),
                        "audio": (audio_path.name, af, "audio/wav"),
                    },
                )
            resp.raise_for_status()
        return resp.content

    def _save_clip(self, mp4_bytes: bytes, out_dir: Path) -> Path:
        """Write synced MP4 bytes to out_dir/synced.mp4 and return the path."""
        path = out_dir / "synced.mp4"
        path.write_bytes(mp4_bytes)
        return path

    def _audio_duration(self, audio_path: Path) -> float:
        """Read duration from the WAV header; return 0.0 if unreadable."""
        try:
            with wave.open(str(audio_path)) as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            return 0.0
