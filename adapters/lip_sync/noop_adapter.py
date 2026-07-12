import shutil
import wave
from pathlib import Path

from core.capabilities.base import LipSync
from core.models.capabilities import LipSyncRequest, VideoClip
from core.models.common import CostEstimate, HealthStatus


class NoopLipSyncAdapter(LipSync):
    """Fast path for jobs where mouth alignment is intentionally skipped."""

    version = "1.0.0"

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir

    async def health(self) -> HealthStatus:
        return HealthStatus(status="ok")

    async def estimate_cost(self, req: LipSyncRequest) -> CostEstimate:
        return CostEstimate(amount=0.0, notes="Lip-sync repair skipped by request.")

    async def run(self, req: LipSyncRequest) -> VideoClip:
        source = Path(req.video_uri)
        if not source.exists():
            raise FileNotFoundError(f"Video for lip-sync skip not found: {source}")

        out_dir = self.work_dir / req.shot_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "synced.mp4"
        if source.resolve() != out_path.resolve():
            shutil.copyfile(source, out_path)
        return VideoClip(
            uri=str(out_path),
            duration_sec=self._audio_duration(Path(req.audio_uri)),
            shot_id=req.shot_id,
        )

    def _audio_duration(self, path: Path) -> float:
        try:
            with wave.open(str(path), "rb") as handle:
                frames = handle.getnframes()
                rate = handle.getframerate()
                return frames / float(rate) if rate else 0.0
        except Exception:
            return 0.0
