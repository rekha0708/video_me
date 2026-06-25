import asyncio
import json
import logging
import shutil
from pathlib import Path

from core.capabilities.base import FetchMedia
from core.models.capabilities import FetchMediaRequest, FetchMediaResult
from core.models.common import CostEstimate, HealthStatus
from core.observability import log_event

logger = logging.getLogger(__name__)

_SYSTEM_TOOLS = ("yt-dlp", "ffmpeg", "ffprobe")

# Logged at ingest time as a required source-policy acknowledgement.
_TOS_NOTE = (
    "Downloading from third-party platforms may conflict with their Terms of Service. "
    "Ensure the source URL is permitted under the project source-link policy "
    "(own/licensed/public-domain/transformed content only)."
)


class YtDlpAdapter(FetchMedia):
    """
    Ingest adapter: download a video with yt-dlp, extract audio with ffmpeg.

    One instance per job — work_dir should be job-scoped (e.g. data_dir/media/<job_id>)
    so concurrent jobs don't collide.
    """

    version = "1.0.0"

    def __init__(self, work_dir: Path) -> None:
        self.work_dir = work_dir

    async def health(self) -> HealthStatus:
        missing = [t for t in _SYSTEM_TOOLS if shutil.which(t) is None]
        if missing:
            return HealthStatus(
                status="down",
                reason=f"Missing system tools: {', '.join(missing)}",
            )
        return HealthStatus(status="ok")

    async def estimate_cost(self, req: FetchMediaRequest) -> CostEstimate:
        return CostEstimate(amount=0.0, notes="yt-dlp and ffmpeg are free tools")

    async def run(self, req: FetchMediaRequest) -> FetchMediaResult:
        self.work_dir.mkdir(parents=True, exist_ok=True)

        log_event(
            logger,
            "fetch_media_tos_note",
            source_url=req.source_url,
            note=_TOS_NOTE,
        )

        info = await self._get_info(req.source_url)
        duration_sec = float(info.get("duration") or 0.0)

        video_path = await self._download(req.source_url)
        audio_path = await self._extract_audio(video_path)

        return FetchMediaResult(
            video_uri=str(video_path),
            audio_uri=str(audio_path),
            duration_sec=duration_sec,
            source_url=req.source_url,
        )

    # ------------------------------------------------------------------
    # Private helpers (mockable in unit tests)
    # ------------------------------------------------------------------

    async def _get_info(self, url: str) -> dict:
        """Fetch video metadata without downloading."""
        proc = await asyncio.create_subprocess_exec(
            "yt-dlp",
            "--dump-json",
            "--skip-download",
            "--no-playlist",
            "--js-runtimes", "node",
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"yt-dlp metadata fetch failed (exit {proc.returncode}): "
                f"{stderr.decode(errors='replace').strip()}"
            )
        return json.loads(stdout.decode())

    async def _download(self, url: str) -> Path:
        """Download best-quality video; merge to mp4."""
        proc = await asyncio.create_subprocess_exec(
            "yt-dlp",
            "--format", "bestvideo+bestaudio/best",
            "--merge-output-format", "mp4",
            "--no-playlist",
            "--js-runtimes", "node",
            "--output", str(self.work_dir / "video.%(ext)s"),
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"yt-dlp download failed (exit {proc.returncode}): "
                f"{stderr.decode(errors='replace').strip()}"
            )

        mp4s = sorted(self.work_dir.glob("video.mp4"))
        if not mp4s:
            raise RuntimeError(
                f"No video.mp4 found in {self.work_dir} after yt-dlp download."
            )
        return mp4s[0]

    async def _extract_audio(self, video_path: Path) -> Path:
        """Extract mono 44.1 kHz WAV — sufficient for Whisper transcription."""
        audio_path = self.work_dir / "audio.wav"
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-i", str(video_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "1",
            "-y",
            str(audio_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg audio extraction failed (exit {proc.returncode}): "
                f"{stderr.decode(errors='replace').strip()}"
            )
        return audio_path
