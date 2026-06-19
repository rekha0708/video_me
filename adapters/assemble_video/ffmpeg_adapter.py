import asyncio
import logging
import shutil
import textwrap
from pathlib import Path

from core.capabilities.base import AssembleVideo
from core.models.capabilities import AssembleRequest, FinalVideo
from core.models.common import CostEstimate, HealthStatus
from core.observability import log_event

logger = logging.getLogger(__name__)

# Output resolution for 9:16 portrait (YouTube Shorts / TikTok).
_DEFAULT_WIDTH: int = 1080
_DEFAULT_HEIGHT: int = 1920

# Approx characters per line at fontsize 50 on a 1080px canvas.
_DEFAULT_WRAP_WIDTH: int = 38

# Pixels from bottom edge where caption baseline sits.
_DEFAULT_CAPTION_MARGIN: int = 100


class FfmpegAssembleAdapter(AssembleVideo):
    """
    assemble_video adapter: stitch synced clips → add audio → burn captions
    → export 9:16 MP4 using ffmpeg.

    Pipeline:
      1. Write a concat list (ffmpeg concat demuxer).
      2. Write caption text to a file (avoids shell-quoting issues in drawtext).
      3. Build a -filter_complex chain: scale+pad to 9:16 → caption → optional
         AI-disclosure label.
      4. Run ffmpeg, replacing the audio stream with the provided AudioTrack.
      5. Output ``work_dir/final.mp4``.

    Requires ffmpeg with libx264 and aac (standard installs include both).

    Args:
        work_dir:            Output directory (job-scoped).
        ffmpeg_bin:          Path to the ffmpeg binary (default: "ffmpeg" on $PATH).
        width / height:      Output canvas size in pixels. Default: 1080×1920 (9:16).
        video_codec:         libx264 (default) or libx265.
        audio_codec:         aac (default).
        crf:                 Constant-rate factor for libx264 (18 = high quality, 28 = small).
        font_size:           Caption font size in pixels.
        font_color:          Caption font colour (ffmpeg colour name or hex).
        caption_margin:      Pixels from the bottom edge for the caption baseline.
        caption_wrap_width:  Characters per line before wrapping.
    """

    version = "1.0.0"

    def __init__(
        self,
        work_dir: Path,
        ffmpeg_bin: str = "ffmpeg",
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        video_codec: str = "libx264",
        audio_codec: str = "aac",
        crf: int = 23,
        font_size: int = 50,
        font_color: str = "white",
        caption_margin: int = _DEFAULT_CAPTION_MARGIN,
        caption_wrap_width: int = _DEFAULT_WRAP_WIDTH,
    ) -> None:
        self.work_dir = work_dir
        self._ffmpeg_bin = ffmpeg_bin
        self._width = width
        self._height = height
        self._video_codec = video_codec
        self._audio_codec = audio_codec
        self._crf = crf
        self._font_size = font_size
        self._font_color = font_color
        self._caption_margin = caption_margin
        self._caption_wrap_width = caption_wrap_width

    async def health(self) -> HealthStatus:
        if not shutil.which(self._ffmpeg_bin):
            return HealthStatus(
                status="down",
                reason=f"ffmpeg not found on PATH: {self._ffmpeg_bin}",
            )
        try:
            proc = await asyncio.create_subprocess_exec(
                self._ffmpeg_bin, "-version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            if proc.returncode != 0:
                return HealthStatus(status="down", reason="ffmpeg -version returned non-zero")
            return HealthStatus(status="ok")
        except Exception as exc:
            return HealthStatus(status="down", reason=f"ffmpeg error: {exc}")

    async def estimate_cost(self, req: AssembleRequest) -> CostEstimate:
        return CostEstimate(amount=0.0, notes="Local ffmpeg; no per-call cost.")

    async def run(self, req: AssembleRequest) -> FinalVideo:
        self._check_clips(req.clips)

        self.work_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.work_dir / "final.mp4"

        concat_file = self._write_concat_list(req.clips, self.work_dir)
        caption_file = self._write_caption_file(req.caption_text, self.work_dir)
        audio_path = Path(req.audio.uri)

        total_duration = sum(c.duration_sec for c in req.clips)

        log_event(
            logger,
            "assemble_video_started",
            clip_count=len(req.clips),
            total_duration_sec=total_duration,
            disclosure_required=req.disclosure_label_required,
        )

        cmd = self._build_ffmpeg_args(
            concat_file, audio_path, caption_file, output_path, req
        )
        await self._run_ffmpeg(cmd)

        log_event(
            logger,
            "assemble_video_completed",
            output=str(output_path),
            duration_sec=total_duration,
        )

        return FinalVideo(uri=str(output_path), duration_sec=total_duration)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_clips(self, clips: list) -> None:
        for clip in clips:
            if not Path(clip.uri).exists():
                raise FileNotFoundError(
                    f"Clip not found: {clip.uri} (shot_id={clip.shot_id}). "
                    "lip_sync must complete before assemble_video."
                )

    def _write_concat_list(self, clips: list, work_dir: Path) -> Path:
        """Write ffmpeg concat demuxer file with absolute paths."""
        lines = [f"file '{Path(c.uri).resolve()}'" for c in clips]
        path = work_dir / "concat.txt"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _write_caption_file(self, caption_text: str, work_dir: Path) -> Path:
        """Write word-wrapped caption text; drawtext reads it via textfile=."""
        wrapped = textwrap.fill(caption_text, width=self._caption_wrap_width)
        path = work_dir / "caption.txt"
        path.write_text(wrapped, encoding="utf-8")
        return path

    def _build_filter(
        self, caption_file: Path, disclosure_required: bool
    ) -> str:
        """Build the -filter_complex string: scale+pad → caption → optional disclosure."""
        scale_pad = (
            f"scale={self._width}:{self._height}"
            ":force_original_aspect_ratio=decrease,"
            f"pad={self._width}:{self._height}"
            ":(ow-iw)/2:(oh-ih)/2:color=black"
        )
        caption = (
            f"drawtext=textfile={caption_file}"
            f":fontsize={self._font_size}"
            f":fontcolor={self._font_color}"
            ":box=1:boxcolor=black@0.6:boxborderw=8"
            ":x=(w-tw)/2"
            f":y=h-th-{self._caption_margin}"
        )

        if disclosure_required:
            disclosure = (
                "drawtext=text='AI\\-Generated Content'"
                ":fontsize=28:fontcolor=white@0.8:x=20:y=20"
            )
            return (
                f"[0:v]{scale_pad},{caption}[labeled]"
                f";[labeled]{disclosure}[v]"
            )

        return f"[0:v]{scale_pad},{caption}[v]"

    def _build_ffmpeg_args(
        self,
        concat_file: Path,
        audio_path: Path,
        caption_file: Path,
        output_path: Path,
        req: AssembleRequest,
    ) -> list[str]:
        vf = self._build_filter(caption_file, req.disclosure_label_required)
        return [
            self._ffmpeg_bin, "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-i", str(audio_path),
            "-filter_complex", vf,
            "-map", "[v]",
            "-map", "1:a",
            "-c:v", self._video_codec,
            "-crf", str(self._crf),
            "-preset", "medium",
            "-c:a", self._audio_codec,
            "-b:a", "128k",
            "-movflags", "+faststart",
            "-shortest",
            str(output_path),
        ]

    async def _run_ffmpeg(self, cmd: list[str]) -> None:
        """Run ffmpeg; raise RuntimeError with stderr tail if it exits non-zero."""
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            tail = stderr.decode(errors="replace")[-2000:]
            raise RuntimeError(
                f"ffmpeg exited {proc.returncode}:\n{tail}"
            )
