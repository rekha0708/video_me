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

# Chart/diagram overlay panels: top offset (clears the y=20 disclosure text)
# and a sane cap on extra ffmpeg inputs.
_OVERLAY_Y: int = 80
_MAX_OVERLAY_INPUTS: int = 10

# Crossfade at clip boundaries — each 5-8s shot is sampled independently by the
# video model, so a hard cut can show a visible grey/brightness jump. Small
# enough to stay unnoticeable as a "transition" while smoothing the seam.
_DEFAULT_CROSSFADE_SEC: float = 0.3
_DEFAULT_TARGET_FPS: int = 24


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
        crossfade_sec: float = _DEFAULT_CROSSFADE_SEC,
        target_fps: int = _DEFAULT_TARGET_FPS,
        video_upscale_enabled: bool = False,
        upscale_target_fps: int = 48,
        upscale_interpolation: str = "minterpolate",
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
        self._crossfade_sec = crossfade_sec
        self._target_fps = target_fps
        self._caption_wrap_width = caption_wrap_width
        self._video_upscale_enabled = video_upscale_enabled
        self._upscale_target_fps = max(1, int(upscale_target_fps))
        self._upscale_interpolation = upscale_interpolation

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
        overlays = self._usable_overlays(req.overlays)

        transition = self._crossfade_transition(req)
        raw_duration = sum(c.duration_sec for c in req.clips)
        total_duration = raw_duration - transition * (len(req.clips) - 1)

        log_event(
            logger,
            "assemble_video_started",
            clip_count=len(req.clips),
            total_duration_sec=total_duration,
            disclosure_required=req.disclosure_label_required,
            overlay_count=len(overlays),
            crossfade_sec=transition,
            preserve_timing=req.preserve_timing,
            video_upscale_enabled=self._video_upscale_enabled,
            upscale_target_fps=(
                self._upscale_target_fps if self._video_upscale_enabled else None
            ),
        )

        cmd = self._build_ffmpeg_args(
            concat_file, audio_path, caption_file, output_path, req, overlays
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

    def _usable_overlays(self, overlays: list) -> list:
        """Cap the overlay input count and drop entries whose PNG is missing."""
        usable = []
        for ov in overlays:
            if not Path(ov.png_uri).exists():
                logger.warning("Overlay PNG missing for %s: %s — skipping", ov.shot_id, ov.png_uri)
                continue
            usable.append(ov)
        if len(usable) > _MAX_OVERLAY_INPUTS:
            logger.warning(
                "Capping overlays at %d (got %d) — later panels dropped",
                _MAX_OVERLAY_INPUTS, len(usable),
            )
            usable = usable[:_MAX_OVERLAY_INPUTS]
        return usable

    def _scale_pad_filter(self, *, require_fps: bool = False) -> str:
        scale_flags = ":flags=lanczos" if self._video_upscale_enabled else ""
        scale_pad = (
            f"scale={self._width}:{self._height}{scale_flags}"
            ":force_original_aspect_ratio=decrease,"
            f"pad={self._width}:{self._height}"
            ":(ow-iw)/2:(oh-ih)/2:color=black"
        )
        if self._video_upscale_enabled:
            fps = self._upscale_target_fps
            if self._upscale_interpolation == "minterpolate":
                return (
                    f"minterpolate=fps={fps}:mi_mode=mci:mc_mode=aobmc"
                    ":me_mode=bidir:vsbmc=1,"
                    f"{scale_pad},setsar=1"
                )
            return f"fps={fps},{scale_pad},setsar=1"
        if require_fps:
            return f"fps={self._target_fps},{scale_pad},setsar=1"
        return scale_pad

    def _build_filter(
        self,
        caption_file: Path,
        disclosure_required: bool,
        overlays: list = (),
        *,
        base_video_label: str = "[0:v]",
        overlay_input_offset: int = 2,
    ) -> str:
        """Build the video half of -filter_complex: scale+pad → overlay panels →
        caption → disclosure. base_video_label is the already-processed input
        (either the raw concat stream [0:v], or a crossfaded chain's output)."""
        scale_pad = self._scale_pad_filter()
        caption = (
            f"drawtext=textfile={caption_file}"
            f":fontsize={self._font_size}"
            f":fontcolor={self._font_color}"
            ":box=1:boxcolor=black@0.6:boxborderw=8"
            ":x=(w-tw)/2"
            f":y=h-th-{self._caption_margin}"
        )

        if overlays:
            # Chain: <base>scale+pad[base0]; [base0][N:v]overlay[base1]; ...; caption last.
            # Overlay inputs start at overlay_input_offset (legacy: 2 = concat + audio).
            stmts = [f"{base_video_label}{scale_pad}[base0]"]
            for i, ov in enumerate(overlays):
                stmts.append(
                    f"[base{i}][{overlay_input_offset + i}:v]overlay="
                    f"x=(W-w)/2:y={_OVERLAY_Y}"
                    f":enable='between(t,{ov.start_sec:.3f},{ov.end_sec:.3f})'"
                    f"[base{i + 1}]"
                )
            head = f"[base{len(overlays)}]{caption}"
            chain = ";".join(stmts) + ";" + head
        else:
            chain = f"{base_video_label}{scale_pad},{caption}"

        if disclosure_required:
            disclosure = (
                "drawtext=text='AI\\-Generated Content'"
                ":fontsize=28:fontcolor=white@0.8:x=20:y=20"
            )
            return f"{chain}[labeled];[labeled]{disclosure}[v]"

        return f"{chain}[v]"

    def _crossfade_transition(self, req: AssembleRequest) -> float:
        """Per-boundary crossfade duration, or 0.0 if crossfading isn't usable.

        Requires >=2 clips and a parallel per-shot audio_tracks list (so video
        xfade and audio acrossfade shrink the timeline by the exact same
        amount at the exact same points — otherwise dialogue gets clipped at
        the tail). Clamped so it never eats more than a fraction of the
        shortest clip.
        """
        if req.preserve_timing:
            return 0.0
        if self._crossfade_sec <= 0 or len(req.clips) < 2:
            return 0.0
        if len(req.audio_tracks) != len(req.clips):
            return 0.0
        shortest = min(c.duration_sec for c in req.clips)
        return max(0.0, min(self._crossfade_sec, shortest / 2 - 0.05))

    def _build_crossfade_video_chain(
        self, n: int, durations: list[float], transition: float,
    ) -> tuple[str, str]:
        """xfade chain over n pre-indexed [0:v]..[n-1:v] inputs. Returns
        (filter statements joined by ';', final output label)."""
        scale_pad = self._scale_pad_filter(require_fps=True)
        stmts = []
        labels = []
        for i in range(n):
            lbl = f"vs{i}"
            stmts.append(f"[{i}:v]{scale_pad}[{lbl}]")
            labels.append(lbl)

        prev = labels[0]
        cum = durations[0]
        for i in range(1, n):
            offset = max(cum - transition, 0.0)
            out = f"vx{i}"
            stmts.append(
                f"[{prev}][{labels[i]}]xfade=transition=fade"
                f":duration={transition:.3f}:offset={offset:.3f}[{out}]"
            )
            cum = cum + durations[i] - transition
            prev = out
        return ";".join(stmts), prev

    def _build_crossfade_audio_chain(
        self, n: int, input_offset: int, transition: float,
    ) -> tuple[str, str]:
        """acrossfade chain over n audio inputs starting at input_offset."""
        stmts = []
        prev = f"{input_offset}:a"
        for i in range(1, n):
            out = f"ax{i}"
            stmts.append(
                f"[{prev}][{input_offset + i}:a]acrossfade=d={transition:.3f}[{out}]"
            )
            prev = out
        return ";".join(stmts), prev

    def _build_ffmpeg_args(
        self,
        concat_file: Path,
        audio_path: Path,
        caption_file: Path,
        output_path: Path,
        req: AssembleRequest,
        overlays: list = (),
    ) -> list[str]:
        transition = self._crossfade_transition(req)

        if transition > 0:
            return self._build_crossfade_ffmpeg_args(
                caption_file, output_path, req, overlays, transition
            )

        vf = self._build_filter(caption_file, req.disclosure_label_required, overlays)
        args = [
            self._ffmpeg_bin, "-y",
            "-f", "concat", "-safe", "0", "-i", str(concat_file),
            "-i", str(audio_path),
        ]
        for ov in overlays:
            # overlay's default eof_action=repeat holds the single PNG frame for
            # the whole timeline; the enable window gates visibility.
            args += ["-i", str(ov.png_uri)]
        args += [
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
        return args

    def _build_crossfade_ffmpeg_args(
        self,
        caption_file: Path,
        output_path: Path,
        req: AssembleRequest,
        overlays: list,
        transition: float,
    ) -> list[str]:
        """One -i per clip + one -i per per-shot audio track, joined with
        xfade/acrossfade instead of a hard-cut concat demuxer."""
        n = len(req.clips)
        durations = [c.duration_sec for c in req.clips]

        video_stmts, video_label = self._build_crossfade_video_chain(n, durations, transition)
        audio_stmts, audio_label = self._build_crossfade_audio_chain(n, n, transition)

        caption_filter = self._build_filter(
            caption_file, req.disclosure_label_required, overlays,
            base_video_label=f"[{video_label}]",
            overlay_input_offset=2 * n,
        )
        vf = f"{video_stmts};{audio_stmts};{caption_filter}"

        args = [self._ffmpeg_bin, "-y"]
        for clip in req.clips:
            args += ["-i", str(Path(clip.uri).resolve())]
        for track in req.audio_tracks:
            args += ["-i", str(Path(track.uri).resolve())]
        for ov in overlays:
            args += ["-i", str(ov.png_uri)]
        args += [
            "-filter_complex", vf,
            "-map", "[v]",
            "-map", f"[{audio_label}]",
            "-c:v", self._video_codec,
            "-crf", str(self._crf),
            "-preset", "medium",
            "-c:a", self._audio_codec,
            "-b:a", "128k",
            "-movflags", "+faststart",
            "-shortest",
            str(output_path),
        ]
        return args

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
