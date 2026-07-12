import asyncio
import logging
import shutil
from pathlib import Path

from core.capabilities.base import EnhanceVideo
from core.models.capabilities import VideoEnhanceRequest, VideoEnhanceResult
from core.models.common import CostEstimate, HealthStatus
from core.observability import log_event

logger = logging.getLogger(__name__)


class FfmpegVideoEnhanceAdapter(EnhanceVideo):
    """
    Standalone video_enhance adapter using ffmpeg scale/pad plus FPS interpolation.

    This is the baseline implementation for the separate enhancement pipeline.
    It intentionally does not use AI models; RIFE/FILM/latent adapters can
    later implement the same request/result contract.
    """

    version = "0.1.0"

    def __init__(self, work_dir: Path, ffmpeg_bin: str = "ffmpeg") -> None:
        self.work_dir = work_dir
        self._ffmpeg_bin = ffmpeg_bin

    async def health(self) -> HealthStatus:
        if not shutil.which(self._ffmpeg_bin):
            return HealthStatus(
                status="down",
                reason=f"ffmpeg not found on PATH: {self._ffmpeg_bin}",
            )
        return HealthStatus(status="ok")

    async def estimate_cost(self, req: VideoEnhanceRequest) -> CostEstimate:
        return CostEstimate(amount=0.0, notes="Local ffmpeg enhancement pass.")

    async def run(self, req: VideoEnhanceRequest) -> VideoEnhanceResult:
        input_path = Path(req.video_uri)
        if not input_path.exists():
            raise FileNotFoundError(f"Video to enhance not found: {input_path}")

        self.work_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.work_dir / req.output_name

        log_event(
            logger,
            "video_enhance_started",
            adapter="ffmpeg",
            input=str(input_path),
            output=str(output_path),
            target_width=req.target_width,
            target_height=req.target_height,
            target_fps=req.target_fps,
            interpolation=req.interpolation,
            stage=req.stage,
            has_burned_text=req.has_burned_text,
        )

        cmd = self._build_ffmpeg_args(input_path, output_path, req)
        await self._run_ffmpeg(cmd)

        log_event(
            logger,
            "video_enhance_completed",
            adapter="ffmpeg",
            output=str(output_path),
        )
        notes = [
            "ffmpeg baseline only; no AI super-resolution or AI frame interpolation",
        ]
        if req.has_burned_text:
            notes.append("input was marked as containing burned text/captions")
        return VideoEnhanceResult(
            video_uri=str(output_path),
            duration_sec=req.duration_sec,
            width=req.target_width,
            height=req.target_height,
            fps=float(req.target_fps),
            adapter="ffmpeg",
            notes=notes,
        )

    def _build_ffmpeg_args(
        self,
        input_path: Path,
        output_path: Path,
        req: VideoEnhanceRequest,
    ) -> list[str]:
        vf = self._filter(req)
        args = [
            self._ffmpeg_bin,
            "-y",
            "-i",
            str(input_path),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-preset",
            "medium",
        ]
        if req.preserve_audio:
            args += ["-c:a", "copy"]
        else:
            args += ["-an"]
        args.append(str(output_path))
        return args

    def _filter(self, req: VideoEnhanceRequest) -> str:
        scale_pad = (
            f"scale={req.target_width}:{req.target_height}:flags=lanczos"
            ":force_original_aspect_ratio=decrease,"
            f"pad={req.target_width}:{req.target_height}"
            ":(ow-iw)/2:(oh-ih)/2:color=black"
        )
        if req.interpolation == "minterpolate":
            return (
                f"minterpolate=fps={req.target_fps}:mi_mode=mci:mc_mode=aobmc"
                ":me_mode=bidir:vsbmc=1,"
                f"{scale_pad},setsar=1"
            )
        return f"fps={req.target_fps},{scale_pad},setsar=1"

    async def _run_ffmpeg(self, cmd: list[str]) -> None:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg video enhance failed ({proc.returncode}): "
                f"{stderr.decode(errors='replace')[-2000:]}"
            )
