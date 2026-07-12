"""Run the experimental standalone video_enhance pipeline on one local MP4."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
from pathlib import Path

from adapters.video_enhance.ai_adapter import AiVideoEnhanceAdapter
from adapters.video_enhance.ffmpeg_adapter import FfmpegVideoEnhanceAdapter
from core.models.capabilities import VideoEnhanceRequest


def _probe_duration_sec(path: Path, ffprobe_bin: str = "ffprobe") -> float:
    if not shutil.which(ffprobe_bin):
        return 0.0
    proc = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return 0.0
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enhance one existing MP4 with the experimental video_enhance pipeline."
    )
    parser.add_argument("input_video", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path(".local/video_enhance"))
    parser.add_argument("--output-name", default="enhanced.mp4")
    parser.add_argument(
        "--adapter",
        choices=[
            "ffmpeg",
            "rife",
            "film",
            "realesrgan_rife",
            "realesrgan_film",
            "latent_rife",
            "latent_film",
        ],
        default="ffmpeg",
    )
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--fps", type=int, default=48)
    parser.add_argument("--interpolation", choices=["fps", "minterpolate"], default="minterpolate")
    parser.add_argument("--stage", choices=["clip", "final"], default="clip")
    parser.add_argument("--duration-sec", type=float, default=None)
    parser.add_argument("--has-burned-text", action="store_true")
    parser.add_argument("--drop-audio", action="store_true")
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--realesrgan-python", default="/workspace/.venv_video_enhance/bin/python")
    parser.add_argument("--realesrgan-dir", type=Path, default=Path("/workspace/Real-ESRGAN"))
    parser.add_argument("--realesrgan-model", default="realesr-general-x4v3")
    parser.add_argument("--realesrgan-outscale", type=float, default=2.0)
    parser.add_argument("--realesrgan-tile", type=int, default=256)
    parser.add_argument("--rife-python", default="/workspace/.venv_video_enhance/bin/python")
    parser.add_argument("--rife-dir", type=Path, default=Path("/workspace/ECCV2022-RIFE"))
    parser.add_argument("--rife-model-dir", type=Path, default=Path("/workspace/ECCV2022-RIFE/train_log"))
    parser.add_argument("--film-python", default="/workspace/.venv_film/bin/python")
    parser.add_argument("--film-dir", type=Path, default=Path("/workspace/frame-interpolation"))
    parser.add_argument("--film-model-path", type=Path, default=Path("/workspace/FILM/film_net/Style/saved_model"))
    parser.add_argument("--film-times", type=int, default=2)
    parser.add_argument("--latent-workflow", type=Path, default=Path("assets/comfyui_workflows/video_enhance_latent.json"))
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    duration_sec = (
        args.duration_sec
        if args.duration_sec is not None
        else _probe_duration_sec(args.input_video)
    )
    if args.adapter == "ffmpeg":
        adapter = FfmpegVideoEnhanceAdapter(args.work_dir, ffmpeg_bin=args.ffmpeg_bin)
    else:
        adapter = AiVideoEnhanceAdapter(
            backend=args.adapter,
            work_dir=args.work_dir,
            ffmpeg_bin=args.ffmpeg_bin,
            ffprobe_bin=args.ffprobe_bin,
            realesrgan_python=args.realesrgan_python,
            realesrgan_dir=args.realesrgan_dir,
            realesrgan_model=args.realesrgan_model,
            realesrgan_outscale=args.realesrgan_outscale,
            realesrgan_tile=args.realesrgan_tile,
            rife_python=args.rife_python,
            rife_dir=args.rife_dir,
            rife_model_dir=args.rife_model_dir,
            film_python=args.film_python,
            film_dir=args.film_dir,
            film_model_path=args.film_model_path,
            film_times_to_interpolate=args.film_times,
            latent_workflow=args.latent_workflow,
        )
    health = await adapter.health()
    if health.status == "down":
        raise RuntimeError(f"video_enhance adapter is down: {health.reason}")
    result = await adapter.run(
        VideoEnhanceRequest(
            video_uri=str(args.input_video),
            duration_sec=duration_sec,
            output_name=args.output_name,
            target_width=args.width,
            target_height=args.height,
            target_fps=args.fps,
            interpolation=args.interpolation,
            stage=args.stage,
            has_burned_text=args.has_burned_text,
            preserve_audio=not args.drop_audio,
        )
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
