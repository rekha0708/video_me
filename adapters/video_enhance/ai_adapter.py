import asyncio
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any, Literal

from core.capabilities.base import EnhanceVideo
from core.models.capabilities import VideoEnhanceRequest, VideoEnhanceResult
from core.models.common import CostEstimate, HealthStatus
from core.observability import log_event

logger = logging.getLogger(__name__)

AiEnhanceBackend = Literal[
    "rife",
    "film",
    "realesrgan_rife",
    "realesrgan_film",
    "latent_rife",
    "latent_film",
]

_REALESRGAN_WEIGHTS = {
    "RealESRGAN_x4plus": ["RealESRGAN_x4plus.pth"],
    "RealESRNet_x4plus": ["RealESRNet_x4plus.pth"],
    "RealESRGAN_x4plus_anime_6B": ["RealESRGAN_x4plus_anime_6B.pth"],
    "RealESRGAN_x2plus": ["RealESRGAN_x2plus.pth"],
    "realesr-animevideov3": ["realesr-animevideov3.pth"],
    "realesr-general-x4v3": [
        "realesr-general-x4v3.pth",
        "realesr-general-wdn-x4v3.pth",
    ],
}


class AiVideoEnhanceAdapter(EnhanceVideo):
    """
    Experimental AI video enhancement pipeline.

    Supported production path:
      Real-ESRGAN video restoration -> RIFE or FILM interpolation -> final
      ffmpeg normalization/audio reattach.

    Latent backends are wired through a ComfyUI workflow file, but the workflow
    itself must be supplied and validated separately. Health reports down when
    the workflow is missing.
    """

    version = "0.2.0"

    def __init__(
        self,
        *,
        backend: AiEnhanceBackend,
        work_dir: Path,
        ffmpeg_bin: str = "ffmpeg",
        ffprobe_bin: str = "ffprobe",
        realesrgan_python: str = "/workspace/.venv_video_enhance/bin/python",
        realesrgan_dir: Path = Path("/workspace/Real-ESRGAN"),
        realesrgan_model: str = "realesr-general-x4v3",
        realesrgan_outscale: float = 2.0,
        realesrgan_tile: int = 256,
        rife_python: str = "/workspace/.venv_video_enhance/bin/python",
        rife_dir: Path = Path("/workspace/ECCV2022-RIFE"),
        rife_model_dir: Path = Path("/workspace/ECCV2022-RIFE/train_log"),
        rife_exp: int = 1,
        rife_scale: float = 1.0,
        rife_fp16: bool = True,
        film_python: str = "/workspace/.venv_film/bin/python",
        film_dir: Path = Path("/workspace/frame-interpolation"),
        film_model_path: Path = Path("/workspace/FILM/film_net/Style/saved_model"),
        film_times_to_interpolate: int = 2,
        film_block_height: int = 1,
        film_block_width: int = 1,
        comfyui_base_url: str = "http://localhost:8188",
        latent_workflow: Path = Path("assets/comfyui_workflows/video_enhance_latent.json"),
    ) -> None:
        self.backend = backend
        self.work_dir = work_dir
        self._ffmpeg_bin = ffmpeg_bin
        self._ffprobe_bin = ffprobe_bin
        self._realesrgan_python = realesrgan_python
        self._realesrgan_dir = realesrgan_dir
        self._realesrgan_model = realesrgan_model
        self._realesrgan_outscale = realesrgan_outscale
        self._realesrgan_tile = realesrgan_tile
        self._rife_python = rife_python
        self._rife_dir = rife_dir
        self._rife_model_dir = rife_model_dir
        self._rife_exp = rife_exp
        self._rife_scale = rife_scale
        self._rife_fp16 = rife_fp16
        self._film_python = film_python
        self._film_dir = film_dir
        self._film_model_path = film_model_path
        self._film_times_to_interpolate = film_times_to_interpolate
        self._film_block_height = film_block_height
        self._film_block_width = film_block_width
        self._comfyui_base_url = comfyui_base_url.rstrip("/")
        self._latent_workflow = latent_workflow

    async def health(self) -> HealthStatus:
        missing = self._missing_common()
        if self._uses_realesrgan():
            missing.extend(self._missing_realesrgan())
        if self._uses_rife():
            missing.extend(self._missing_rife())
        if self._uses_film():
            missing.extend(self._missing_film())
        if self._uses_latent():
            missing.extend(self._missing_latent())
        if missing:
            return HealthStatus(status="down", reason="; ".join(missing))
        if self._uses_latent():
            try:
                import httpx

                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{self._comfyui_base_url}/system_stats")
                    resp.raise_for_status()
            except Exception as exc:
                return HealthStatus(
                    status="down",
                    reason=f"ComfyUI unreachable for latent enhance at {self._comfyui_base_url}: {exc}",
                )
        return HealthStatus(status="ok")

    async def estimate_cost(self, req: VideoEnhanceRequest) -> CostEstimate:
        return CostEstimate(
            amount=0.0,
            notes=f"Local AI video enhance backend: {self.backend}.",
        )

    async def run(self, req: VideoEnhanceRequest) -> VideoEnhanceResult:
        input_path = Path(req.video_uri)
        if not input_path.exists():
            raise FileNotFoundError(f"Video to enhance not found: {input_path}")

        self.work_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.work_dir / req.output_name
        scratch_dir = self.work_dir / f"{output_path.stem}_scratch"
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir)
        scratch_dir.mkdir(parents=True, exist_ok=True)

        started = time.perf_counter()
        current = input_path
        notes: list[str] = []

        log_event(
            logger,
            "video_enhance_started",
            adapter=self.backend,
            input=str(input_path),
            output=str(output_path),
            target_width=req.target_width,
            target_height=req.target_height,
            target_fps=req.target_fps,
            interpolation=req.interpolation,
            stage=req.stage,
        )

        if self._uses_realesrgan():
            current = await self._run_realesrgan(current, scratch_dir)
            notes.append(
                f"Real-ESRGAN super-resolution model={self._realesrgan_model} "
                f"outscale={self._realesrgan_outscale:g}"
            )

        if self._uses_latent():
            current = await self._run_latent_comfyui(req, current, scratch_dir)
            notes.append("latent two-pass ComfyUI workflow applied before interpolation")

        if self._uses_rife():
            current = await self._run_rife(req, current, scratch_dir)
            notes.append(f"RIFE interpolation to {req.target_fps} fps")

        if self._uses_film():
            current = await self._run_film(req, current, scratch_dir)
            notes.append(
                f"FILM interpolation times={self._film_times_to_interpolate}; "
                f"normalized to {req.target_fps} fps"
            )

        await self._normalize_output(input_path, current, output_path, req)
        elapsed = time.perf_counter() - started

        log_event(
            logger,
            "video_enhance_completed",
            adapter=self.backend,
            output=str(output_path),
            elapsed_sec=round(elapsed, 3),
        )
        notes.append(f"elapsed_sec={elapsed:.2f}")
        if req.has_burned_text:
            notes.append("input was marked as containing burned text/captions")
        return VideoEnhanceResult(
            video_uri=str(output_path),
            duration_sec=req.duration_sec,
            width=req.target_width,
            height=req.target_height,
            fps=float(req.target_fps),
            adapter=self.backend,
            notes=notes,
        )

    def _missing_common(self) -> list[str]:
        missing = []
        for label, cmd in (("ffmpeg", self._ffmpeg_bin), ("ffprobe", self._ffprobe_bin)):
            if not shutil.which(cmd):
                missing.append(f"{label} not found on PATH: {cmd}")
        return missing

    def _missing_realesrgan(self) -> list[str]:
        missing = self._missing_python("Real-ESRGAN", self._realesrgan_python)
        script = self._realesrgan_dir / "inference_realesrgan_video.py"
        if not script.exists():
            missing.append(f"Real-ESRGAN script missing: {script}")
        weight_names = _REALESRGAN_WEIGHTS.get(self._realesrgan_model, [])
        if not weight_names:
            missing.append(f"Unsupported Real-ESRGAN model: {self._realesrgan_model}")
        for name in weight_names:
            weight = self._realesrgan_dir / "weights" / name
            if not weight.exists():
                missing.append(f"Real-ESRGAN weight missing: {weight}")
        return missing

    def _missing_rife(self) -> list[str]:
        missing = self._missing_python("RIFE", self._rife_python)
        script = self._rife_dir / "inference_video.py"
        if not script.exists():
            missing.append(f"RIFE script missing: {script}")
        if not self._rife_model_dir.exists():
            missing.append(f"RIFE model dir missing: {self._rife_model_dir}")
        elif not any(self._rife_model_dir.glob("*.pkl")):
            missing.append(f"RIFE model weights missing in {self._rife_model_dir} (*.pkl)")
        return missing

    def _missing_film(self) -> list[str]:
        missing = self._missing_python("FILM", self._film_python)
        cli = self._film_dir / "eval" / "interpolator_cli.py"
        if not cli.exists():
            missing.append(f"FILM CLI missing: {cli}")
        if not self._film_model_path.exists():
            missing.append(f"FILM SavedModel missing: {self._film_model_path}")
        return missing

    def _missing_latent(self) -> list[str]:
        if not self._latent_workflow.exists():
            return [f"latent enhance workflow missing: {self._latent_workflow}"]
        return []

    @staticmethod
    def _missing_python(label: str, python_bin: str) -> list[str]:
        if not Path(python_bin).exists() and shutil.which(python_bin) is None:
            return [f"{label} python not found: {python_bin}"]
        return []

    def _uses_realesrgan(self) -> bool:
        return self.backend.startswith("realesrgan_")

    def _uses_rife(self) -> bool:
        return self.backend in {"rife", "realesrgan_rife", "latent_rife"}

    def _uses_film(self) -> bool:
        return self.backend in {"film", "realesrgan_film", "latent_film"}

    def _uses_latent(self) -> bool:
        return self.backend.startswith("latent_")

    async def _run_realesrgan(self, input_path: Path, scratch_dir: Path) -> Path:
        output_dir = scratch_dir / "realesrgan"
        output_dir.mkdir(parents=True, exist_ok=True)
        suffix = "realesrgan"
        cmd = [
            self._realesrgan_python,
            str(self._realesrgan_dir / "inference_realesrgan_video.py"),
            "-i",
            str(input_path),
            "-n",
            self._realesrgan_model,
            "-s",
            f"{self._realesrgan_outscale:g}",
            "-o",
            str(output_dir),
            "--suffix",
            suffix,
            "--tile",
            str(self._realesrgan_tile),
            "--ffmpeg_bin",
            self._ffmpeg_bin,
        ]
        await self._run_command(cmd, cwd=self._realesrgan_dir)
        output = output_dir / f"{input_path.stem}_{suffix}.mp4"
        if not output.exists():
            raise RuntimeError(f"Real-ESRGAN did not produce expected output: {output}")
        return output

    async def _run_rife(
        self,
        req: VideoEnhanceRequest,
        input_path: Path,
        scratch_dir: Path,
    ) -> Path:
        output = scratch_dir / "rife.mp4"
        cmd = [
            self._rife_python,
            str(self._rife_dir / "inference_video.py"),
            "--video",
            str(input_path),
            "--output",
            str(output),
            "--model",
            str(self._rife_model_dir),
            "--fps",
            str(req.target_fps),
            "--scale",
            f"{self._rife_scale:g}",
            "--exp",
            str(self._rife_exp),
        ]
        if self._rife_fp16:
            cmd.append("--fp16")
        await self._run_command(cmd, cwd=self._rife_dir)
        if not output.exists():
            raise RuntimeError(f"RIFE did not produce expected output: {output}")
        return output

    async def _run_film(
        self,
        req: VideoEnhanceRequest,
        input_path: Path,
        scratch_dir: Path,
    ) -> Path:
        frames_dir = scratch_dir / "film_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        await self._run_command(
            [
                self._ffmpeg_bin,
                "-y",
                "-i",
                str(input_path),
                "-vsync",
                "0",
                str(frames_dir / "%08d.png"),
            ]
        )
        meta = await self._probe_video(input_path)
        source_fps = float(meta.get("fps") or req.target_fps)
        film_fps = max(1, int(round(source_fps * (2 ** self._film_times_to_interpolate))))
        cmd = [
            self._film_python,
            "-m",
            "eval.interpolator_cli",
            "--pattern",
            str(frames_dir),
            "--model_path",
            str(self._film_model_path),
            "--times_to_interpolate",
            str(self._film_times_to_interpolate),
            "--fps",
            str(film_fps),
            "--block_height",
            str(self._film_block_height),
            "--block_width",
            str(self._film_block_width),
            "--output_video",
        ]
        await self._run_command(cmd, cwd=self._film_dir)
        output = frames_dir / "interpolated.mp4"
        if not output.exists():
            raise RuntimeError(f"FILM did not produce expected output: {output}")
        return output

    async def _run_latent_comfyui(
        self,
        req: VideoEnhanceRequest,
        input_path: Path,
        scratch_dir: Path,
    ) -> Path:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for latent ComfyUI video enhance") from exc

        workflow = json.loads(self._latent_workflow.read_text())
        output_prefix = f"{scratch_dir.name}_latent"
        self._apply_comfy_placeholders(workflow, req, input_path, output_prefix)

        async with httpx.AsyncClient(timeout=600.0) as client:
            resp = await client.post(f"{self._comfyui_base_url}/prompt", json={"prompt": workflow})
            resp.raise_for_status()
            prompt_id = resp.json()["prompt_id"]
            video_bytes = await self._wait_for_comfy_video(client, prompt_id)

        output = scratch_dir / "latent.mp4"
        output.write_bytes(video_bytes)
        return output

    @staticmethod
    def _apply_comfy_placeholders(
        workflow: dict[str, Any],
        req: VideoEnhanceRequest,
        input_path: Path,
        output_prefix: str,
    ) -> None:
        replacements = {
            "__INPUT_VIDEO__": str(input_path),
            "__TARGET_WIDTH__": req.target_width,
            "__TARGET_HEIGHT__": req.target_height,
            "__TARGET_FPS__": req.target_fps,
            "__OUTPUT_PREFIX__": output_prefix,
            "__SEED__": int(time.time()),
        }
        for node in workflow.values():
            if not isinstance(node, dict):
                continue
            title = (node.get("_meta") or {}).get("title")
            if title not in replacements:
                continue
            value = replacements[title]
            inputs = node.setdefault("inputs", {})
            if title == "__INPUT_VIDEO__":
                for key in ("video", "video_path", "path", "filename", "image"):
                    if key in inputs:
                        inputs[key] = value
            elif title == "__OUTPUT_PREFIX__":
                for key in ("filename_prefix", "prefix", "output_prefix"):
                    if key in inputs:
                        inputs[key] = value
            elif title == "__SEED__":
                for key in ("seed", "noise_seed"):
                    if key in inputs:
                        inputs[key] = value
            else:
                for key in ("width", "height", "fps", "target_width", "target_height", "target_fps"):
                    if key in inputs:
                        inputs[key] = value

    async def _wait_for_comfy_video(self, client: Any, prompt_id: str) -> bytes:
        deadline = time.monotonic() + 900.0
        while time.monotonic() < deadline:
            resp = await client.get(f"{self._comfyui_base_url}/history/{prompt_id}")
            resp.raise_for_status()
            history = resp.json()
            if prompt_id not in history:
                await asyncio.sleep(3.0)
                continue
            outputs = history[prompt_id].get("outputs", {})
            for output in outputs.values():
                for key in ("videos", "gifs"):
                    for item in output.get(key, []) or []:
                        view = await client.get(
                            f"{self._comfyui_base_url}/view",
                            params={
                                "filename": item["filename"],
                                "subfolder": item.get("subfolder", ""),
                                "type": item.get("type", "output"),
                            },
                        )
                        view.raise_for_status()
                        return view.content
            raise RuntimeError(f"ComfyUI prompt {prompt_id} finished without a video output")
        raise TimeoutError(f"ComfyUI latent enhance did not finish prompt {prompt_id}")

    async def _normalize_output(
        self,
        original_input: Path,
        enhanced_input: Path,
        output_path: Path,
        req: VideoEnhanceRequest,
    ) -> None:
        scale_pad = (
            f"fps={req.target_fps},"
            f"scale={req.target_width}:{req.target_height}:flags=lanczos"
            ":force_original_aspect_ratio=decrease,"
            f"pad={req.target_width}:{req.target_height}"
            ":(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
        )
        cmd = [
            self._ffmpeg_bin,
            "-y",
            "-i",
            str(enhanced_input),
        ]
        if req.preserve_audio:
            cmd += ["-i", str(original_input), "-map", "0:v:0", "-map", "1:a?"]
        else:
            cmd += ["-map", "0:v:0", "-an"]
        cmd += [
            "-vf",
            scale_pad,
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
            cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd.append(str(output_path))
        await self._run_command(cmd)

    async def _probe_video(self, path: Path) -> dict[str, float]:
        proc = await asyncio.create_subprocess_exec(
            self._ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffprobe failed ({proc.returncode}): {stderr.decode(errors='replace')[-1000:]}"
            )
        data = json.loads(stdout.decode() or "{}")
        stream = (data.get("streams") or [{}])[0]
        fps = _parse_rate(stream.get("avg_frame_rate"))
        return {
            "width": float(stream.get("width") or 0),
            "height": float(stream.get("height") or 0),
            "fps": fps,
            "duration": float((data.get("format") or {}).get("duration") or 0),
        }

    async def _run_command(self, cmd: list[str], cwd: Path | None = None) -> None:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(cwd) if cwd else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            tail = (stderr or stdout).decode(errors="replace")[-3000:]
            raise RuntimeError(f"video enhance command failed ({proc.returncode}): {tail}")


def _parse_rate(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" not in value:
        try:
            return float(value)
        except ValueError:
            return 0.0
    num, den = value.split("/", 1)
    try:
        denominator = float(den)
        if denominator == 0:
            return 0.0
        return float(num) / denominator
    except ValueError:
        return 0.0
