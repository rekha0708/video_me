from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
import signal
from urllib.parse import unquote, urlparse

from core.capabilities.base import GenerateVideo
from core.models.capabilities import PreparedWanAnimateInput, VideoClip, VideoRequest
from core.models.common import CostEstimate, HealthStatus
from core.observability import log_event

logger = logging.getLogger(__name__)

_PREPROCESS_CONTRACT_VERSION = 2


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _revision_hint(path: Path) -> str:
    """Return a cheap, deterministic revision hint for a repo/model directory."""
    root = path.expanduser().resolve()
    head = root / ".git" / "HEAD"
    try:
        value = head.read_text(encoding="utf-8").strip()
        if value.startswith("ref: "):
            ref = root / ".git" / value.removeprefix("ref: ")
            if ref.is_file():
                value = ref.read_text(encoding="utf-8").strip()
        if value:
            return value
    except OSError:
        pass
    for candidate in (root / "model_index.json", root / "config.json", root):
        try:
            stat = candidate.stat()
            return f"{root}:{stat.st_size}:{stat.st_mtime_ns}"
        except OSError:
            continue
    return str(root)


async def _terminate_process_group(process: asyncio.subprocess.Process) -> None:
    """Terminate a cancelled preprocessor/FFmpeg and every child it spawned."""
    if process.returncode is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (AttributeError, ProcessLookupError, PermissionError):
        process.kill()
    try:
        await asyncio.wait_for(process.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError, PermissionError):
            process.kill()
        await process.wait()


def _local_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    if parsed.scheme:
        raise ValueError(f"Wan Animate driver must be a local file, got: {uri}")
    return Path(uri)


class WanAnimateAdapter(GenerateVideo):
    """Official Wan2.2 Animate motion-transfer/character-replacement backend."""

    version = "1.1.0"
    managed_vram = True
    native_lipsync = False

    def __init__(
        self,
        *,
        work_dir: Path,
        base_url: str,
        python_bin: str,
        wan_dir: Path,
        model_dir: Path,
        mode: str = "animate",
        driver_source: str = "job_source",
        driver_uri: str = "",
        timeline: str = "source_timestamps",
        fps: int = 30,
        resolution_area: str = "720p",
        subject_selection: str = "largest",
        retarget_pose: bool = False,
        use_flux_retarget: bool = False,
        refert_num: int = 1,
        sampling_steps: int = 20,
        mask_iterations: int = 3,
        mask_kernel: int = 7,
        mask_w_len: int = 1,
        mask_h_len: int = 1,
        ffmpeg_bin: str = "ffmpeg",
        ffprobe_bin: str = "ffprobe",
    ) -> None:
        self.work_dir = work_dir
        self.preprocess_dir = work_dir.parent.parent / "wan_animate_preprocess"
        self._base_url = base_url.rstrip("/")
        self._python_bin = python_bin
        self._wan_dir = Path(wan_dir)
        self._model_dir = Path(model_dir)
        self.mode = mode
        self.driver_source = driver_source
        self.driver_uri = driver_uri
        self.timeline = timeline
        self.fps = fps
        self.resolution_area = resolution_area
        self.subject_selection = subject_selection
        self.retarget_pose = retarget_pose
        self.use_flux_retarget = use_flux_retarget
        self.refert_num = refert_num
        self.sampling_steps = sampling_steps
        self.mask_iterations = mask_iterations
        self.mask_kernel = mask_kernel
        self.mask_w_len = mask_w_len
        self.mask_h_len = mask_h_len
        self._ffmpeg_bin = ffmpeg_bin
        self._ffprobe_bin = ffprobe_bin
        self._prepared: dict[str, PreparedWanAnimateInput] = {}
        self._wan_revision = _revision_hint(self._wan_dir)
        self._model_revision = _revision_hint(self._model_dir)

        if mode not in {"animate", "replace"}:
            raise ValueError(f"Unknown Wan Animate mode: {mode}")
        if use_flux_retarget and not retarget_pose:
            raise ValueError("Flux retargeting requires pose retargeting")
        if mode == "replace" and (retarget_pose or use_flux_retarget):
            raise ValueError("Pose retargeting is supported only in animate mode")

    async def health(self) -> HealthStatus:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self._base_url}/health")
                response.raise_for_status()
            return HealthStatus(status="ok")
        except Exception as exc:
            return HealthStatus(status="down", reason=f"Wan Animate unavailable: {exc}")

    async def load(self) -> None:
        import httpx
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{self._base_url}/load", data={"mode": self.mode})
            response.raise_for_status()

    async def unload(self) -> bool:
        import httpx
        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(f"{self._base_url}/unload")
        except httpx.ConnectError:
            return False
        if response.status_code >= 400:
            raise RuntimeError(
                f"Wan Animate refused unload ({response.status_code}): {response.text[:500]}"
            )
        return True

    async def wait_until_loaded(self, timeout_sec: float, poll_sec: float = 10.0) -> None:
        import httpx
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_sec
        async with httpx.AsyncClient(timeout=10.0) as client:
            while loop.time() < deadline:
                response = await client.get(f"{self._base_url}/health")
                response.raise_for_status()
                body = response.json()
                if body.get("model_loaded") and body.get("mode") == self.mode:
                    return
                if body.get("error"):
                    raise RuntimeError(f"Wan Animate load failed: {body['error']}")
                await asyncio.sleep(poll_sec)
        raise TimeoutError(f"Wan Animate did not load within {timeout_sec:.0f}s")

    async def estimate_cost(self, req: VideoRequest) -> CostEstimate:
        return CostEstimate(amount=0.0, notes="Self-hosted Wan2.2-Animate-14B")

    async def prepare_inputs(
        self, requests: list[VideoRequest]
    ) -> dict[str, PreparedWanAnimateInput]:
        """Normalize all driver slices, then run one batch preprocessing process."""
        if not requests:
            return {}
        self.preprocess_dir.mkdir(parents=True, exist_ok=True)
        pending: list[dict] = []
        driver_durations: dict[Path, float] = {}
        driver_hashes: dict[Path, str] = {}
        reference_hashes: dict[Path, str] = {}

        for req in requests:
            if req.driver is None:
                raise ValueError(f"Wan Animate shot {req.shot_id} has no driving video")
            driver = _local_path(req.driver.uri).expanduser().resolve()
            reference = Path(req.image_uri).expanduser().resolve()
            if not driver.is_file():
                raise FileNotFoundError(f"Driving video not found for {req.shot_id}: {driver}")
            if not reference.is_file():
                raise FileNotFoundError(f"Reference image not found for {req.shot_id}: {reference}")
            if req.driver.end_sec <= req.driver.start_sec:
                raise ValueError(f"Invalid driver range for {req.shot_id}")
            if driver not in driver_durations:
                driver_durations[driver] = await self._probe_duration(driver)
            if driver not in driver_hashes:
                driver_hashes[driver] = await asyncio.to_thread(_sha256_path, driver)
            if reference not in reference_hashes:
                reference_hashes[reference] = await asyncio.to_thread(_sha256_path, reference)
            available = driver_durations[driver]
            if req.driver.end_sec > available + 0.05:
                raise ValueError(
                    f"Driving video is too short for {req.shot_id}: needs through "
                    f"{req.driver.end_sec:.2f}s, available {available:.2f}s"
                )

            shot_dir = self.preprocess_dir / req.shot_id
            shot_dir.mkdir(parents=True, exist_ok=True)
            normalized = shot_dir / "driver_normalized.mp4"
            cache_key = self._cache_key(
                req,
                driver,
                reference,
                driver_sha256=driver_hashes[driver],
                reference_sha256=reference_hashes[reference],
            )
            manifest_path = shot_dir / "manifest.json"
            cached = self._read_cached(manifest_path, cache_key)
            if cached is not None:
                self._prepared[req.shot_id] = cached
                continue

            await self._normalize_driver(
                driver,
                normalized,
                req.driver.start_sec,
                req.driver.end_sec,
            )
            pending.append({
                "shot_id": req.shot_id,
                "video_path": str(normalized),
                "reference_path": str(reference),
                "output_path": str(shot_dir),
                "cache_key": cache_key,
                "driver_uri": req.driver.uri,
                "start_sec": req.driver.start_sec,
                "end_sec": req.driver.end_sec,
            })

        if pending:
            batch_path = self.preprocess_dir / "batch.json"
            batch_path.write_text(json.dumps({"items": pending}, indent=2), encoding="utf-8")
            cmd = [
                self._python_bin,
                "-m", "services.wan_animate_preprocess",
                "--batch", str(batch_path),
                "--wan-dir", str(self._wan_dir),
                "--model-dir", str(self._model_dir),
                "--mode", self.mode,
                "--fps", str(self.fps),
                "--resolution", self.resolution_area,
                "--subject-selection", self.subject_selection,
                "--iterations", str(self.mask_iterations),
                "--kernel", str(self.mask_kernel),
                "--w-len", str(self.mask_w_len),
                "--h-len", str(self.mask_h_len),
            ]
            if self.retarget_pose:
                cmd.append("--retarget-pose")
            if self.use_flux_retarget:
                cmd.append("--use-flux")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                stdout, _ = await process.communicate()
            except asyncio.CancelledError:
                await _terminate_process_group(process)
                raise
            if process.returncode != 0:
                raise RuntimeError(
                    "Wan Animate preprocessing failed: "
                    + stdout.decode(errors="replace")[-4000:]
                )

        for req in requests:
            manifest = json.loads(
                (self.preprocess_dir / req.shot_id / "manifest.json").read_text(encoding="utf-8")
            )
            prepared = PreparedWanAnimateInput.model_validate(manifest["prepared"])
            prepared.cache_hit = req.shot_id not in {item["shot_id"] for item in pending}
            self._prepared[req.shot_id] = prepared
        return dict(self._prepared)

    async def run(self, req: VideoRequest) -> VideoClip:
        prepared = self._prepared.get(req.shot_id)
        if prepared is None and req.driver and req.driver.prepared_dir:
            prepared = PreparedWanAnimateInput(
                shot_id=req.shot_id,
                prepared_dir=req.driver.prepared_dir,
                driver_uri=req.driver.uri,
                start_sec=req.driver.start_sec,
                end_sec=req.driver.end_sec,
                frame_count=0,
                fps=self.fps,
                width=0,
                height=0,
            )
        if prepared is None:
            raise RuntimeError(f"Wan Animate input for {req.shot_id} was not preprocessed")

        import httpx
        log_event(logger, "generate_video_started", adapter="wan_animate", shot_id=req.shot_id)
        out_dir = self.work_dir / req.shot_id
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / "clip.mp4"
        output.unlink(missing_ok=True)
        try:
            async with httpx.AsyncClient(timeout=7200.0) as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/generate",
                    data={
                        "prepared_dir": prepared.prepared_dir,
                        "mode": self.mode,
                        "fps": str(self.fps),
                        "refert_num": str(self.refert_num),
                        "sampling_steps": str(self.sampling_steps),
                        "seed": "-1",
                    },
                ) as response:
                    if response.status_code >= 400:
                        detail = (await response.aread()).decode(errors="replace")[:2000]
                        logger.error(
                            "Wan Animate error %s: %s", response.status_code, detail
                        )
                    response.raise_for_status()
                    with output.open("wb") as handle:
                        async for chunk in response.aiter_bytes(8 * 1024 * 1024):
                            handle.write(chunk)
        except BaseException:
            output.unlink(missing_ok=True)
            raise
        return VideoClip(uri=str(output), duration_sec=req.duration_sec, shot_id=req.shot_id)

    def _cache_key(
        self,
        req: VideoRequest,
        driver: Path,
        reference: Path,
        *,
        driver_sha256: str | None = None,
        reference_sha256: str | None = None,
    ) -> str:
        assert req.driver is not None
        payload = {
            "preprocess_contract_version": _PREPROCESS_CONTRACT_VERSION,
            "adapter_version": self.version,
            "wan_revision": self._wan_revision,
            "model_revision": self._model_revision,
            "driver_sha256": driver_sha256 or _sha256_path(driver),
            "reference_sha256": reference_sha256 or _sha256_path(reference),
            "start": req.driver.start_sec,
            "end": req.driver.end_sec,
            "mode": self.mode,
            "fps": self.fps,
            "resolution": self.resolution_area,
            "subject": self.subject_selection,
            "retarget": self.retarget_pose,
            "flux": self.use_flux_retarget,
            "mask": [self.mask_iterations, self.mask_kernel, self.mask_w_len, self.mask_h_len],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _read_cached(self, path: Path, cache_key: str) -> PreparedWanAnimateInput | None:
        if not path.exists():
            return None
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            prepared = PreparedWanAnimateInput.model_validate(manifest["prepared"])
            required = ["src_ref.png", "src_pose.mp4", "src_face.mp4"]
            if self.mode == "replace":
                required += ["src_bg.mp4", "src_mask.mp4"]
            if manifest.get("cache_key") != cache_key:
                return None
            if not all((Path(prepared.prepared_dir) / name).exists() for name in required):
                return None
            prepared.cache_hit = True
            return prepared
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            return None

    async def _normalize_driver(
        self, source: Path, output: Path, start_sec: float, end_sec: float
    ) -> None:
        duration = end_sec - start_sec
        cmd = [
            self._ffmpeg_bin, "-y", "-ss", f"{start_sec:.6f}", "-i", str(source),
            "-t", f"{duration:.6f}", "-map", "0:v:0", "-an",
            "-vf", f"fps={self.fps},setsar=1,format=yuv420p",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-metadata:s:v:0", "rotate=0", "-movflags", "+faststart", str(output),
        ]
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            _, stderr = await process.communicate()
        except asyncio.CancelledError:
            await _terminate_process_group(process)
            raise
        if process.returncode != 0:
            raise RuntimeError(
                f"Failed to normalize driver video: {stderr.decode(errors='replace')[-2000:]}"
            )

    async def _probe_duration(self, source: Path) -> float:
        process = await asyncio.create_subprocess_exec(
            self._ffprobe_bin,
            "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(source),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode != 0:
            raise ValueError(
                f"Driving video is unreadable: {stderr.decode(errors='replace')[-1000:]}"
            )
        try:
            duration = float(stdout.decode().strip())
        except ValueError as exc:
            raise ValueError("Driving video has no finite duration") from exc
        if duration <= 0:
            raise ValueError("Driving video has no positive duration")
        return duration
