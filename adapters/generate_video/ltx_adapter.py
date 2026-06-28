import json
import logging
import time
from pathlib import Path

from core.capabilities.base import GenerateVideo
from core.models.capabilities import VideoClip, VideoRequest
from core.models.common import CostEstimate, HealthStatus
from core.observability import log_event

logger = logging.getLogger(__name__)

_PROMPT_PREFIX = "children's cartoon animation, vibrant colors"
_PROMPT_SUFFIX = "smooth motion, expressive, high quality"

# Default FPS — LTX-Video 2.3 natively supports up to 50 FPS; 24 is standard.
_DEFAULT_FPS: int = 24

_WORKFLOW_TEMPLATE = Path("assets/comfyui_workflows/ltx_i2v.json")

_POLL_INTERVAL = 3.0
_TIMEOUT = 600.0    # LTX 8-step distilled is fast; 10 min is generous


class LtxAdapter(GenerateVideo):
    """
    generate_video adapter: per-shot image-to-video via LTX-Video 2.3 running inside ComfyUI.

    LTX-Video 2.3 has native audio/lip-sync — pass audio_uri in VideoRequest and the
    adapter uploads the WAV to ComfyUI so lip movements are generated in the same
    diffusion pass. The separate lip_sync stage (MuseTalk) is skipped automatically
    because native_lipsync = True.

    Workflow template: assets/comfyui_workflows/ltx_i2v.json
    Placeholder node titles:
      "__IMAGE__"     — LoadImage        (image field)
      "__AUDIO__"     — LoadAudio        (audio field, optional)
      "__PROMPT__"    — CLIPTextEncode   (text field)
      "__FRAMES__"    — LTXVConditioning (length field)
      "__STEPS__"     — LTXVScheduler    (steps field)
      "__SEED__"      — RandomNoise      (noise_seed field)

    Stick to native LTX resolutions (1280×720 or 480×832) — non-standard sizes
    cause audio-latent misalignment in ComfyUI nodes.

    Args:
        work_dir:  Output directory. Each shot → work_dir/<shot_id>/clip.mp4
        base_url:  ComfyUI API root (e.g. http://localhost:8188).
        fps:       Output frame rate. 24 FPS minimum for accurate lip-sync.
        steps:     Diffusion steps. LTX distilled runs well at 8 steps.
    """

    version = "1.0.0"
    native_lipsync: bool = True

    def __init__(
        self,
        work_dir: Path,
        base_url: str = "http://localhost:8188",
        fps: int = _DEFAULT_FPS,
        steps: int = 8,
    ) -> None:
        self.work_dir = work_dir
        self._base_url = base_url.rstrip("/")
        self._fps = fps
        self._steps = steps

    # ------------------------------------------------------------------
    # Capability interface
    # ------------------------------------------------------------------

    async def health(self) -> HealthStatus:
        try:
            import httpx
        except ImportError:
            return HealthStatus(status="down", reason="httpx not installed")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/system_stats")
                resp.raise_for_status()
            return HealthStatus(status="ok")
        except Exception as exc:
            return HealthStatus(
                status="down",
                reason=f"ComfyUI (LTX) unreachable at {self._base_url}: {exc}",
            )

    async def estimate_cost(self, req: VideoRequest) -> CostEstimate:
        return CostEstimate(
            amount=0.0,
            notes="Self-hosted LTX-Video 2.3 via ComfyUI; GPU compute cost via rented hardware.",
        )

    async def run(self, req: VideoRequest) -> VideoClip:
        image_path = Path(req.image_uri)
        if not image_path.exists():
            raise FileNotFoundError(
                f"Render image not found for shot {req.shot_id}: {image_path}. "
                "render_character must run before generate_video."
            )

        out_dir = self.work_dir / req.shot_id
        out_dir.mkdir(parents=True, exist_ok=True)

        prompt = self._build_prompt(req.action)
        num_frames = max(9, int(req.duration_sec * self._fps))
        # LTX frame count must be (divisible by 8) + 1
        num_frames = ((num_frames - 1) // 8) * 8 + 1

        log_event(
            logger,
            "generate_video_started",
            shot_id=req.shot_id,
            adapter="ltx",
            image=str(image_path),
            duration_sec=req.duration_sec,
            num_frames=num_frames,
            prompt=prompt,
        )

        import httpx

        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            comfy_image_name = await self._upload_image(client, image_path)
            comfy_audio_name = None
            if req.audio_uri:
                audio_path = Path(req.audio_uri)
                if audio_path.exists():
                    comfy_audio_name = await self._upload_audio(client, audio_path)

            workflow = self._build_workflow(
                image_name=comfy_image_name,
                prompt_text=prompt,
                num_frames=num_frames,
                seed=int(time.time()),
                audio_name=comfy_audio_name,
            )
            prompt_id = await self._submit_prompt(client, workflow)
            mp4_bytes = await self._wait_for_video(client, prompt_id)

        clip_path = out_dir / "clip.mp4"
        clip_path.write_bytes(mp4_bytes)

        log_event(
            logger,
            "generate_video_completed",
            shot_id=req.shot_id,
            clip=str(clip_path),
        )
        return VideoClip(uri=str(clip_path), duration_sec=req.duration_sec, shot_id=req.shot_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, action: str) -> str:
        return f"{_PROMPT_PREFIX}, {action}, {_PROMPT_SUFFIX}"

    def _build_workflow(
        self,
        image_name: str,
        prompt_text: str,
        num_frames: int,
        seed: int,
        audio_name: str | None = None,
    ) -> dict:
        if _WORKFLOW_TEMPLATE.exists():
            workflow = json.loads(_WORKFLOW_TEMPLATE.read_text())
            for node in workflow.values():
                title = node.get("_meta", {}).get("title", "")
                inputs = node.get("inputs", {})
                if title == "__IMAGE__":
                    inputs["image"] = image_name
                elif title == "__AUDIO__" and audio_name:
                    inputs["audio"] = audio_name
                elif title == "__PROMPT__":
                    inputs["text"] = prompt_text
                elif title == "__FRAMES__":
                    inputs["length"] = num_frames
                elif title == "__STEPS__":
                    inputs["steps"] = self._steps
                elif title == "__SEED__":
                    inputs["noise_seed"] = seed
            return workflow
        return self._minimal_ltx_workflow(image_name, prompt_text, num_frames, seed, audio_name)

    def _minimal_ltx_workflow(
        self,
        image_name: str,
        prompt_text: str,
        num_frames: int,
        seed: int,
        audio_name: str | None = None,
    ) -> dict:
        """Minimal LTX-Video i2v workflow for smoke testing without a template file."""
        return {
            "1": {
                "class_type": "LoadImage",
                "inputs": {"image": image_name},
                "_meta": {"title": "__IMAGE__"},
            },
            "2": {
                "class_type": "LTXVModelLoader",
                "inputs": {"model": "ltx-video-2b-v0.9.5.safetensors"},
            },
            "3": {
                "class_type": "LTXVConditioning",
                "inputs": {
                    "positive": prompt_text,
                    "negative": "blurry, low quality, deformed",
                    "frame_rate": self._fps,
                    "length": num_frames,
                    "batch_size": 1,
                },
                "_meta": {"title": "__FRAMES__"},
            },
            "4": {
                "class_type": "LTXVScheduler",
                "inputs": {"steps": self._steps, "max_shift": 2.05, "base_shift": 0.95},
                "_meta": {"title": "__STEPS__"},
            },
            "5": {
                "class_type": "RandomNoise",
                "inputs": {"noise_seed": seed},
                "_meta": {"title": "__SEED__"},
            },
            "6": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["5", 0],
                    "guider": ["3", 0],
                    "sampler": ["4", 0],
                    "sigmas": ["4", 1],
                    "latent_image": ["3", 1],
                },
            },
            "7": {
                "class_type": "LTXVDecoder",
                "inputs": {"samples": ["6", 0], "vae": ["2", 1], "enable_vae_tiling": False},
            },
            "8": {
                "class_type": "SaveAnimatedWEBP",
                "inputs": {
                    "images": ["7", 0],
                    "filename_prefix": "video_me_ltx",
                    "fps": self._fps,
                    "lossless": False,
                    "quality": 85,
                    "method": "default",
                },
            },
        }

    async def _upload_image(self, client, image_path: Path) -> str:
        with image_path.open("rb") as f:
            resp = await client.post(
                f"{self._base_url}/upload/image",
                files={"image": (image_path.name, f, "image/png")},
            )
        resp.raise_for_status()
        return resp.json()["name"]

    async def _upload_audio(self, client, audio_path: Path) -> str:
        with audio_path.open("rb") as f:
            resp = await client.post(
                f"{self._base_url}/upload/image",  # ComfyUI uses same endpoint for audio
                files={"image": (audio_path.name, f, "audio/wav")},
                data={"type": "input", "subfolder": "audio"},
            )
        resp.raise_for_status()
        return resp.json()["name"]

    async def _submit_prompt(self, client, workflow: dict) -> str:
        resp = await client.post(
            f"{self._base_url}/prompt",
            json={"prompt": workflow},
        )
        resp.raise_for_status()
        return resp.json()["prompt_id"]

    async def _wait_for_video(self, client, prompt_id: str) -> bytes:
        """Poll /history until done, then download the output video."""
        deadline = time.monotonic() + _TIMEOUT
        while time.monotonic() < deadline:
            await _async_sleep(_POLL_INTERVAL)
            resp = await client.get(f"{self._base_url}/history/{prompt_id}")
            resp.raise_for_status()
            history = resp.json()
            if prompt_id not in history:
                continue
            outputs = history[prompt_id].get("outputs", {})
            for node_output in outputs.values():
                # ComfyUI video outputs use "gifs" key (even for MP4/WEBP)
                for key in ("gifs", "videos", "images"):
                    items = node_output.get(key, [])
                    if items:
                        meta = items[0]
                        dl = await client.get(
                            f"{self._base_url}/view",
                            params={
                                "filename": meta["filename"],
                                "subfolder": meta.get("subfolder", ""),
                                "type": meta.get("type", "output"),
                            },
                        )
                        dl.raise_for_status()
                        return dl.content
        raise TimeoutError(
            f"ComfyUI (LTX) did not finish prompt {prompt_id} within {_TIMEOUT}s"
        )


async def _async_sleep(seconds: float) -> None:
    import asyncio
    await asyncio.sleep(seconds)
