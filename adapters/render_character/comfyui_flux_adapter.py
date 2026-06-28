import json
import logging
import time
from pathlib import Path

from core.capabilities.base import RenderCharacter
from core.models.capabilities import ImageSet, RenderCharacterRequest
from core.models.common import CostEstimate, HealthStatus
from core.models.profile import CastMember
from core.observability import log_event

logger = logging.getLogger(__name__)

_LORA_EXTENSIONS = (".safetensors", ".pt", ".ckpt")
_PLACEHOLDER_LORA_PREFIX = b"TEST-ONLY placeholder"

# Path to the bundled workflow template (relative to repo root).
_WORKFLOW_TEMPLATE = Path("assets/comfyui_workflows/flux_lora_txt2img.json")

_POLL_INTERVAL = 3.0   # seconds between /history polls
_TIMEOUT = 300.0       # max wait for ComfyUI to finish a prompt


def _is_placeholder_lora(path: Path) -> bool:
    try:
        return path.read_bytes()[:64].startswith(_PLACEHOLDER_LORA_PREFIX)
    except OSError:
        return False


class ComfyUIFluxAdapter(RenderCharacter):
    """
    render_character adapter: per-member still images via ComfyUI + Flux.1-dev + LoRA.

    ComfyUI must be running and have:
      - Flux.1-dev (UNET + CLIP-L + T5XXL + AE) loaded
      - The character LoRA in its loras/ folder
      - LTX-Video nodes (optional, only needed for generate_video)

    Expects a ComfyUI workflow JSON template at:
      assets/comfyui_workflows/flux_lora_txt2img.json

    The template uses these placeholder node titles (set via _meta.title in each node):
      "__PROMPT__"       — CLIPTextEncodeFlux positive node
      "__LORA_NAME__"    — LoraLoader node  (lora_name field)
      "__LORA_WEIGHT__"  — LoraLoader node  (strength_model field)
      "__WIDTH__"        — EmptySD3LatentImage width field
      "__HEIGHT__"       — EmptySD3LatentImage height field
      "__STEPS__"        — KSampler or BasicScheduler steps field
      "__SEED__"         — RandomNoise seed field

    Args:
        work_dir:    Output directory for rendered PNG files.
        base_url:    ComfyUI API root (e.g. http://localhost:8188).
        lora_dir:    Local directory where LoRA .safetensors files live.
        lora_weight: LoRA strength (0–1).
        steps:       Flux sampling steps. 20 is good for quality/speed.
        width:       Output image width (must be divisible by 8).
        height:      Output image height (must be divisible by 8).
        num_images:  Number of candidate images per call (runs workflow N times).
        allow_placeholder_lora: Accept TEST-ONLY placeholder files (smoke tests only).
    """

    version = "1.0.0"

    def __init__(
        self,
        work_dir: Path,
        base_url: str = "http://localhost:8188",
        lora_dir: Path = Path("loras"),
        lora_weight: float = 0.9,
        steps: int = 20,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        allow_placeholder_lora: bool = False,
    ) -> None:
        self.work_dir = work_dir
        self._base_url = base_url.rstrip("/")
        self._lora_dir = lora_dir
        self._lora_weight = lora_weight
        self._steps = steps
        self._width = width
        self._height = height
        self._num_images = num_images
        self._allow_placeholder_lora = allow_placeholder_lora

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
                reason=f"ComfyUI unreachable at {self._base_url}: {exc}",
            )

    async def estimate_cost(self, req: RenderCharacterRequest) -> CostEstimate:
        return CostEstimate(
            amount=0.0,
            notes="Self-hosted ComfyUI + Flux; GPU compute cost via rented hardware.",
        )

    async def run(self, req: RenderCharacterRequest) -> ImageSet:
        lora_path = self._check_lora(req.member)
        placeholder = _is_placeholder_lora(lora_path)
        if placeholder and not self._allow_placeholder_lora:
            raise RuntimeError(
                f"LoRA for '{req.member.name}' is a TEST-ONLY placeholder: {lora_path}. "
                "Replace with trained Flux LoRA weights, or set "
                "VIDEO_ME_RENDER_ALLOW_PLACEHOLDER_LORA=true for smoke tests."
            )

        import httpx

        out_dir = self.work_dir / req.member.id
        out_dir.mkdir(parents=True, exist_ok=True)

        lora_name = self.lora_name(req.member.lora_ref)
        prompt_text = self._build_prompt(req, skip_lora=placeholder)

        log_event(
            logger,
            "render_character_started",
            member_id=req.member.id,
            adapter="comfyui_flux",
            lora=lora_name,
            placeholder=placeholder,
            prompt=prompt_text,
        )

        image_uris: list[str] = []
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            for i in range(self._num_images):
                workflow = self._build_workflow(
                    prompt_text=prompt_text,
                    lora_name=lora_name if not placeholder else "",
                    seed=int(time.time()) + i,
                )
                prompt_id = await self._submit_prompt(client, workflow)
                image_bytes = await self._wait_for_image(client, prompt_id)
                path = out_dir / f"render_{i:02d}.png"
                path.write_bytes(image_bytes)
                image_uris.append(str(path))

        log_event(
            logger,
            "render_character_completed",
            member_id=req.member.id,
            image_count=len(image_uris),
        )
        return ImageSet(images=image_uris, member_id=req.member.id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def lora_name(self, lora_ref: str) -> str:
        """Derive LoRA filename stem from lora_ref (same logic as DiffusionRenderAdapter)."""
        parts = Path(lora_ref).parts
        if parts and parts[0] == "loras":
            parts = parts[1:]
        return "_".join(parts)

    def _check_lora(self, member: CastMember) -> Path:
        name = self.lora_name(member.lora_ref)
        for ext in _LORA_EXTENSIONS:
            path = self._lora_dir / f"{name}{ext}"
            if path.exists():
                return path
        expected = self._lora_dir / f"{name}.safetensors"
        raise RuntimeError(
            f"LoRA for '{member.name}' not found. Expected: {expected}. "
            "Complete Track B (Flux LoRA training) before running render_character."
        )

    def _build_prompt(self, req: RenderCharacterRequest, *, skip_lora: bool) -> str:
        parts = [req.member.visual_descriptor, f"in {req.setting}"]
        if req.expression:
            parts.append(req.expression)
        parts += ["children's animation style, cartoon, vibrant colors",
                  "high quality, clean lines, expressive character"]
        return ", ".join(parts)

    def _build_workflow(self, prompt_text: str, lora_name: str, seed: int) -> dict:
        """
        Load the workflow template and substitute placeholders.
        Falls back to a minimal hard-coded Flux workflow if the template file is absent.
        """
        if _WORKFLOW_TEMPLATE.exists():
            workflow = json.loads(_WORKFLOW_TEMPLATE.read_text())
            for node in workflow.values():
                title = node.get("_meta", {}).get("title", "")
                inputs = node.get("inputs", {})
                if title == "__PROMPT__":
                    inputs["text"] = prompt_text
                elif title == "__LORA_NAME__":
                    inputs["lora_name"] = f"{lora_name}.safetensors" if lora_name else ""
                    inputs["strength_model"] = self._lora_weight
                    inputs["strength_clip"] = self._lora_weight
                elif title == "__WIDTH__":
                    inputs["width"] = self._width
                elif title == "__HEIGHT__":
                    inputs["height"] = self._height
                elif title == "__STEPS__":
                    inputs["steps"] = self._steps
                elif title == "__SEED__":
                    inputs["noise_seed"] = seed
            return workflow
        # Minimal fallback workflow (no LoRA, basic Flux txt2img).
        return self._minimal_flux_workflow(prompt_text, seed)

    def _minimal_flux_workflow(self, prompt_text: str, seed: int) -> dict:
        """Hard-coded minimal Flux txt2img workflow for smoke testing without a template file."""
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "flux1-dev.safetensors"},
                "_meta": {"title": "Load Checkpoint"},
            },
            "2": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt_text, "clip": ["1", 1]},
                "_meta": {"title": "__PROMPT__"},
            },
            "3": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": self._width, "height": self._height, "batch_size": 1},
            },
            "4": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": self._steps,
                    "cfg": 3.5,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["2", 0],
                    "latent_image": ["3", 0],
                },
                "_meta": {"title": "__STEPS__"},
            },
            "5": {"class_type": "VAEDecode", "inputs": {"samples": ["4", 0], "vae": ["1", 2]}},
            "6": {
                "class_type": "SaveImage",
                "inputs": {"images": ["5", 0], "filename_prefix": "video_me_render"},
            },
        }

    async def _submit_prompt(self, client, workflow: dict) -> str:
        resp = await client.post(
            f"{self._base_url}/prompt",
            json={"prompt": workflow},
        )
        resp.raise_for_status()
        return resp.json()["prompt_id"]

    async def _wait_for_image(self, client, prompt_id: str) -> bytes:
        """Poll /history until the prompt completes, then download the first output image."""
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
                images = node_output.get("images", [])
                if images:
                    img_meta = images[0]
                    dl = await client.get(
                        f"{self._base_url}/view",
                        params={
                            "filename": img_meta["filename"],
                            "subfolder": img_meta.get("subfolder", ""),
                            "type": img_meta.get("type", "output"),
                        },
                    )
                    dl.raise_for_status()
                    return dl.content
        raise TimeoutError(
            f"ComfyUI did not finish prompt {prompt_id} within {_TIMEOUT}s"
        )


async def _async_sleep(seconds: float) -> None:
    import asyncio
    await asyncio.sleep(seconds)
