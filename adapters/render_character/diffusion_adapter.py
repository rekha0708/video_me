import base64
import logging
from pathlib import Path

from core.capabilities.base import RenderCharacter
from core.models.capabilities import ImageSet, RenderCharacterRequest
from core.models.common import CostEstimate, HealthStatus
from core.models.profile import CastMember
from core.observability import log_event

logger = logging.getLogger(__name__)

_NEGATIVE_PROMPT = (
    "blurry, low quality, deformed, ugly, bad anatomy, extra limbs, "
    "watermark, signature, text, nsfw, scary, disturbing, violent"
)

_LORA_EXTENSIONS = (".safetensors", ".pt", ".ckpt")
_PLACEHOLDER_LORA_PREFIX = b"TEST-ONLY placeholder"


def is_placeholder_lora(path: Path) -> bool:
    """Return True for explicit test-only LoRA placeholder files."""
    try:
        return path.read_bytes()[:64].startswith(_PLACEHOLDER_LORA_PREFIX)
    except OSError:
        return False


class DiffusionRenderAdapter(RenderCharacter):
    """
    render_character adapter: per-member still images via Stable Diffusion +
    per-member LoRA, served by an AUTOMATIC1111-compatible HTTP API.

    **Track B dependency**: each cast member needs a trained LoRA file at
    ``lora_dir/<derived_name>.safetensors`` before this adapter can run.
    The adapter raises a clear error pointing to Track B when the file is absent.

    One instance per job — ``work_dir`` should be job-scoped so concurrent
    jobs don't overwrite each other's renders.

    Args:
        work_dir:   Output directory for rendered PNG files.
        base_url:   AUTOMATIC1111-compatible SD API root (e.g. http://localhost:7860).
        lora_dir:   Local directory containing ``<lora_name>.safetensors`` files.
        lora_weight: Strength of the character LoRA in the prompt (0–1).
        steps:      Diffusion steps. 28 is a good quality/speed trade-off.
        cfg_scale:  Classifier-free guidance scale. 7 is the SD default.
        width:      Output image width in pixels.
        height:     Output image height in pixels.
        num_images: Number of candidate renders to produce per call.
        allow_placeholder_lora:
            If True, explicit TEST-ONLY placeholder LoRA files are accepted
            for smoke tests and omitted from the prompt. Real runs should keep
            this False so fake weights fail before the SD API call.
    """

    version = "1.0.0"

    def __init__(
        self,
        work_dir: Path,
        base_url: str = "http://localhost:7860",
        lora_dir: Path = Path("loras"),
        lora_weight: float = 0.9,
        steps: int = 28,
        cfg_scale: float = 7.0,
        width: int = 768,
        height: int = 768,
        num_images: int = 1,
        negative_prompt: str = _NEGATIVE_PROMPT,
        allow_placeholder_lora: bool = False,
    ) -> None:
        self.work_dir = work_dir
        self._base_url = base_url.rstrip("/")
        self._lora_dir = lora_dir
        self._lora_weight = lora_weight
        self._steps = steps
        self._cfg_scale = cfg_scale
        self._width = width
        self._height = height
        self._num_images = num_images
        self._negative_prompt = negative_prompt
        self._allow_placeholder_lora = allow_placeholder_lora

    async def health(self) -> HealthStatus:
        try:
            import httpx
        except ImportError:
            return HealthStatus(
                status="down",
                reason="httpx not installed. Run: pip install httpx",
            )
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/sdapi/v1/sd-models")
                resp.raise_for_status()
            return HealthStatus(status="ok")
        except Exception as exc:
            return HealthStatus(
                status="down",
                reason=f"Diffusion service unreachable at {self._base_url}: {exc}",
            )

    async def estimate_cost(self, req: RenderCharacterRequest) -> CostEstimate:
        return CostEstimate(
            amount=0.0,
            notes="Self-hosted diffusion service; GPU compute cost via rented hardware.",
        )

    async def run(self, req: RenderCharacterRequest) -> ImageSet:
        lora_path = self._check_lora(req.member)
        placeholder_lora = is_placeholder_lora(lora_path)
        if placeholder_lora and not self._allow_placeholder_lora:
            raise RuntimeError(
                f"LoRA for '{req.member.name}' is a TEST-ONLY placeholder: {lora_path}. "
                "Replace it with trained weights for a real render, or set "
                "VIDEO_ME_RENDER_ALLOW_PLACEHOLDER_LORA=true for temporary smoke tests."
            )

        import httpx
        out_dir = self.work_dir / req.member.id
        out_dir.mkdir(parents=True, exist_ok=True)

        prompt = self._build_prompt(req, lora_path=lora_path)

        log_event(
            logger,
            "render_character_started",
            member_id=req.member.id,
            member_name=req.member.name,
            lora=str(lora_path),
            placeholder_lora=placeholder_lora,
            setting=req.setting,
            prompt=prompt,
            negative_prompt=self._negative_prompt,
        )

        payload = {
            "prompt": prompt,
            "negative_prompt": self._negative_prompt,
            "steps": self._steps,
            "cfg_scale": self._cfg_scale,
            "width": self._width,
            "height": self._height,
            "n_iter": 1,
            "batch_size": self._num_images,
            "sampler_name": "DPM++ 2M Karras",
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/sdapi/v1/txt2img",
                json=payload,
            )
            resp.raise_for_status()

        image_uris = self._save_images(resp.json()["images"], out_dir)

        log_event(
            logger,
            "render_character_completed",
            member_id=req.member.id,
            image_count=len(image_uris),
        )

        return ImageSet(images=image_uris, member_id=req.member.id)

    # ------------------------------------------------------------------
    # Private helpers (mockable in tests)
    # ------------------------------------------------------------------

    def lora_name(self, lora_ref: str) -> str:
        """
        Derive LoRA filename stem from lora_ref.
        "loras/pig_kids_placeholder/c1" → "pig_kids_placeholder_c1"
        """
        parts = Path(lora_ref).parts
        if parts and parts[0] == "loras":
            parts = parts[1:]
        return "_".join(parts)

    def _check_lora(self, member: CastMember) -> Path:
        """
        Return the LoRA file path for this member.
        Raises RuntimeError with a Track B prompt if the file is missing.
        """
        name = self.lora_name(member.lora_ref)
        for ext in _LORA_EXTENSIONS:
            path = self._lora_dir / f"{name}{ext}"
            if path.exists():
                return path
        expected = self._lora_dir / f"{name}.safetensors"
        raise RuntimeError(
            f"LoRA for '{member.name}' (id={member.id}) not found. "
            f"Expected: {expected}. "
            "Complete Track B (character design + LoRA training) before running render_character."
        )

    def _build_prompt(
        self,
        req: RenderCharacterRequest,
        lora_path: Path | None = None,
    ) -> str:
        name = self.lora_name(req.member.lora_ref)
        parts: list[str] = []
        if not (
            lora_path is not None
            and self._allow_placeholder_lora
            and is_placeholder_lora(lora_path)
        ):
            parts.append(f"<lora:{name}:{self._lora_weight}>")
        parts += [req.member.visual_descriptor, f"in {req.setting}"]
        if req.expression:
            parts.append(req.expression)
        parts += [
            "children's animation style, cartoon, vibrant colors",
            "high quality, clean lines, expressive character",
        ]
        return ", ".join(parts)

    def _save_images(self, b64_images: list[str], out_dir: Path) -> list[str]:
        """Decode base64 PNG responses and write to out_dir; return file URIs."""
        uris: list[str] = []
        for i, b64 in enumerate(b64_images):
            img_bytes = base64.b64decode(b64)
            path = out_dir / f"render_{i:02d}.png"
            path.write_bytes(img_bytes)
            uris.append(str(path))
        return uris
