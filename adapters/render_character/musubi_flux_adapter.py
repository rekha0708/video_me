import asyncio
import logging
import shutil
from pathlib import Path

from core.capabilities.base import RenderCharacter
from core.models.capabilities import ImageSet, RenderCharacterRequest
from core.models.common import CostEstimate, HealthStatus
from core.models.profile import CastMember
from core.observability import log_event

logger = logging.getLogger(__name__)

_LORA_EXTENSIONS = (".safetensors", ".pt", ".ckpt")
_PLACEHOLDER_LORA_PREFIX = b"TEST-ONLY placeholder"

_MUSUBI_SCRIPT = Path("/workspace/musubi-tuner/src/musubi_tuner/flux_2_generate_image.py")
# Dedicated venv (python3 -m venv --system-site-packages) with musubi-tuner installed —
# inherits system torch, isolated from the lightweight orchestration .venv (which must
# not carry heavy ML deps). sys.executable would resolve to the orchestration venv,
# where musubi_tuner is never installed.
_MUSUBI_PYTHON = Path("/workspace/.venv_musubi/bin/python")
_DIT = Path("/workspace/ComfyUI/models/diffusion_models/flux2-dev.safetensors")
_VAE = Path("/workspace/ComfyUI/models/diffusion_models/ae.safetensors")
_TEXT_ENCODER = Path(
    "/workspace/FLUX2-text-encoder/text_encoder/model-00001-of-00010.safetensors"
)


def _is_placeholder_lora(path: Path) -> bool:
    try:
        return path.read_bytes()[:64].startswith(_PLACEHOLDER_LORA_PREFIX)
    except OSError:
        return False


class MusubiFluxAdapter(RenderCharacter):
    """
    render_character adapter: per-member still images via musubi-tuner
    flux_2_generate_image.py + Flux 2.0 Dev + Mistral 3 text encoder + LoRA.

    Uses musubi-tuner's own inference script directly — no ComfyUI needed for
    image generation. ComfyUI is still used for LTX video generation.

    Args:
        work_dir:    Output directory for rendered PNG files.
        lora_dir:    Local directory where LoRA .safetensors files live.
        lora_weight: LoRA strength (0–1).
        steps:       Flux sampling steps.
        width:       Output image width.
        height:      Output image height.
        num_images:  Number of candidate images per call.
        allow_placeholder_lora: Accept TEST-ONLY placeholder files (smoke tests only).
        guidance_scale: Flux guidance scale.
    """

    version = "1.0.0"

    def __init__(
        self,
        work_dir: Path,
        lora_dir: Path = Path("loras"),
        lora_weight: float = 0.9,
        steps: int = 20,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        allow_placeholder_lora: bool = False,
        guidance_scale: float = 3.5,
    ) -> None:
        self.work_dir = work_dir
        self._lora_dir = lora_dir
        self._lora_weight = lora_weight
        self._steps = steps
        self._width = width
        self._height = height
        self._num_images = num_images
        self._allow_placeholder_lora = allow_placeholder_lora
        self._guidance_scale = guidance_scale

    async def health(self) -> HealthStatus:
        missing = []
        if not _MUSUBI_PYTHON.exists():
            missing.append(str(_MUSUBI_PYTHON))
        if not _MUSUBI_SCRIPT.exists():
            missing.append(str(_MUSUBI_SCRIPT))
        if not _DIT.exists():
            missing.append(str(_DIT))
        if not _VAE.exists():
            missing.append(str(_VAE))
        if not _TEXT_ENCODER.exists():
            missing.append(str(_TEXT_ENCODER))
        if missing:
            return HealthStatus(status="down", reason=f"Missing: {', '.join(missing)}")
        return HealthStatus(status="ok")

    async def estimate_cost(self, req: RenderCharacterRequest) -> CostEstimate:
        return CostEstimate(
            amount=0.0,
            notes="Self-hosted musubi-tuner Flux 2.0 inference; GPU cost via rented hardware.",
        )

    async def run(self, req: RenderCharacterRequest) -> ImageSet:
        lora_path = self._check_lora(req.member)
        placeholder = _is_placeholder_lora(lora_path)
        if placeholder and not self._allow_placeholder_lora:
            raise RuntimeError(
                f"LoRA for '{req.member.name}' is a TEST-ONLY placeholder: {lora_path}. "
                "Replace with trained Flux 2.0 LoRA weights, or set "
                "VIDEO_ME_RENDER_ALLOW_PLACEHOLDER_LORA=true for smoke tests."
            )

        # Shot-scoped when shot_id is set, so per-shot backgrounds don't collide;
        # falls back to member-keyed for callers that don't pass shot_id.
        out_dir = (
            self.work_dir / req.shot_id / req.member.id
            if req.shot_id
            else self.work_dir / req.member.id
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        prompt_text = self._build_prompt(req, skip_lora=placeholder)

        log_event(
            logger,
            "render_character_started",
            member_id=req.member.id,
            adapter="musubi_flux",
            lora=str(lora_path),
            placeholder=placeholder,
            prompt=prompt_text,
        )

        image_uris: list[str] = []
        for i in range(self._num_images):
            out_path = out_dir / f"render_{i:02d}.png"
            seed = i  # deterministic per-candidate seed
            await self._generate(
                prompt=prompt_text,
                lora_path=lora_path if not placeholder else None,
                out_path=out_path,
                seed=seed,
            )
            image_uris.append(str(out_path))

        log_event(
            logger,
            "render_character_completed",
            member_id=req.member.id,
            image_count=len(image_uris),
        )
        return ImageSet(images=image_uris, member_id=req.member.id)

    async def _generate(
        self,
        prompt: str,
        lora_path: Path | None,
        out_path: Path,
        seed: int,
    ) -> None:
        # flux_2_generate_image.py treats --save_path as an output DIRECTORY and
        # invents its own filename inside it (save_images_grid: "{time_flag}_{seed}_000.png")
        # — it does not accept a literal target file path. Point it at a scratch
        # dir per candidate, then move the single image it writes to out_path.
        scratch_dir = out_path.parent / f"_scratch_{out_path.stem}"
        shutil.rmtree(scratch_dir, ignore_errors=True)
        scratch_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(_MUSUBI_PYTHON),
            str(_MUSUBI_SCRIPT),
            "--dit", str(_DIT),
            "--vae", str(_VAE),
            "--text_encoder", str(_TEXT_ENCODER),
            "--prompt", prompt,
            "--image_size", str(self._width), str(self._height),
            "--infer_steps", str(self._steps),
            "--guidance_scale", str(self._guidance_scale),
            "--seed", str(seed),
            "--save_path", str(scratch_dir),
            "--fp8", "--fp8_scaled",
            "--attn_mode", "flash",
            "--model_version", "dev",
        ]
        if lora_path is not None:
            cmd += ["--lora_weight", str(lora_path), "--lora_multiplier", str(self._lora_weight)]

        logger.info("Running musubi-tuner inference: %s", " ".join(cmd[-6:]))
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"flux_2_generate_image.py failed (exit {proc.returncode}):\n"
                f"{stdout.decode(errors='replace')[-2000:]}"
            )
        generated = sorted(scratch_dir.glob("*.png"))
        if not generated:
            raise RuntimeError(
                f"flux_2_generate_image.py exited 0 but no image found in: {scratch_dir}"
            )
        shutil.move(str(generated[0]), str(out_path))
        shutil.rmtree(scratch_dir, ignore_errors=True)

    def lora_name(self, lora_ref: str) -> str:
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
            "Complete Track B (Flux 2.0 LoRA training) before running render_character."
        )

    def _build_prompt(self, req: RenderCharacterRequest, *, skip_lora: bool) -> str:
        from adapters.render_character.prompt_util import camera_phrase

        parts = [req.member.visual_descriptor, f"in {req.setting}"]
        framing = camera_phrase(req.camera)
        if framing:
            parts.append(framing)
        if req.expression:
            parts.append(req.expression)
        parts += [
            "children's animation style, cartoon, vibrant colors",
            "high quality, clean lines, expressive character",
        ]
        return ", ".join(parts)
