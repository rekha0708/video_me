import asyncio
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass, field
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


# Output files are named "{time_flag}_{seed}__{index:03d}.png" (double underscore
# before the index) by musubi-tuner's save_images_grid; the seed token is how we
# map batch outputs back to prompts.
_OUTPUT_SEED_RE = re.compile(r"_(\d+)__\d{3}\.png$")


@dataclass
class _RenderEntry:
    """One request's bookkeeping inside a batched run_many() call."""

    request: RenderCharacterRequest
    out_dir: Path
    prompt: str
    lora_path: Path | None  # None → placeholder LoRA, rendered without LoRA
    lora_weight: float
    steps: int
    guidance_scale: float
    image_uris: list[str] = field(default_factory=list)


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

    version = "1.1.0"
    # Checked with `is True` by the workflow (MagicMock attrs are truthy but not
    # True) — opts Phase A into cross-shot batching via run_many().
    supports_batch = True

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
        return (await self.run_many([req]))[0]

    async def run_many(self, requests: list[RenderCharacterRequest]) -> list[ImageSet]:
        """Render candidates for many requests with ONE model load per LoRA group.

        The single-prompt path pays a full ~40 s text-encoder + 64 GB DiT load
        per candidate image, leaving the GPU near-idle half the time. musubi's
        --from_file batch mode loads the text encoder, DiT and LoRA once and
        then generates every prompt line, so we translate all requests into one
        prompts file per (LoRA path, weight) group — batch mode merges a single
        LoRA set per invocation, so mixed-member casts get one subprocess per
        member instead of one per candidate.

        Failure semantics: batch outputs are decoded and written only after all
        latents are generated, so a mid-batch crash produces no partial PNGs —
        a resumed job simply re-renders the whole group.
        """
        if not requests:
            return []

        entries: list[_RenderEntry] = []
        for req in requests:
            lora_path = self._check_lora(req)
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

            # Per-cast render tuning (from config/casts/<cast>/params.py) overrides
            # the adapter defaults when the request carries it.
            entries.append(
                _RenderEntry(
                    request=req,
                    out_dir=out_dir,
                    prompt=prompt_text,
                    lora_path=None if placeholder else lora_path,
                    lora_weight=(
                        req.lora_weight if req.lora_weight is not None else self._lora_weight
                    ),
                    steps=req.steps if req.steps is not None else self._steps,
                    guidance_scale=(
                        req.guidance_scale
                        if req.guidance_scale is not None
                        else self._guidance_scale
                    ),
                )
            )

        groups: dict[tuple[str, float] | None, list[_RenderEntry]] = {}
        for entry in entries:
            key = (str(entry.lora_path), entry.lora_weight) if entry.lora_path else None
            groups.setdefault(key, []).append(entry)

        for group in groups.values():
            await self._generate_group(group)

        results: list[ImageSet] = []
        for entry in entries:
            log_event(
                logger,
                "render_character_completed",
                member_id=entry.request.member.id,
                image_count=len(entry.image_uris),
            )
            results.append(
                ImageSet(images=entry.image_uris, member_id=entry.request.member.id)
            )
        return results

    @staticmethod
    def _sanitize_prompt_line(prompt: str) -> str:
        """Make a prompt safe for the line-oriented --from_file format.

        Newlines would split the prompt across lines, and " --" starts an
        option token in parse_prompt_line — collapse whitespace and neutralise
        double dashes.
        """
        return " ".join(prompt.split()).replace("--", "—")

    async def _generate_group(self, group: list[_RenderEntry]) -> None:
        """Render every candidate for one LoRA group in a single subprocess."""
        first = group[0]
        # flux_2_generate_image.py treats --save_path as an output DIRECTORY and
        # names files "{time_flag}_{seed}_000.png" — it does not accept a target
        # file path. Render into a scratch dir, then move images into place.
        scratch_dir = Path(tempfile.mkdtemp(prefix="_musubi_batch_", dir=self.work_dir))

        # One line per candidate. The seed doubles as the line's unique ID: it is
        # embedded in the output filename, which is the only reliable way to map
        # batch outputs back to prompts (timestamps can collide within a second).
        lines: list[str] = []
        seed_map: dict[int, tuple[_RenderEntry, int]] = {}
        for entry in group:
            for cand_idx in range(self._num_images):
                seed = len(lines)
                seed_map[seed] = (entry, cand_idx)
                lines.append(
                    f"{self._sanitize_prompt_line(entry.prompt)}"
                    f" --d {seed} --s {entry.steps} --g {entry.guidance_scale}"
                )
        prompts_file = scratch_dir / "prompts.txt"
        prompts_file.write_text("\n".join(lines) + "\n")

        cmd = [
            str(_MUSUBI_PYTHON),
            str(_MUSUBI_SCRIPT),
            "--dit", str(_DIT),
            "--vae", str(_VAE),
            "--text_encoder", str(_TEXT_ENCODER),
            "--from_file", str(prompts_file),
            "--image_size", str(self._width), str(self._height),
            "--infer_steps", str(first.steps),
            "--guidance_scale", str(first.guidance_scale),
            "--save_path", str(scratch_dir),
            "--fp8", "--fp8_scaled",
            "--attn_mode", "flash",
            "--model_version", "dev",
        ]
        if first.lora_path is not None:
            cmd += [
                "--lora_weight", str(first.lora_path),
                "--lora_multiplier", str(first.lora_weight),
            ]

        logger.info(
            "Running musubi-tuner batch inference: %d image(s) across %d request(s) "
            "(lora=%s) — model loads once for the whole batch",
            len(lines), len(group), first.lora_path,
        )
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

        found: set[int] = set()
        for png in scratch_dir.glob("*.png"):
            match = _OUTPUT_SEED_RE.search(png.name)
            if not match:
                continue
            seed = int(match.group(1))
            if seed not in seed_map:
                continue
            entry, cand_idx = seed_map[seed]
            shutil.move(str(png), str(entry.out_dir / f"render_{cand_idx:02d}.png"))
            found.add(seed)

        missing = sorted(set(seed_map) - found)
        if missing:
            raise RuntimeError(
                f"flux_2_generate_image.py exited 0 but produced no image for "
                f"prompt line(s) {missing} in: {scratch_dir}"
            )

        for entry in group:
            entry.image_uris = [
                str(entry.out_dir / f"render_{i:02d}.png") for i in range(self._num_images)
            ]
        shutil.rmtree(scratch_dir, ignore_errors=True)

    def lora_name(self, lora_ref: str) -> str:
        parts = Path(lora_ref).parts
        if parts and parts[0] == "loras":
            parts = parts[1:]
        return "_".join(parts)

    def _check_lora(self, req: RenderCharacterRequest) -> Path:
        # Prefer the explicit per-cast params filename; else derive from lora_ref.
        if req.lora_file:
            path = self._lora_dir / req.lora_file
            if path.exists():
                return path
            raise RuntimeError(
                f"LoRA for '{req.member.name}' not found. Expected: {path} "
                "(from config/casts/<cast>/params.py). Complete Track B "
                "(Flux 2.0 LoRA training) before running render_character."
            )
        name = self.lora_name(req.member.lora_ref)
        for ext in _LORA_EXTENSIONS:
            path = self._lora_dir / f"{name}{ext}"
            if path.exists():
                return path
        expected = self._lora_dir / f"{name}.safetensors"
        raise RuntimeError(
            f"LoRA for '{req.member.name}' not found. Expected: {expected}. "
            "Complete Track B (Flux 2.0 LoRA training) before running render_character."
        )

    def _build_prompt(self, req: RenderCharacterRequest, *, skip_lora: bool) -> str:
        from adapters.render_character.prompt_util import camera_phrase

        parts = []
        if req.trigger.strip():
            parts.append(req.trigger.strip())
        parts += [req.member.visual_descriptor, f"in {req.setting}"]
        if req.action.strip():
            # Shot-specific pose/angle so consecutive shots in the same setting
            # don't all start from an identical-looking base frame.
            parts.append(req.action.strip())
        for other in req.other_members:
            parts.append(f"also present: {other.visual_descriptor}")
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
