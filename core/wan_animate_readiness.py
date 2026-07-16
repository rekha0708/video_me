"""Pure filesystem readiness checks shared by Animate API and worker code."""

from __future__ import annotations

from pathlib import Path


WAN_ANIMATE_REQUIRED_MODEL_FILES = (
    "config.json",
    "diffusion_pytorch_model-00001-of-00004.safetensors",
    "diffusion_pytorch_model-00002-of-00004.safetensors",
    "diffusion_pytorch_model-00003-of-00004.safetensors",
    "diffusion_pytorch_model-00004-of-00004.safetensors",
    "diffusion_pytorch_model.safetensors.index.json",
    "models_t5_umt5-xxl-enc-bf16.pth",
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    "Wan2.1_VAE.pth",
    "relighting_lora.ckpt",
    "google/umt5-xxl/spiece.model",
    "xlm-roberta-large/sentencepiece.bpe.model",
    "process_checkpoint/det/yolov10m.onnx",
    "process_checkpoint/pose2d/vitpose_h_wholebody.onnx",
    "process_checkpoint/sam2/sam2_hiera_large.pt",
)


def wan_animate_component_ready(path: Path) -> bool:
    """Check one required Animate checkpoint path is present and non-empty.

    ONNX external-data checkpoints (e.g. process_checkpoint/pose2d/vitpose_h_wholebody.onnx)
    ship upstream on HF Hub as a directory of tensor blobs plus an end2end.onnx graph
    file, not a single file — mirror the os.path.isdir() resolution that
    wan/modules/animate/preprocess/pose2d.py:SimpleOnnxInference already does at
    load time, so a correctly-installed directory isn't flagged as missing.
    """

    try:
        if path.is_dir():
            graph = path / "end2end.onnx"
            return graph.is_file() and graph.stat().st_size > 0
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def wan_animate_model_readiness(model_dir: str | Path) -> tuple[bool, str]:
    """Reject partial Animate snapshots rather than treating a directory as ready."""

    root = Path(model_dir).expanduser()
    missing: list[str] = []
    for relative in WAN_ANIMATE_REQUIRED_MODEL_FILES:
        path = root / relative
        if not wan_animate_component_ready(path):
            missing.append(relative)
    if missing:
        shown = ", ".join(missing[:4])
        suffix = f" (+{len(missing) - 4} more)" if len(missing) > 4 else ""
        return (
            False,
            f"Wan2.2-Animate-14B is incomplete: {shown}{suffix}. Run "
            "setup_gpu.sh --with-wan-animate to resume and verify the download.",
        )
    return True, "Wan2.2-Animate-14B model files are installed."


def wan_flux_retarget_readiness(model_dir: str | Path) -> tuple[bool, str]:
    """Check the optional FLUX.1 Kontext tree used only for pose retargeting.

    A directory alone is not sufficient: interrupted Hugging Face downloads can
    leave configs and cache metadata behind. Require each inference component
    and at least one local weight shard before enabling the dashboard control.
    """

    root = (
        Path(model_dir).expanduser()
        / "process_checkpoint"
        / "FLUX.1-Kontext-dev"
    )
    required = (
        root / "model_index.json",
        root / "transformer" / "config.json",
        root / "text_encoder_2" / "config.json",
        root / "vae" / "config.json",
    )
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]

    def has_weights(component: str) -> bool:
        directory = root / component
        return any(directory.glob("*.safetensors")) or any(directory.glob("*.bin"))

    for component in ("transformer", "text_encoder_2", "vae"):
        if not has_weights(component):
            missing.append(f"{component} weights")

    if missing:
        return (
            False,
            "Optional FLUX.1 Kontext retargeting is not installed completely "
            f"({', '.join(missing)}). Run setup_gpu.sh "
            "--with-wan-animate-flux-retarget.",
        )
    return True, "FLUX.1 Kontext pose-retarget refinement is installed."


__all__ = [
    "WAN_ANIMATE_REQUIRED_MODEL_FILES",
    "wan_animate_component_ready",
    "wan_animate_model_readiness",
    "wan_flux_retarget_readiness",
]
