"""
Upload raw reference images for a character into the kohya_ss training dataset.

Usage:
    python -m scripts.upload_reference_images --character max image1.png image2.jpg ...
    python -m scripts.upload_reference_images --character zoe ~/photos/*.jpg

What it does:
  1. Copies each source image into assets/kids_duo/training/images/<character>/
  2. Converts non-PNG files to PNG (requires Pillow).
  3. Writes a .txt caption file next to each image using the character's caption template.
  4. Prints the final image count so you can verify 20+ images are present before training.

After running, kick off training:
    accelerate launch flux_train_network.py \\
        --config_file assets/kids_duo/training/kohya_config.toml
"""

import argparse
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAINING_DIR = _REPO_ROOT / "assets" / "kids_duo" / "training" / "images"
_CAPTION_TEMPLATES = {
    "max": _REPO_ROOT / "assets" / "kids_duo" / "training" / "caption_template_max.txt",
    "zoe": _REPO_ROOT / "assets" / "kids_duo" / "training" / "caption_template_zoe.txt",
}

_SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def _load_caption_template(character: str) -> str:
    path = _CAPTION_TEMPLATES.get(character)
    if path and path.exists():
        return path.read_text().strip()
    # Fallback trigger token if no template exists yet.
    return f"kids_duo_{character}, cartoon character, children's animation style"


def _to_png(src: Path, dest: Path) -> None:
    if src.suffix.lower() == ".png":
        shutil.copy2(src, dest)
        return
    try:
        from PIL import Image
        with Image.open(src) as img:
            img.convert("RGB").save(dest, "PNG")
    except ImportError:
        print("Pillow not installed — copying file as-is (rename manually to .png if needed).")
        shutil.copy2(src, dest.with_suffix(src.suffix))


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload reference images for LoRA training.")
    parser.add_argument(
        "--character", required=True, choices=["max", "zoe"],
        help="Which character these images are for.",
    )
    parser.add_argument(
        "--caption", default=None,
        help="Override caption text (defaults to caption template file).",
    )
    parser.add_argument("images", nargs="+", help="Source image files to import.")
    args = parser.parse_args()

    out_dir = _TRAINING_DIR / args.character
    out_dir.mkdir(parents=True, exist_ok=True)

    caption = args.caption or _load_caption_template(args.character)
    sources = [Path(p) for p in args.images]

    existing = list(out_dir.glob("*.png"))
    start_idx = len(existing) + 1

    imported = 0
    for i, src in enumerate(sources, start=start_idx):
        if not src.exists():
            print(f"  SKIP (not found): {src}")
            continue
        if src.suffix.lower() not in _SUPPORTED_EXTS:
            print(f"  SKIP (unsupported format): {src}")
            continue

        dest_png = out_dir / f"{args.character}_{i:03d}.png"
        _to_png(src, dest_png)

        # Write caption alongside the image.
        dest_png.with_suffix(".txt").write_text(caption)
        print(f"  OK: {src.name} → {dest_png.name}")
        imported += 1

    total = len(list(out_dir.glob("*.png")))
    print(f"\nImported {imported} image(s). Total in dataset: {total}")
    if total < 15:
        print("WARNING: Fewer than 15 images — aim for 20+ for best LoRA quality.")
    else:
        print("Dataset looks good. Run kohya_ss training when ready.")


if __name__ == "__main__":
    main()
