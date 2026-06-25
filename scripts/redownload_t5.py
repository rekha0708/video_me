"""
Re-download models_t5_umt5-xxl-enc-bf16.pth from HuggingFace.

Uses huggingface_hub with HF_HUB_DISABLE_XET=1 to avoid the xet CDN protocol
that caused the previous download corruption. Streams to a temp file then
renames on success to avoid leaving a partial file in place.
"""
import os
import sys
import shutil
import hashlib
import tempfile
import urllib.request
import urllib.error
from pathlib import Path

REPO_ID = "Wan-AI/Wan2.2-I2V-A14B"
FILENAME = "models_t5_umt5-xxl-enc-bf16.pth"
DEST_DIR = Path("/workspace/Wan2.2-I2V-A14B")
EXPECTED_SIZE = 11_361_920_418  # bytes from x-linked-size header
CHUNK = 32 * 1024 * 1024  # 32 MB chunks


def hf_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"


def download(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".tmp")
    headers = {"User-Agent": "python-urllib/3"}

    print(f"Downloading {url}")
    print(f"  -> {tmp}")

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        content_len = int(resp.headers.get("Content-Length", 0))
        print(f"  Content-Length: {content_len:,} bytes ({content_len/1e9:.2f} GB)")

        with open(tmp, "wb") as f:
            downloaded = 0
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                pct = downloaded / max(content_len, 1) * 100
                print(f"  {downloaded/1e9:.2f} GB  ({pct:.1f}%)", flush=True)

    actual = tmp.stat().st_size
    print(f"\nDownloaded {actual:,} bytes")
    if EXPECTED_SIZE and actual != EXPECTED_SIZE:
        print(f"WARNING: expected {EXPECTED_SIZE:,} bytes, got {actual:,}")

    tmp.rename(dest)
    print(f"Saved to {dest}")


def main() -> None:
    dest = DEST_DIR / FILENAME

    if dest.exists():
        size = dest.stat().st_size
        if size == EXPECTED_SIZE:
            print(f"File already exists with correct size ({size:,}). Nothing to do.")
            return
        print(f"Removing corrupt file (size={size:,}, expected={EXPECTED_SIZE:,})")
        dest.unlink()

    # Disable xet protocol so HF redirects to a plain CDN HTTP download
    os.environ["HF_HUB_DISABLE_XET"] = "1"

    url = hf_url(REPO_ID, FILENAME)

    # Follow the HF redirect to get the real CDN URL
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib/3"}, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            final_url = r.geturl()
    except Exception:
        final_url = url  # fall back to original; urllib follows redirects for GET anyway

    download(final_url, dest)
    print("Done.")


if __name__ == "__main__":
    main()
