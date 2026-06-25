"""
Bootstrap reference voice WAVs for Max and Zoe using ChatterboxTTS.
Run with: /workspace/venv/bin/python3 scripts/generate_voices.py

No reference audio required — generates zero-shot voices then saves them
as the Track B reference files:
  voices/kids_duo/max.wav
  voices/kids_duo/zoe.wav
"""

import io
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
VOICE_DIR = ROOT / "voices" / "kids_duo"
VOICE_DIR.mkdir(parents=True, exist_ok=True)

# ~20 seconds of speech per character — energetic but simple lines
MAX_TEXT = (
    "Hi there! I'm Max! I love teaching letters. "
    "Ready to learn something super cool today? "
    "Let's start with the letter A! "
    "A is for apple, A is for ant, A is for amazing — just like you! "
    "You're doing great! Let's keep going!"
)

ZOE_TEXT = (
    "Hi! I'm Zoe! I love cooking and painting and all sorts of fun things. "
    "Watch me! First you put the flour in the bowl, then you stir, stir, stir. "
    "See? Easy! Now you try. "
    "You did it! I knew you could do it! Let's make something together!"
)

CHARACTERS = [
    ("max", MAX_TEXT, 0.6),   # slightly higher exaggeration for energetic big-kid
    ("zoe", ZOE_TEXT, 0.7),   # more expressive for confident toddler
]


def generate(model, text: str, exaggeration: float, out_path: Path) -> None:
    import inspect
    import torchaudio

    sig = inspect.signature(model.generate)
    kwargs = {"exaggeration": exaggeration} if "exaggeration" in sig.parameters else {}

    print(f"  Generating {len(text)} chars (exaggeration={exaggeration}) ...", flush=True)
    wav = model.generate(text, **kwargs)

    buf = io.BytesIO()
    torchaudio.save(buf, wav, model.sr, format="wav")
    buf.seek(0)
    out_path.write_bytes(buf.read())
    print(f"  Saved → {out_path}", flush=True)


def main() -> None:
    import torch
    from chatterbox.tts import ChatterboxTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ChatterboxTTS on {device} ...", flush=True)
    model = ChatterboxTTS.from_pretrained(device=device)
    print(f"Model ready (sample rate {model.sr} Hz)\n", flush=True)

    for name, text, exaggeration in CHARACTERS:
        out = VOICE_DIR / f"{name}.wav"
        if out.exists():
            print(f"[{name}] {out} already exists — skipping (delete to regenerate)")
            continue
        print(f"[{name}]")
        generate(model, text, exaggeration, out)
        print()

    print("Done. Run: python -m scripts.check_track_b")


if __name__ == "__main__":
    main()
