# Track B Voice References

Place final `kids_duo` reference voices here:

```text
voices/
  kids_duo/
    max.wav
    zoe.wav
```

Audio requirements:

- 10-30 seconds of clear single-speaker speech per character.
- PCM WAV preferred, 16 kHz or higher.
- No background music, room noise, or other speakers.
- Designed synthetic or consented performer voice only; do not clone an
  identifiable real child or public figure.

The adapter also accepts `.mp3` or `.flac`, but `.wav` is preferred. These
binary files are intentionally ignored by git.

For development only, provisional local WAVs can be generated with macOS `say`.
The current local test pick is documented in `VOICE_SELECTION.md`.

Run this from the repo root after placing voices:

```bash
python -m scripts.check_track_b
```
