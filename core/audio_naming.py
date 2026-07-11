"""Shared naming for per-shot TTS output files.

Both TTS adapters (Fish S2, Chatterbox) and the workflow's resume lookup must
agree on the synthesized-audio filename. The name is keyed by shot AND by a
hash of the exact line text: the shot key makes resume per-shot (a cached wav
for shot s01 can never be served to shot s03 — the bug this module fixes), and
the text hash self-invalidates the cache when a re-plan changes the dialogue.
"""

import hashlib


def text_hash_stem(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def shot_audio_filename(shot_id: str, text: str) -> str:
    """'<shot_id>_<text-hash>.wav', or the legacy '<text-hash>.wav' without a shot."""
    stem = text_hash_stem(text)
    return f"{shot_id}_{stem}.wav" if shot_id else f"{stem}.wav"
