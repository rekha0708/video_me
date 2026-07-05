"""Shared helpers for building render prompts across render_character adapters."""

# Shot.camera value → image-framing phrase for the render prompt. Framing
# strongly affects Flux composition; without it the model frames arbitrarily.
# "reaction" (cut to the listener) is rendered as a medium reaction shot of the
# on-screen character.
_CAMERA_PHRASES = {
    "close-up": "close-up shot",
    "medium": "medium shot",
    "wide": "wide establishing shot",
    "reaction": "medium reaction shot",
}


def camera_phrase(camera: str) -> str:
    """Map a Shot.camera value to a framing phrase; '' when empty/unknown."""
    return _CAMERA_PHRASES.get((camera or "").strip().lower(), "")
