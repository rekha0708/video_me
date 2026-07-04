"""Deterministic story-text parsing (no LLM).

Structured format — one segment per non-blank line:

    0-4.5: Max finds a red leaf in the garden.
    4.5-9: Zoe wonders why leaves change color.

``parse_structured_story`` accepts the text only when EVERY non-blank line
matches and the timeline is monotonic; otherwise it returns None and the
caller falls back to LLM segmentation (``llm_adapter.LlmStorySegmentAdapter``).

``heuristic_segments`` is the last-resort deterministic fallback: sentences
timed at the pipeline's 2 words/sec convention, 5 s floor / 8 s cap per
segment (matches the shot-duration rules in plan_shots).
"""
from __future__ import annotations

import re

from core.models.capabilities import TranscribeResult, TranscriptSegment

_LINE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*:\s*(.+)$")

_WORDS_PER_SEC = 2.0
_MIN_SEGMENT_SEC = 5.0
_MAX_SEGMENT_SEC = 8.0


def parse_structured_story(text: str, language: str = "en") -> TranscribeResult | None:
    """Parse `start-end: text` lines into a TranscribeResult, or None if not structured."""
    segments: list[TranscriptSegment] = []
    prev_end = 0.0
    for line in text.splitlines():
        if not line.strip():
            continue
        match = _LINE_RE.match(line)
        if match is None:
            return None
        start, end = float(match.group(1)), float(match.group(2))
        if end <= start or start < prev_end:
            return None  # non-monotonic or empty span — treat as free text
        segments.append(TranscriptSegment(text=match.group(3).strip(), start=start, end=end))
        prev_end = end

    if not segments:
        return None
    return TranscribeResult(
        segments=segments,
        language=language,
        full_text=" ".join(seg.text for seg in segments),
    )


def heuristic_segments(text: str, language: str = "en") -> TranscribeResult:
    """Deterministic fallback: sentence-split and time at 2 words/sec (5–8 s per segment)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", text.strip()) if s.strip()]
    if not sentences:
        sentences = [text.strip() or "..."]

    segments: list[TranscriptSegment] = []
    cursor = 0.0
    for sentence in sentences:
        duration = len(sentence.split()) / _WORDS_PER_SEC
        duration = min(max(duration, _MIN_SEGMENT_SEC), _MAX_SEGMENT_SEC)
        segments.append(
            TranscriptSegment(text=sentence, start=cursor, end=round(cursor + duration, 2))
        )
        cursor = round(cursor + duration, 2)

    return TranscribeResult(
        segments=segments,
        language=language,
        full_text=" ".join(seg.text for seg in segments),
    )
