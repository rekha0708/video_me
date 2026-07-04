"""LLM-powered story segmentation adapter.

Takes free-form story text and returns a TranscribeResult with timed segments,
as if the story had been transcribed from a video. Used by the dashboard worker
when the pasted story is not in the structured `start-end: text` format.

Never hard-fails: any LLM/parse problem falls back to the deterministic
``parser.heuristic_segments``.
"""
from __future__ import annotations

import json
import logging

from adapters.story_ingest.parser import heuristic_segments
from core.models.capabilities import TranscribeResult, TranscriptSegment

logger = logging.getLogger(__name__)

_MAX_CHARS = 12_000

_SYSTEM = """\
You segment a children's story into short timed narration segments, as if it
were the transcript of a narrated video.

Rules:
- Split the story into segments of roughly one sentence or beat each.
- Assign start/end times in seconds assuming a narration pace of ~2 words per
  second; each segment should span 5 to 8 seconds.
- Times must be monotonic and non-overlapping, starting at 0.
- Preserve the story's wording — do not rewrite, summarize, or add content.
- Return ONLY valid JSON with this schema:
  {"segments": [{"text": "...", "start": 0.0, "end": 5.0}, ...]}
- No markdown fences, no prose."""

_USER_TEMPLATE = """\
Story text:
{story}

Return the segments JSON."""


class LlmStorySegmentAdapter:
    """Segment free-form story text into a timed TranscribeResult via an LLM."""

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    async def segment(self, text: str, language: str = "en") -> TranscribeResult:
        import httpx

        story = text.strip()
        if len(story) > _MAX_CHARS:
            story = story[:_MAX_CHARS] + "... [truncated]"

        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _USER_TEMPLATE.format(story=story)},
            ],
            "max_tokens": 8192,
            "stream": False,
            "extra_body": {"think": False},
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(f"{self._base_url}/chat/completions", json=payload)
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            logger.warning("Story segmentation LLM call failed (%s); using heuristic split", exc)
            return heuristic_segments(text, language)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json
                data = json.loads(repair_json(raw))
            except Exception:
                logger.warning("LLM returned unrepairable JSON; using heuristic split")
                return heuristic_segments(text, language)

        segments = self._validate_segments(data)
        if segments is None:
            logger.warning("LLM segments invalid; using heuristic split")
            return heuristic_segments(text, language)

        return TranscribeResult(
            segments=segments,
            language=language,
            full_text=" ".join(seg.text for seg in segments),
        )

    @staticmethod
    def _validate_segments(data: object) -> list[TranscriptSegment] | None:
        if not isinstance(data, dict):
            return None
        raw_segments = data.get("segments")
        if not isinstance(raw_segments, list) or not raw_segments:
            return None
        segments: list[TranscriptSegment] = []
        prev_end = 0.0
        for item in raw_segments:
            try:
                seg = TranscriptSegment(
                    text=str(item["text"]).strip(),
                    start=float(item["start"]),
                    end=float(item["end"]),
                )
            except Exception:
                return None
            if not seg.text or seg.end <= seg.start or seg.start < prev_end:
                return None
            segments.append(seg)
            prev_end = seg.end
        return segments
