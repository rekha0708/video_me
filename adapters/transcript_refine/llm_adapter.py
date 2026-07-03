"""LLM-powered transcript refinement adapter.

Takes a raw transcript dict (segments + metadata) plus user correction notes,
and returns a corrected transcript dict in the same shape.

Used by the dashboard worker's transcript review gate when the operator rejects
the raw Whisper output with notes.
"""
from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

_MAX_CHARS = 12_000

_SYSTEM = """\
You are a transcript editor. You receive a raw auto-generated transcript and
operator correction notes. Your job is to return a corrected transcript.

Rules:
- Fix only what the notes describe (typos, wrong words, speaker labels, etc.)
- Preserve the segment structure (start/end times are approximate — keep them)
- Keep the same JSON schema: {"segments": [{text, start, end}, ...], "language": "...", "duration": ...}
- Return ONLY valid JSON — no markdown fences, no prose."""

_USER_TEMPLATE = """\
Operator notes:
{notes}

Raw transcript JSON:
{transcript_json}

Return the corrected transcript JSON."""


class LlmTranscriptRefineAdapter:
    """Refine a raw transcript using an LLM + operator notes."""

    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    async def refine(
        self,
        transcript: dict,
        notes: str,
    ) -> dict:
        """Return a corrected transcript dict.

        Args:
            transcript: The raw transcript dict from the artifact store.
            notes: Operator correction notes from the rejection form.
        """
        import httpx

        transcript_json = json.dumps(transcript, ensure_ascii=False)
        if len(transcript_json) > _MAX_CHARS:
            transcript_json = transcript_json[:_MAX_CHARS] + "... [truncated]"

        messages = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": _USER_TEMPLATE.format(
                    notes=notes.strip(),
                    transcript_json=transcript_json,
                ),
            },
        ]

        payload = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 8192,
            "stream": False,
            "extra_body": {"think": False},
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"].strip()

        try:
            corrected = json.loads(raw)
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json
                corrected = json.loads(repair_json(raw))
            except Exception:
                logger.warning("LLM returned non-JSON for transcript refine; keeping original")
                return transcript

        if not isinstance(corrected, dict) or "segments" not in corrected:
            logger.warning("LLM response missing 'segments'; keeping original transcript")
            return transcript

        return corrected
