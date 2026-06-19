import json
import logging
import re

from core.capabilities.base import AnalyzeContent
from core.models.capabilities import AnalyzeRequest
from core.models.common import CostEstimate, HealthStatus
from core.models.content import ContentMetadata
from core.observability import log_event

logger = logging.getLogger(__name__)

_TRANSCRIPT_MAX_CHARS = 6_000

_SYSTEM_PROMPT = """\
You are an educational video content analyst specialising in early childhood (ages 3–6).
Given a transcript and channel context, extract structured metadata.
Return ONLY valid JSON — no markdown fences, no prose."""

# Braces in the JSON template are doubled so .format() leaves them literal.
_USER_TEMPLATE = """\
Channel context:
  genre: {genre}
  target audience: {audience}
  tone: {tone}
  pedagogy: {pedagogy}

Transcript ({duration:.0f}s, language: {language}):
{transcript}

Return JSON with exactly this structure (fill every field; use null where genuinely unknown):
{{
  "content_genre": "{genre}",
  "topic": "<2-5 word topic label>",
  "tone": "<tone of the video>",
  "hook": "<opening hook or attention-grabbing moment>",
  "structure": ["<beat 1>", "<beat 2>", "<beat 3>"],
  "pacing": "<slow|moderate|fast>",
  "call_to_action": "<what viewers are asked to do, or null>",
  "music_genre": null,
  "visual_style": null,
  "learning_objective": {{
    "concept": "<the single core concept taught>",
    "age_range": "{age_range}",
    "success_phrase": "<complete sentence a child can say after watching>",
    "key_vocabulary": ["<word1>", "<word2>", "<word3>"],
    "reinforcement_count": <how many times the concept is revisited, integer>
  }}
}}"""


def _strip_markdown_fence(text: str) -> str:
    """Remove ```json ... ``` fences that some models add despite instructions."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text


class LlmAnalyzeAdapter(AnalyzeContent):
    """
    analyze_content adapter: transcript → ContentMetadata via an OpenAI-compatible LLM.

    Works with Ollama (local dev), vLLM (rented GPU), or any OpenAI-compatible endpoint.
    language and length_sec are derived directly from the transcript; everything else
    comes from the LLM so those fields are reliable regardless of model quality.

    Args:
        model:       Model name as known to the serving endpoint.
        base_url:    OpenAI-compatible endpoint. Default: local Ollama.
        api_key:     API key (Ollama ignores this; set for cloud providers).
        temperature: Low temperature for deterministic extraction.
        max_tokens:  Response budget — 1024 is ample for this JSON shape.
    """

    version = "1.0.0"

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def health(self) -> HealthStatus:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            return HealthStatus(
                status="down",
                reason="openai package not installed. Run: pip install openai",
            )
        try:
            client = AsyncOpenAI(
                base_url=self._base_url,
                api_key=self._api_key,
                timeout=5.0,
            )
            await client.models.list()
            return HealthStatus(status="ok")
        except Exception as exc:
            return HealthStatus(status="down", reason=f"LLM API unreachable: {exc}")

    async def estimate_cost(self, req: AnalyzeRequest) -> CostEstimate:
        return CostEstimate(amount=0.0, notes="Local/self-hosted LLM; no per-call API cost.")

    async def run(self, req: AnalyzeRequest) -> ContentMetadata:
        from openai import AsyncOpenAI

        log_event(
            logger,
            "analyze_content_started",
            model=self._model,
            segments=len(req.transcript.segments),
        )

        client = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)
        messages = self._build_messages(req)

        response = await client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or ""
        metadata = self._parse_response(raw, req)

        log_event(
            logger,
            "analyze_content_completed",
            topic=metadata.topic,
            concept=(
                metadata.learning_objective.concept
                if metadata.learning_objective
                else None
            ),
        )

        if metadata.learning_objective is None:
            logger.warning(
                "analyze_content: LLM did not return a learning_objective — "
                "adapt_script will lack a LearningObjective to work from."
            )

        return metadata

    # ------------------------------------------------------------------
    # Private helpers (mockable in tests)
    # ------------------------------------------------------------------

    def _build_messages(self, req: AnalyzeRequest) -> list[dict]:
        transcript_text = req.transcript.full_text
        if len(transcript_text) > _TRANSCRIPT_MAX_CHARS:
            transcript_text = transcript_text[:_TRANSCRIPT_MAX_CHARS] + " [truncated]"

        duration = req.transcript.segments[-1].end if req.transcript.segments else 0.0

        profile = req.channel_profile
        audience = profile.target_audience or {}
        age_range = audience.get("age_range", "3-6") if isinstance(audience, dict) else "3-6"

        user_content = _USER_TEMPLATE.format(
            genre=profile.genre_content,
            audience=audience,
            tone=profile.tone,
            pedagogy=profile.pedagogy,
            duration=duration,
            language=req.transcript.language,
            transcript=transcript_text,
            age_range=age_range,
        )

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _parse_response(self, raw: str, req: AnalyzeRequest) -> ContentMetadata:
        cleaned = _strip_markdown_fence(raw)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"LLM returned invalid JSON: {exc}\nRaw (first 500 chars): {raw[:500]}"
            ) from exc

        # Derive these fields from the transcript — more reliable than the LLM's guess.
        data["language"] = req.transcript.language
        data["length_sec"] = (
            int(req.transcript.segments[-1].end)
            if req.transcript.segments
            else 0
        )

        return ContentMetadata.model_validate(data)
