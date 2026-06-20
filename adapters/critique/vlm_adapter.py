import json
import logging
import re
from pathlib import Path

from pydantic import ValidationError

from core.capabilities.base import Critique
from core.models.capabilities import CritiqueRequest, CritiqueResult
from core.models.common import CostEstimate, HealthStatus
from core.models.content import Script
from core.observability import log_event

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a children's educational video critic for animated shorts aimed at ages 3-6.
Judge whether the candidate is safe, clear, age-appropriate, and ready for human review.
Return ONLY valid JSON — no markdown fences, no prose."""

_USER_TEMPLATE = """\
Candidate video URI:
{video_uri}

Channel profile id:
{channel_profile_id}

Script summary:
{script_summary}

Evaluate these dimensions with scores from 0.0 to 1.0:
- age_appropriateness
- learning_clarity
- caption_readability
- character_consistency
- engagement

Verdict rules:
- "pass" only if the video is safe for ages 3-6 and good enough for manual review.
- "regenerate" if quality/clarity is fixable by generating another candidate.
- "reject" if there is a safety, rights, scary imagery, or kids-policy concern.

Return JSON with exactly this shape:
{{
  "scores": {{
    "age_appropriateness": 0.0,
    "learning_clarity": 0.0,
    "caption_readability": 0.0,
    "character_consistency": 0.0,
    "engagement": 0.0
  }},
  "verdict": "pass",
  "reasons": ["short reason"],
  "suggested_param_overrides": {{}}
}}"""


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _is_remote_uri(uri: str) -> bool:
    return uri.startswith(("http://", "https://", "s3://"))


def _script_line_count(script: Script) -> int:
    return sum(len(scene.lines) for scene in script.scenes)


def _script_summary(script: Script, max_chars: int = 3000) -> str:
    parts: list[str] = [
        f"mode: {script.mode}",
        f"objective: {script.learning_objective.success_phrase}",
        f"caption_text: {script.caption_text}",
        "dialogue:",
    ]
    for scene_index, scene in enumerate(script.scenes, start=1):
        parts.append(f"scene {scene_index}: {scene.setting}")
        for line in scene.lines:
            parts.append(f"- {line.speaker}: {line.text}")
    text = "\n".join(parts)
    return text if len(text) <= max_chars else text[:max_chars] + "\n[truncated]"


class VlmCritiqueAdapter(Critique):
    """
    critique adapter: automated preflight + OpenAI-compatible VLM/LLM judgment.

    The adapter currently sends the candidate video URI and script to an
    OpenAI-compatible endpoint. Some local VLM services may need a wrapper that
    extracts frames or accepts video input; that wrapper can sit behind this same
    JSON contract without changing the workflow.
    """

    version = "1.0.0"

    def __init__(
        self,
        model: str = "llava:7b",
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
            return HealthStatus(status="down", reason=f"Critique API unreachable: {exc}")

    async def estimate_cost(self, req: CritiqueRequest) -> CostEstimate:
        return CostEstimate(
            amount=0.0,
            notes="Local/self-hosted critique model; GPU compute cost via rented hardware.",
        )

    async def run(self, req: CritiqueRequest) -> CritiqueResult:
        preflight = self._preflight(req)
        if preflight is not None:
            return preflight

        from openai import AsyncOpenAI

        log_event(
            logger,
            "critique_started",
            model=self._model,
            video_uri=req.video_uri,
            channel_profile_id=req.channel_profile_id,
        )

        client = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)
        response = await client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(req),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or ""
        result = self._parse_response(raw)

        log_event(
            logger,
            "critique_completed",
            verdict=result.verdict,
            scores=result.scores,
            reasons=result.reasons,
        )

        return result

    # ------------------------------------------------------------------
    # Private helpers (mockable in tests)
    # ------------------------------------------------------------------

    def _preflight(self, req: CritiqueRequest) -> CritiqueResult | None:
        if not _is_remote_uri(req.video_uri) and not Path(req.video_uri).exists():
            raise FileNotFoundError(
                f"Candidate video not found: {req.video_uri}. "
                "assemble_video must complete before critique."
            )

        if not req.script.scenes or _script_line_count(req.script) == 0:
            return CritiqueResult(
                scores={"script_completeness": 0.0},
                verdict="reject",
                reasons=["Script has no scenes or dialogue lines."],
                suggested_param_overrides={},
            )

        return None

    def _build_messages(self, req: CritiqueRequest) -> list[dict]:
        user_content = _USER_TEMPLATE.format(
            video_uri=req.video_uri,
            channel_profile_id=req.channel_profile_id,
            script_summary=_script_summary(req.script),
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _parse_response(self, raw: str) -> CritiqueResult:
        cleaned = _strip_markdown_fence(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Critique model returned invalid JSON: {exc}\n"
                f"Raw (first 500 chars): {raw[:500]}"
            ) from exc

        try:
            return CritiqueResult.model_validate(data)
        except ValidationError as exc:
            raise RuntimeError(f"Critique model returned invalid result shape: {exc}") from exc
