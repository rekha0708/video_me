import json
import logging
import re

from core.capabilities.base import AdaptScript
from core.models.capabilities import AdaptScriptRequest
from core.models.common import CostEstimate, HealthStatus
from core.models.content import Script
from core.models.guardrails import SourceRights
from core.observability import log_event

logger = logging.getLogger(__name__)

# Source rights injected by this adapter — always transformed, never verbatim.
_TRANSFORMED_RIGHTS = SourceRights(
    kind="transformed",
    rights_cleared=True,
    notes=(
        "Concept and structure extracted from reference; "
        "no source dialogue, script, or assets reproduced."
    ),
)

_SYSTEM_PROMPT = """\
You are a children's educational script writer for animated shorts aimed at ages 3–6.
Your job: write a warm, simple, encouraging script that teaches ONE concept.
Rules:
- Every line of dialogue: 10 words or fewer.
- Use only simple, concrete words a 3-year-old knows.
- Reinforce the key concept naturally across scenes.
- Assign lines by each character's personality.
- Return ONLY valid JSON — no markdown fences, no prose."""

# Only ask the LLM for scenes. learning_objective, source_rights, mode, and
# caption_text are all injected by the adapter after parsing.
_USER_TEMPLATE = """\
Learning objective:
  concept: {concept}
  age range: {age_range}
  key vocabulary: {vocabulary}
  success phrase: "{success_phrase}"
  reinforce concept: {reinforcement_count} times across the script

Cast — use these exact strings as the "speaker" value in every line:
{cast_block}

Channel tone: {tone}
Pedagogy: {pedagogy}
Target length: ~{target_length_sec}s (≈{line_count} short lines total, 2–3s per line)

Scene guide (write exactly 4 scenes):
  Scene 1 — Hook: c1 asks a simple question about the concept. 2-3 lines.
  Scene 2 — Discovery: c3 introduces and demonstrates the concept. 3-4 lines.
  Scene 3 — Practice: all characters try it together, c2 makes it fun. 3-4 lines.
  Scene 4 — Celebration: everyone cheers, c4 says the success phrase. 2-3 lines.

Return JSON with exactly this structure:
{{
  "scenes": [
    {{
      "setting": "<where this scene takes place>",
      "characters_present": ["<member_id>", ...],
      "lines": [
        {{
          "speaker": "<member_id>",
          "text": "<dialogue — 10 words max>",
          "expression": "<one of that character's expressions>",
          "action": null
        }}
      ]
    }}
  ]
}}"""


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _format_cast_block(cast) -> str:
    lines = []
    for m in cast.members:
        exprs = ", ".join(m.signature_expressions) if m.signature_expressions else "none listed"
        lines.append(
            f'  "{m.id}" — {m.name} [{m.gender or "?"}]: {m.personality} | '
            f"expressions: {exprs}"
        )
    return "\n".join(lines)


def _compute_caption_text(scenes: list[dict]) -> str:
    parts = []
    for scene in scenes:
        for line in scene.get("lines", []):
            text = line.get("text", "").strip()
            if text:
                parts.append(text)
    return " ".join(parts)


class LlmAdaptScriptAdapter(AdaptScript):
    """
    adapt_script adapter: ContentMetadata + Cast → Script via an OpenAI-compatible LLM.

    The LLM only writes scenes. The adapter injects:
      - mode = "transformed" (always)
      - source_rights (always transformed + rights_cleared = True)
      - learning_objective (from ContentMetadata, not re-derived by the LLM)
      - caption_text (computed from the returned lines)

    This keeps the LLM's job focused on creative dialogue and makes the
    guardrail fields tamper-proof regardless of model behaviour.
    """

    version = "1.0.0"

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        temperature: float = 0.4,
        max_tokens: int = 2048,
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

    async def estimate_cost(self, req: AdaptScriptRequest) -> CostEstimate:
        return CostEstimate(amount=0.0, notes="Local/self-hosted LLM; no per-call API cost.")

    async def run(self, req: AdaptScriptRequest) -> Script:
        if req.metadata.learning_objective is None:
            raise ValueError(
                "adapt_script requires ContentMetadata.learning_objective to be set. "
                "Ensure analyze_content returned a complete LearningObjective."
            )

        from openai import AsyncOpenAI

        log_event(
            logger,
            "adapt_script_started",
            model=self._model,
            concept=req.metadata.learning_objective.concept,
            cast_members=len(req.cast.members),
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
        script = self._parse_response(raw, req)

        log_event(
            logger,
            "adapt_script_completed",
            scene_count=len(script.scenes),
            total_lines=sum(len(s.lines) for s in script.scenes),
        )

        return script

    # ------------------------------------------------------------------
    # Private helpers (mockable in tests)
    # ------------------------------------------------------------------

    def _build_messages(self, req: AdaptScriptRequest) -> list[dict]:
        obj = req.metadata.learning_objective  # guaranteed non-None by run()
        profile = req.channel_profile
        target_sec = profile.target_length_sec
        # rough estimate: 2.5s per line
        line_count = max(10, int(target_sec / 2.5))

        user_content = _USER_TEMPLATE.format(
            concept=obj.concept,
            age_range=obj.age_range,
            vocabulary=", ".join(obj.key_vocabulary) or "simple everyday words",
            success_phrase=obj.success_phrase,
            reinforcement_count=obj.reinforcement_count,
            cast_block=_format_cast_block(req.cast),
            tone=profile.tone,
            pedagogy=profile.pedagogy,
            target_length_sec=target_sec,
            line_count=line_count,
        )

        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _parse_response(self, raw: str, req: AdaptScriptRequest) -> Script:
        cleaned = _strip_markdown_fence(raw)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"LLM returned invalid JSON: {exc}\nRaw (first 500 chars): {raw[:500]}"
            ) from exc

        scenes = data.get("scenes", [])

        # Inject all guardrail and derived fields — these are never taken from the LLM.
        data["mode"] = "transformed"
        data["source_rights"] = _TRANSFORMED_RIGHTS.model_dump()
        data["learning_objective"] = req.metadata.learning_objective.model_dump()
        data["caption_text"] = (
            data.get("caption_text") or _compute_caption_text(scenes)
        ).strip() or "."

        return Script.model_validate(data)
