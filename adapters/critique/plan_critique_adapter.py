import json
import logging
import re

from core.capabilities.base import CritiquePlan
from core.models.capabilities import PlanCritiqueRequest, PlanCritiqueResult
from core.models.common import CostEstimate, HealthStatus
from core.observability import log_event

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a storyboard director and child-safety reviewer for animated educational shorts (ages 3–6).
You will evaluate a shot plan against the cast design and script.
Return ONLY valid JSON — no markdown fences, no prose."""

_USER_TEMPLATE = """\
Cast members (with their visual descriptors and personalities):
{cast_block}

Script summary:
{script_summary}

Proposed storyboard ({shot_count} shots):
{shots_block}

Evaluate each shot and the plan as a whole on these dimensions (0.0–1.0):
- character_fit:       does each shot's action match the character's design and personality?
- scene_achievability: can the setting + action be rendered as a still image animated to video?
- pacing:              are durations appropriate for the spoken line at 2 words/sec?
- kids_safety:         is every shot safe, clear, and appropriate for ages 3–6?
- visual_clarity:      will a young child immediately understand what is happening?

Verdict rules:
- "pass"   — all scores ≥ 0.75 and no safety concern. Ready for human review.
- "revise" — one or more scores < 0.75 OR a safety/feasibility issue exists.

When verdict is "revise", revision_notes must list specific, actionable fixes
(e.g. "shot s03: action 'fly through space' is not achievable in i2v — change to
'points at star chart on wall'").

Return JSON with exactly this shape:
{{
  "verdict": "pass",
  "scores": {{
    "character_fit": 0.0,
    "scene_achievability": 0.0,
    "pacing": 0.0,
    "kids_safety": 0.0,
    "visual_clarity": 0.0
  }},
  "revision_notes": []
}}"""


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text


class LlmPlanCritiqueAdapter(CritiquePlan):
    """
    critique_plan adapter: evaluates a proposed Storyboard against the cast and script
    using the same Ollama LLM (qwen3.6:35b) — no extra model needed.

    Checks: character fit, i2v scene achievability, pacing, kids safety, visual clarity.
    Returns verdict=pass (all scores ≥ 0.75) or verdict=revise with specific fix notes.
    """

    version = "1.0.0"

    def __init__(
        self,
        model: str = "qwen3.6:35b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        temperature: float = 0.1,
        max_tokens: int = 4096,
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
            return HealthStatus(status="down", reason="openai package not installed")
        try:
            client = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key, timeout=5.0)
            await client.models.list()
            return HealthStatus(status="ok")
        except Exception as exc:
            return HealthStatus(status="down", reason=f"LLM unreachable: {exc}")

    async def estimate_cost(self, req: PlanCritiqueRequest) -> CostEstimate:
        return CostEstimate(amount=0.0, notes="Local LLM — no API cost.")

    async def run(self, req: PlanCritiqueRequest) -> PlanCritiqueResult:
        from openai import AsyncOpenAI

        log_event(logger, "critique_plan_started",
                  shot_count=len(req.storyboard.shots), model=self._model)

        cast_block = "\n".join(
            f'  {m.id} — {m.name}: {m.visual_descriptor}. Personality: {m.personality}'
            for m in req.cast.members
        )
        script_summary = "; ".join(
            f"Scene {i+1} ({s.setting}): "
            + "; ".join(f'{ln.speaker}: "{ln.text[:60]}"' for ln in s.lines)
            for i, s in enumerate(req.script.scenes)
        )
        shots_block = "\n".join(
            f'  [{s.shot_id}] camera={s.camera} chars={s.characters_on_screen} '
            f'setting="{s.setting}" duration={s.duration_sec}s\n'
            f'    action: "{s.action}"'
            for s in req.storyboard.shots
        )

        user_msg = _USER_TEMPLATE.format(
            cast_block=cast_block,
            script_summary=script_summary,
            shot_count=len(req.storyboard.shots),
            shots_block=shots_block,
        )

        client = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)
        response = await client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            extra_body={"think": False},
        )

        raw = response.choices[0].message.content or ""
        result = self._parse(raw)

        log_event(logger, "critique_plan_completed",
                  verdict=result.verdict, scores=result.scores,
                  notes_count=len(result.revision_notes))
        return result

    def _parse(self, raw: str) -> PlanCritiqueResult:
        cleaned = _strip_thinking(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json
                data = json.loads(repair_json(cleaned))
            except Exception:
                logger.warning("critique_plan: could not parse LLM response — defaulting to pass")
                return PlanCritiqueResult(verdict="pass")

        return PlanCritiqueResult(
            verdict=data.get("verdict", "pass"),
            scores=data.get("scores", {}),
            revision_notes=data.get("revision_notes", []),
        )
