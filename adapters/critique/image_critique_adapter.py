"""
VLM image critique adapter — selects the best candidate image for each planned shot.

Flow per shot:
  1. Read last N entries from cast-specific critique_feedback.jsonl (few-shot context).
  2. Send all candidate images as base64 data-URLs to the VLM with a structured rubric.
  3. Parse winner_index + per-candidate scores + reasoning.
  4. Append a feedback entry (without human_override yet — filled in by the approval gate).

Self-learning: the feedback log grows with each run. On the next run, the last
FEEDBACK_WINDOW entries are injected as few-shot examples so the critique learns
which picks the human validated and which were overridden.
"""

import base64
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from core.capabilities.base import CritiqueImages
from core.models.capabilities import (
    ImageCandidateScore,
    ImageCritiqueRequest,
    ImageCritiqueResult,
)
from core.models.common import CostEstimate, HealthStatus
from core.observability import log_event

logger = logging.getLogger(__name__)

FEEDBACK_WINDOW = 5     # how many past entries to use as few-shot context
_FEEDBACK_FILE = "critique_feedback.jsonl"

_SYSTEM_PROMPT = """\
You are a visual quality director for an animated educational kids' series (ages 3–6).
You will receive N candidate still images for a single shot and must select the best one.
Return ONLY valid JSON — no markdown fences, no prose."""

_USER_TEMPLATE = """\
Character visual descriptor:
{cast_descriptor}

Shot context:
{shot_prompt}

You are evaluating {n} candidate images (labelled 0 to {n_minus_1}).

Score each candidate on these dimensions (0.0–1.0):
- character_consistency: does the character match their visual descriptor?
- prompt_adherence:      does the image match the shot's setting and action?
- kids_appropriateness:  safe, clear, and engaging for ages 3–6?
- composition:           good framing, uncluttered, character clearly visible?
- expressiveness:        does the character's expression suit the scene?

{few_shot_block}

Return JSON with exactly this shape:
{{
  "winner_index": 0,
  "candidate_scores": [
    {{"candidate_index": 0, "scores": {{"character_consistency": 0.0, "prompt_adherence": 0.0, "kids_appropriateness": 0.0, "composition": 0.0, "expressiveness": 0.0}}, "reasoning": ""}},
    ...
  ],
  "overall_reasoning": "one sentence explaining the pick"
}}"""


def _encode_image(path: str) -> str:
    """Return a base64 data-URL for a PNG/JPG file."""
    data = Path(path).read_bytes()
    b64 = base64.b64encode(data).decode()
    suffix = Path(path).suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _build_few_shot_block(examples: list[dict]) -> str:
    if not examples:
        return ""
    lines = ["Past critique decisions (learn from these):"]
    for ex in examples:
        pick = ex.get("critique_pick", "?")
        override = ex.get("human_override")
        reason = ex.get("override_reason", "")
        if override is not None and override != pick:
            lines.append(
                f"  • Shot '{ex.get('shot_id')}': critique picked {pick}, "
                f"human overrode to {override} — reason: {reason or 'not given'}"
            )
        else:
            lines.append(
                f"  • Shot '{ex.get('shot_id')}': critique picked {pick}, human confirmed."
            )
    return "\n".join(lines) + "\n\n"


class VlmImageCritiqueAdapter(CritiqueImages):
    """
    Critique N candidate images for a shot using a local VLM (qwen2.5-vl:7b).
    Reads/appends the per-cast feedback log for self-learning.
    """

    version = "1.0.0"

    def __init__(
        self,
        model: str = "qwen2.5-vl:32b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        feedback_log_dir: Path = Path("assets/kids_duo"),
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._log_path = Path(feedback_log_dir) / _FEEDBACK_FILE
        self._temperature = temperature
        self._max_tokens = max_tokens

    # ── Capability interface ──────────────────────────────────────────────────

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
            return HealthStatus(status="down", reason=f"VLM unreachable: {exc}")

    async def estimate_cost(self, req: ImageCritiqueRequest) -> CostEstimate:
        return CostEstimate(amount=0.0, notes="Local VLM — no API cost.")

    async def run(self, req: ImageCritiqueRequest) -> ImageCritiqueResult:
        from openai import AsyncOpenAI

        n = len(req.candidate_uris)
        log_event(logger, "image_critique_started", shot_id=req.shot_id, candidates=n,
                  model=self._model)

        examples = req.feedback_examples or self._load_feedback()
        few_shot = _build_few_shot_block(examples[-FEEDBACK_WINDOW:])

        user_text = _USER_TEMPLATE.format(
            cast_descriptor=req.cast_descriptor,
            shot_prompt=req.shot_prompt,
            n=n,
            n_minus_1=n - 1,
            few_shot_block=few_shot,
        )

        # Build multimodal message: text + one image_url block per candidate.
        content: list[dict] = [{"type": "text", "text": user_text}]
        for i, uri in enumerate(req.candidate_uris):
            content.append({"type": "text", "text": f"Candidate {i}:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": _encode_image(uri)},
            })

        client = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)
        response = await client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )

        raw = response.choices[0].message.content or ""
        result = self._parse(raw, req.candidate_uris)

        log_event(logger, "image_critique_completed", shot_id=req.shot_id,
                  winner=result.winner_index, reasoning=result.overall_reasoning[:80])

        self._append_feedback(req, result)
        return result

    # ── Private helpers ───────────────────────────────────────────────────────

    def _parse(self, raw: str, candidate_uris: list[str]) -> ImageCritiqueResult:
        cleaned = _strip_thinking(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json
                data = json.loads(repair_json(cleaned))
            except Exception:
                logger.warning("image_critique: could not parse VLM response — defaulting to candidate 0")
                return ImageCritiqueResult(winner_index=0, winner_uri=candidate_uris[0])

        winner_index = int(data.get("winner_index", 0))
        winner_index = max(0, min(winner_index, len(candidate_uris) - 1))

        candidate_scores = [
            ImageCandidateScore(
                candidate_index=cs.get("candidate_index", i),
                scores=cs.get("scores", {}),
                reasoning=cs.get("reasoning", ""),
            )
            for i, cs in enumerate(data.get("candidate_scores", []))
        ]

        return ImageCritiqueResult(
            winner_index=winner_index,
            winner_uri=candidate_uris[winner_index],
            candidate_scores=candidate_scores,
            overall_reasoning=data.get("overall_reasoning", ""),
        )

    def _load_feedback(self) -> list[dict]:
        if not self._log_path.exists():
            return []
        entries = []
        for line in self._log_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return entries

    def _append_feedback(self, req: ImageCritiqueRequest, result: ImageCritiqueResult) -> None:
        """Write a new feedback entry. human_override is filled later by the approval gate."""
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "shot_id": req.shot_id,
            "shot_prompt": req.shot_prompt,
            "candidate_uris": req.candidate_uris,
            "critique_pick": result.winner_index,
            "critique_scores": [cs.model_dump() for cs in result.candidate_scores],
            "critique_reasoning": result.overall_reasoning,
            "human_override": None,
            "override_reason": "",
        }
        with self._log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
