import json
import logging
import re

from core.capabilities.base import PlanShots
from core.models.capabilities import PlanShotsRequest
from core.models.common import CostEstimate, HealthStatus
from core.models.content import Scene, Script, Shot, Storyboard
from core.observability import log_event

logger = logging.getLogger(__name__)

# Children's speech rate: slow and clear. Used to derive duration from word count.
_WORDS_PER_SEC: float = 2.0
_MIN_SHOT_SEC: float = 5.0
_MAX_SHOT_SEC: float = 8.0

_SYSTEM_PROMPT = """\
You are a shot director for an animated children's short (ages 3–6).
For each line of dialogue you will choose a camera angle, describe what the character is doing,
and list who is visible on screen (speaker + at most ONE other character).
Return ONLY valid JSON — no markdown fences, no prose."""

_USER_TEMPLATE = """\
Cast (member IDs you may use in characters_on_screen):
{cast_block}

Camera options:
  close-up  — speaker fills frame; use for questions, emotional moments, strong expressions
  medium    — speaker from waist up; default for normal dialogue
  reaction  — listener's face; cut to the character being spoken to
  wide      — whole scene; use only for group celebrations or first-establishing shots

Script lines (produce exactly one shot per line, in order):
{lines_block}

Return JSON:
{{
  "shots": [
    {{
      "ref": "<line ref from above>",
      "camera": "<close-up|medium|reaction|wide>",
      "action": "<what the speaker is doing while saying the line>",
      "characters_on_screen": ["<speaker_id>"]
    }}
  ]
}}

Optional per-shot "overlay": for lines whose dialogue states numbers, comparisons,
rankings, or step lists, you MAY add a chart panel shown above the character:
  "overlay": {{
    "kind": "<bar|line|pie|callout>",
    "title": "<short chart title, 6 words max>",
    "labels": ["<2-6 short labels>"],
    "values": [<one number per label>],
    "caption": "<optional unit/context note>"
  }}
Overlay rules:
- Only add an overlay when the line's dialogue actually mentions the data. Most shots have NO overlay.
- "callout" is big text only: title (+ optional caption); omit labels/values.
- Numbers must come from the script's own dialogue — never invent precise statistics.
- At most one overlay per shot; 2-6 data points.

Rules:
- characters_on_screen must contain the speaker and AT MOST one other member ID.
- Never put more than 2 IDs in characters_on_screen.
- ref must match one of the refs listed above exactly."""


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def make_scene_ref(scene_idx: int) -> str:
    return f"scene-{scene_idx + 1}"


def make_line_ref(scene_idx: int, line_idx: int) -> str:
    return f"scene-{scene_idx + 1}-line-{line_idx}"


def estimate_duration(text: str, max_sec: float = _MAX_SHOT_SEC) -> float:
    words = len(text.split())
    return max(_MIN_SHOT_SEC, min(max_sec, words / _WORDS_PER_SEC))


def trim_characters(characters: list[str], speaker: str) -> list[str]:
    """Enforce ≤2 characters; speaker is always first."""
    if not characters:
        return [speaker]
    ordered = [speaker] + [c for c in characters if c != speaker]
    return ordered[:2]


def parse_overlay(raw: object) -> "ShotOverlay | None":
    """Parse an LLM-provided overlay spec; any malformed spec → None (never fatal)."""
    from pydantic import ValidationError

    from core.models.content import ShotOverlay

    if not isinstance(raw, dict):
        return None
    try:
        return ShotOverlay(
            kind=str(raw.get("kind", "")).lower(),
            title=str(raw.get("title", "")).strip(),
            labels=[str(label) for label in (raw.get("labels") or [])][:6],
            values=[float(v) for v in (raw.get("values") or [])][:6],
            caption=str(raw.get("caption", "")).strip(),
        )
    except (ValueError, TypeError, ValidationError):
        logger.warning("Dropping malformed overlay spec: %r", raw)
        return None


def _format_chart_hints(visual_context) -> str:
    """Reference-video chart hints so overlays are grounded — with the rights rule.

    Empty when there is no visual context or no charts were seen.
    """
    segments = getattr(visual_context, "segments", None) if visual_context else None
    if not segments:
        return ""
    charted = [s for s in segments if getattr(s, "chart", "")]
    if not charted:
        return ""
    total = max(segments[-1].end, 0.001)
    lines = ["\nReference-video visual notes:"]
    for seg in charted:
        pct = int(100 * seg.start / total)
        lines.append(f"  - ~{pct}% through the source ({seg.start:.0f}s–{seg.end:.0f}s) it showed: {seg.chart}.")
    lines.append(
        "If a script line covers the same idea, consider attaching an overlay to that shot.\n"
        "RIGHTS RULE: your overlay must be an ORIGINAL chart built from the adapted script's "
        "own wording and numbers — NEVER copy the source chart's exact data, labels, or design."
    )
    return "\n".join(lines)


def _format_prop_hints(visual_context) -> str:
    """Reference-video props so shots can plausibly use them (e.g. the mic or
    guitar the source subject is actually holding) instead of the LLM
    inventing an ungrounded generic gesture.

    Script.scenes[].setting may already carry props folded into its text (see
    core.workflow._build_verbatim_script / adapt_script's _format_visual_block),
    but that's the scene as a whole, not per-shot, and not every code path
    populates it — this is direct, always-on grounding straight from
    visual_context, independent of how the script was built.

    Empty when there is no visual context or no props were seen.
    """
    segments = getattr(visual_context, "segments", None) if visual_context else None
    if not segments:
        return ""
    propped = [s for s in segments if getattr(s, "props", None)]
    if not propped:
        return ""
    total = max(segments[-1].end, 0.001)
    lines = ["\nReference-video props observed (use naturally in action where relevant):"]
    for seg in propped:
        pct = int(100 * seg.start / total)
        lines.append(
            f"  - ~{pct}% through the source ({seg.start:.0f}s-{seg.end:.0f}s): {', '.join(seg.props)}."
        )
    return "\n".join(lines)


def _format_cast_block(cast) -> str:
    return "\n".join(
        f'  "{m.id}" — {m.name} ({m.gender or "?"}): {m.personality}'
        for m in cast.members
    )


def _format_lines_block(script: Script) -> tuple[str, list[tuple[str, str, str]]]:
    """
    Return (formatted_lines_text, flat_line_info_list).
    flat_line_info_list: [(line_ref, speaker_id, line_text), ...]  in script order.
    """
    parts: list[str] = []
    flat: list[tuple[str, str, str]] = []

    for s_idx, scene in enumerate(script.scenes):
        scene_ref = make_scene_ref(s_idx)
        parts.append(f'Scene {s_idx + 1} — setting: "{scene.setting}"')
        for l_idx, line in enumerate(scene.lines):
            ref = make_line_ref(s_idx, l_idx)
            expr = f" (expression: {line.expression})" if line.expression else ""
            parts.append(f'  [{ref}] {line.speaker}: "{line.text}"{expr}')
            flat.append((ref, line.speaker, line.text))

    return "\n".join(parts), flat


class LlmPlanShotsAdapter(PlanShots):
    """
    plan_shots adapter: Script → Storyboard via an OpenAI-compatible LLM.

    The LLM contributes: camera angle, action description, characters_on_screen (≤2).
    The adapter derives: shot_id, scene_ref, setting, dialogue_line_refs, duration_sec.

    Duration is computed from word count (2 words/sec, clamped 5–8s) — more reliable
    than LLM estimates. If the LLM returns fewer shots than lines, defaults fill the gap.
    """

    version = "1.0.0"

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        temperature: float = 0.2,
        max_tokens: int = 16384,
        max_shot_duration_sec: float = _MAX_SHOT_SEC,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_shot_sec = max_shot_duration_sec

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

    async def estimate_cost(self, req: PlanShotsRequest) -> CostEstimate:
        return CostEstimate(amount=0.0, notes="Local/self-hosted LLM; no per-call API cost.")

    async def run(self, req: PlanShotsRequest) -> Storyboard:
        from openai import AsyncOpenAI

        total_lines = sum(len(s.lines) for s in req.script.scenes)
        log_event(
            logger,
            "plan_shots_started",
            model=self._model,
            scene_count=len(req.script.scenes),
            line_count=total_lines,
        )

        client = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)
        messages, flat_lines = self._build_messages(req)

        response = await client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            extra_body={"think": False},
        )

        raw = response.choices[0].message.content or ""
        storyboard = self._parse_response(raw, req, flat_lines)

        log_event(
            logger,
            "plan_shots_completed",
            shot_count=len(storyboard.shots),
        )

        return storyboard

    # ------------------------------------------------------------------
    # Private helpers (mockable in tests)
    # ------------------------------------------------------------------

    def _build_messages(
        self, req: PlanShotsRequest
    ) -> tuple[list[dict], list[tuple[str, str, str]]]:
        lines_block, flat_lines = _format_lines_block(req.script)
        cast_block = _format_cast_block(req.cast)

        user_content = _USER_TEMPLATE.format(
            cast_block=cast_block,
            lines_block=lines_block,
        )

        chart_hints = _format_chart_hints(req.visual_context)
        if chart_hints:
            user_content += "\n" + chart_hints

        prop_hints = _format_prop_hints(req.visual_context)
        if prop_hints:
            user_content += "\n" + prop_hints

        if req.critique_notes:
            notes_block = "\n".join(f"- {n}" for n in req.critique_notes)
            user_content += (
                f"\n\nPREVIOUS CRITIQUE — fix all of the following in your new plan:\n{notes_block}"
            )

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        return messages, flat_lines

    def _parse_response(
        self,
        raw: str,
        req: PlanShotsRequest,
        flat_lines: list[tuple[str, str, str]],
    ) -> Storyboard:
        cleaned = _strip_markdown_fence(raw)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                from json_repair import repair_json
                repaired = repair_json(cleaned)
                data = json.loads(repaired)
            except (ImportError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"LLM returned invalid JSON: {exc}\nRaw (first 500 chars): {raw[:500]}"
                ) from exc

        llm_shots: list[dict] = data.get("shots", [])

        # Index LLM shots by ref for O(1) lookup; fall back to positional matching.
        llm_by_ref: dict[str, dict] = {}
        for i, shot in enumerate(llm_shots):
            ref = shot.get("ref", f"__pos_{i}")
            llm_by_ref[ref] = shot

        shots: list[Shot] = []

        for global_idx, (line_ref, speaker_id, line_text) in enumerate(flat_lines):
            # Decode scene from the line ref (format: "scene-N-line-M")
            parts = line_ref.split("-")
            scene_idx = int(parts[1]) - 1
            scene: Scene = req.script.scenes[scene_idx]

            llm = llm_by_ref.get(line_ref) or (
                llm_shots[global_idx] if global_idx < len(llm_shots) else {}
            )

            camera: str = llm.get("camera") or "medium"
            action: str = llm.get("action") or "speaks"
            raw_chars: list = llm.get("characters_on_screen") or [speaker_id]
            characters = trim_characters(
                [str(c) for c in raw_chars if c], speaker_id
            )

            shots.append(
                Shot(
                    shot_id=f"s{global_idx + 1:02d}",
                    scene_ref=make_scene_ref(scene_idx),
                    characters_on_screen=characters,
                    setting=scene.setting,
                    camera=camera,
                    action=action,
                    dialogue_line_refs=[line_ref],
                    duration_sec=estimate_duration(line_text, self._max_shot_sec),
                    overlay=parse_overlay(llm.get("overlay")),
                )
            )

        return Storyboard(shots=shots)
