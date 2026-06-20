import asyncio
import base64
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

    The adapter samples frames locally with ffprobe/ffmpeg, embeds those frames
    as data URLs in an OpenAI-compatible multimodal chat request, and persists
    the sampled frame paths on CritiqueResult for audit/debug.
    """

    version = "1.0.0"

    def __init__(
        self,
        model: str = "llava:7b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        work_dir: Path = Path(".local/critique"),
        ffmpeg_bin: str = "ffmpeg",
        ffprobe_bin: str = "ffprobe",
        sample_frames: bool = True,
        frame_count: int = 6,
        frame_width: int = 512,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._work_dir = work_dir
        self._ffmpeg_bin = ffmpeg_bin
        self._ffprobe_bin = ffprobe_bin
        self._sample_frames = sample_frames
        self._frame_count = frame_count
        self._frame_width = frame_width

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

        sampled_frames = await self._sample_video_frames(req.video_uri)

        log_event(
            logger,
            "critique_started",
            model=self._model,
            video_uri=req.video_uri,
            channel_profile_id=req.channel_profile_id,
            sampled_frames=len(sampled_frames),
        )

        client = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)
        response = await client.chat.completions.create(
            model=self._model,
            messages=self._build_messages(req, sampled_frames),
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or ""
        result = self._parse_response(raw)
        result.sampled_frame_uris = [str(path) for path in sampled_frames]

        log_event(
            logger,
            "critique_completed",
            verdict=result.verdict,
            scores=result.scores,
            reasons=result.reasons,
            sampled_frames=result.sampled_frame_uris,
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

    def _build_messages(
        self,
        req: CritiqueRequest,
        sampled_frames: list[Path] | None = None,
    ) -> list[dict]:
        user_content = _USER_TEMPLATE.format(
            video_uri=req.video_uri,
            channel_profile_id=req.channel_profile_id,
            script_summary=_script_summary(req.script),
        )
        content: str | list[dict] = user_content
        if sampled_frames:
            content = [{"type": "text", "text": user_content}]
            for frame in sampled_frames:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _image_data_url(frame),
                            "detail": "low",
                        },
                    }
                )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": content},
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

    async def _sample_video_frames(self, video_uri: str) -> list[Path]:
        if (
            not self._sample_frames
            or self._frame_count <= 0
            or _is_remote_uri(video_uri)
        ):
            return []

        video_path = Path(video_uri)
        duration = await self._probe_duration(video_path)
        timestamps = self._frame_timestamps(duration)
        out_dir = self._work_dir / "sampled_frames" / video_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        frames: list[Path] = []
        for index, timestamp in enumerate(timestamps, start=1):
            frame_path = out_dir / f"frame_{index:02d}.jpg"
            await self._extract_frame(video_path, frame_path, timestamp)
            if frame_path.exists():
                frames.append(frame_path)

        if not frames:
            raise RuntimeError(
                f"Critique frame sampling produced no frames for {video_path}."
            )
        return frames

    def _frame_timestamps(self, duration_sec: float | None) -> list[float]:
        if duration_sec is None or duration_sec <= 0:
            return [float(i) for i in range(self._frame_count)]
        step = duration_sec / (self._frame_count + 1)
        return [round(step * i, 3) for i in range(1, self._frame_count + 1)]

    async def _probe_duration(self, video_path: Path) -> float | None:
        returncode, stdout, _ = await self._run_process(
            self._ffprobe_bin,
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        )
        if returncode != 0:
            return None
        try:
            return float(stdout.decode().strip())
        except ValueError:
            return None

    async def _extract_frame(
        self,
        video_path: Path,
        frame_path: Path,
        timestamp_sec: float,
    ) -> None:
        returncode, _, stderr = await self._run_process(
            self._ffmpeg_bin,
            "-y",
            "-ss", f"{timestamp_sec:.3f}",
            "-i", str(video_path),
            "-frames:v", "1",
            "-vf", f"scale={self._frame_width}:-2",
            "-q:v", "3",
            str(frame_path),
        )
        if returncode != 0:
            tail = stderr.decode(errors="replace")[-1000:]
            raise RuntimeError(
                f"Frame extraction failed for {video_path} at {timestamp_sec:.3f}s "
                f"(exit {returncode}):\n{tail}"
            )

    async def _run_process(self, *args: str) -> tuple[int, bytes, bytes]:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode, stdout, stderr


def _image_data_url(path: Path) -> str:
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{payload}"
