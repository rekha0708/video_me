"""analyze_visuals adapter: describe the source video's settings per transcript segment.

Samples one frame at each segment's midpoint (bounded by ``max_frames``), sends
them to a multimodal LLM (qwen3.6:35b) as ``image_url`` data URLs, and returns a
``VisualContext`` of per-segment {setting, props}. Grounding is best-effort: any
failure (no video, remote/``story://`` URI, ffmpeg error, bad JSON) yields an
EMPTY VisualContext rather than raising — the pipeline then falls back to
LLM-invented settings.

Frame-sampling mirrors adapters/critique/vlm_adapter.py (there is no shared frame
util today; copying keeps each adapter self-contained per the codebase style).
"""
import asyncio
import base64
import json
import logging
import re
from pathlib import Path

from core.capabilities.base import AnalyzeVisuals
from core.models.capabilities import (
    AnalyzeVisualsRequest,
    TranscriptSegment,
    VisualContext,
    VisualSegment,
)
from core.models.common import CostEstimate, HealthStatus
from core.observability import log_event

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You describe the physical setting shown in frames from a reference video.
For each numbered frame you are told the time range it covers. Report ONLY what is
visibly on screen — the location/background and notable props — in plain, concrete
words. Do not guess a story, characters' names, or anything not visible.
Return ONLY valid JSON — no markdown fences, no prose."""

_USER_TEMPLATE = """\
Each image below is one frame, in order. The time ranges are:
{ranges_block}

Return JSON with exactly this shape:
{{
  "segments": [
    {{"start": 0.0, "end": 5.0, "setting": "<location/background visible>", "props": ["<object>", ...],
      "chart": "<if a chart, graph, table, or diagram is visible: one short phrase like 'bar chart comparing X and Y'; otherwise empty string>"}}
  ],
  "summary": "<one sentence listing the distinct locations seen across the video>"
}}
Produce exactly one segment per time range, in the same order."""


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _is_remote_uri(uri: str) -> bool:
    return uri.startswith(("http://", "https://", "s3://", "story://"))


def _image_data_url(path: Path) -> str:
    media_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{media_type};base64,{payload}"


class VlmAnalyzeVisualsAdapter(AnalyzeVisuals):
    """Describe per-segment settings from the source video via a multimodal LLM."""

    version = "1.0.0"

    def __init__(
        self,
        model: str = "qwen3.6:35b",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        work_dir: Path = Path(".local/visuals"),
        ffmpeg_bin: str = "ffmpeg",
        ffprobe_bin: str = "ffprobe",
        max_frames: int = 8,
        frame_width: int = 512,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._api_key = api_key
        self._work_dir = work_dir
        self._ffmpeg_bin = ffmpeg_bin
        self._ffprobe_bin = ffprobe_bin
        self._max_frames = max_frames
        self._frame_width = frame_width
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def health(self) -> HealthStatus:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            return HealthStatus(status="down", reason="openai package not installed.")
        try:
            client = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key, timeout=5.0)
            await client.models.list()
            return HealthStatus(status="ok")
        except Exception as exc:
            return HealthStatus(status="down", reason=f"VLM API unreachable: {exc}")

    async def estimate_cost(self, req: AnalyzeVisualsRequest) -> CostEstimate:
        return CostEstimate(amount=0.0, notes="Local/self-hosted multimodal LLM; no per-call cost.")

    async def run(self, req: AnalyzeVisualsRequest) -> VisualContext:
        # Best-effort: any reason we can't sample the source video → empty context.
        if _is_remote_uri(req.video_uri) or not req.segments:
            log_event(logger, "analyze_visuals_skipped", reason="no local video or no segments")
            return VisualContext()
        video_path = Path(req.video_uri)
        if not video_path.exists():
            log_event(logger, "analyze_visuals_skipped", reason="video file missing",
                      video_uri=req.video_uri)
            return VisualContext()

        buckets = self._bucket_segments(req.segments)
        try:
            frames = await self._sample_frames(video_path, buckets)
        except Exception as exc:  # ffmpeg/ffprobe failure — never fatal
            logger.warning("analyze_visuals frame sampling failed (%s); returning empty", exc)
            return VisualContext()
        if not frames:
            return VisualContext()

        log_event(logger, "analyze_visuals_started", model=self._model, frames=len(frames))
        try:
            raw = await self._call_vlm(buckets, frames)
            context = self._parse_response(raw, buckets)
        except Exception as exc:
            logger.warning("analyze_visuals VLM/parse failed (%s); returning empty", exc)
            return VisualContext()

        log_event(logger, "analyze_visuals_completed", segments=len(context.segments))
        return context

    # ------------------------------------------------------------------
    # Private helpers (mockable in tests)
    # ------------------------------------------------------------------

    def _bucket_segments(
        self, segments: list[TranscriptSegment]
    ) -> list[tuple[float, float]]:
        """Reduce segments to at most max_frames time ranges (merge adjacent if needed)."""
        ranges = [(s.start, s.end) for s in segments]
        if len(ranges) <= self._max_frames:
            return ranges
        # Merge into max_frames evenly-sized groups, preserving order.
        group = -(-len(ranges) // self._max_frames)  # ceil
        merged: list[tuple[float, float]] = []
        for i in range(0, len(ranges), group):
            chunk = ranges[i:i + group]
            merged.append((chunk[0][0], chunk[-1][1]))
        return merged

    async def _sample_frames(
        self, video_path: Path, buckets: list[tuple[float, float]]
    ) -> list[Path]:
        out_dir = self._work_dir / "sampled_frames" / video_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        frames: list[Path] = []
        for index, (start, end) in enumerate(buckets, start=1):
            midpoint = (start + end) / 2.0
            frame_path = out_dir / f"seg_{index:02d}.jpg"
            await self._extract_frame(video_path, frame_path, midpoint)
            if frame_path.exists():
                frames.append(frame_path)
        return frames

    async def _extract_frame(
        self, video_path: Path, frame_path: Path, timestamp_sec: float
    ) -> None:
        returncode, _, stderr = await self._run_process(
            self._ffmpeg_bin, "-y",
            "-ss", f"{max(0.0, timestamp_sec):.3f}",
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
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode, stdout, stderr

    async def _call_vlm(
        self, buckets: list[tuple[float, float]], frames: list[Path]
    ) -> str:
        from openai import AsyncOpenAI

        ranges_block = "\n".join(
            f"  Frame {i}: {start:.1f}s–{end:.1f}s"
            for i, (start, end) in enumerate(buckets, start=1)
        )
        content: list[dict] = [
            {"type": "text", "text": _USER_TEMPLATE.format(ranges_block=ranges_block)}
        ]
        for frame in frames:
            content.append({
                "type": "image_url",
                "image_url": {"url": _image_data_url(frame), "detail": "low"},
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
            extra_body={"think": False},
        )
        return response.choices[0].message.content or ""

    def _parse_response(
        self, raw: str, buckets: list[tuple[float, float]]
    ) -> VisualContext:
        cleaned = _strip_markdown_fence(raw)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            from json_repair import repair_json
            data = json.loads(repair_json(cleaned))

        segments: list[VisualSegment] = []
        for i, item in enumerate(data.get("segments", [])):
            setting = str(item.get("setting", "")).strip()
            if not setting:
                continue
            # Prefer our known time range; fall back to whatever the model returned.
            if i < len(buckets):
                start, end = buckets[i]
            else:
                start, end = float(item.get("start", 0.0)), float(item.get("end", 0.0))
            props = [str(p).strip() for p in (item.get("props") or []) if str(p).strip()]
            chart = str(item.get("chart", "")).strip()
            segments.append(
                VisualSegment(start=start, end=end, setting=setting, props=props, chart=chart)
            )

        return VisualContext(segments=segments, summary=str(data.get("summary", "")).strip())
