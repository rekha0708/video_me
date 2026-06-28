"""
Fish Audio S2 TTS adapter — multi-language voice synthesis (English + Hindi).

Expects a self-hosted Fish Audio S2 server exposing:
  GET  /health            → {"status": "ok"}
  POST /synthesize        → multipart/form-data: text, reference_audio (file),
                            language, format  → raw WAV bytes

Start the server with:
  python scripts/start_fish_s2_server.py --port 8025

**Track B dependency**: each cast member needs a reference WAV (or MP3/FLAC) at
``voice_dir/<name>.(wav|mp3|flac)``.  Missing file → RuntimeError with Track B prompt.
"""

import hashlib
import logging
import wave
from pathlib import Path

from core.capabilities.base import SynthesizeVoice
from core.models.capabilities import AudioTrack, VoiceRequest
from core.models.common import CostEstimate, HealthStatus
from core.observability import log_event

logger = logging.getLogger(__name__)

_VOICE_EXTENSIONS = (".wav", ".mp3", ".flac")
_FALLBACK_WPS: float = 3.0

# Fish Audio S2 language codes
_LANGUAGE_CODES: dict[str, str] = {
    "en": "en",
    "hi": "hi",
}


class FishS2TtsAdapter(SynthesizeVoice):
    """
    synthesize_voice adapter: per-line TTS via a Fish Audio S2 HTTP server.

    Supports English and Hindi. The language is taken from VoiceRequest.language
    (BCP-47 code: "en" or "hi").

    Args:
        work_dir:  Output directory for synthesized WAV files.
        base_url:  Fish Audio S2 server root (e.g. http://localhost:8025).
        voice_dir: Local directory containing per-member reference audio files.
    """

    version = "1.0.0"

    def __init__(
        self,
        work_dir: Path,
        base_url: str = "http://localhost:8025",
        voice_dir: Path = Path("voices"),
    ) -> None:
        self.work_dir = work_dir
        self._base_url = base_url.rstrip("/")
        self._voice_dir = voice_dir

    async def health(self) -> HealthStatus:
        try:
            import httpx
        except ImportError:
            return HealthStatus(status="down", reason="httpx not installed. Run: pip install httpx")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/health")
                resp.raise_for_status()
            return HealthStatus(status="ok")
        except Exception as exc:
            return HealthStatus(
                status="down",
                reason=f"Fish Audio S2 server unreachable at {self._base_url}: {exc}",
            )

    async def estimate_cost(self, req: VoiceRequest) -> CostEstimate:
        return CostEstimate(amount=0.0, notes="Self-hosted Fish Audio S2; GPU compute via rented hardware.")

    async def run(self, req: VoiceRequest) -> AudioTrack:
        voice_path = self._check_voice(req.voice_profile_ref)

        out_dir = self.work_dir / req.speaker_id
        out_dir.mkdir(parents=True, exist_ok=True)

        language_code = _LANGUAGE_CODES.get(req.language, req.language)

        log_event(
            logger,
            "synthesize_voice_started",
            speaker_id=req.speaker_id,
            voice_ref=req.voice_profile_ref,
            text_chars=len(req.text),
            language=req.language,
            expression=req.expression,
        )

        wav_bytes = await self._call_tts(req.text, voice_path, language_code)
        audio_path, duration = self._save_audio(wav_bytes, out_dir, req.text)

        log_event(
            logger,
            "synthesize_voice_completed",
            speaker_id=req.speaker_id,
            duration_sec=duration,
        )

        return AudioTrack(
            uri=str(audio_path),
            duration_sec=duration,
            speaker_id=req.speaker_id,
        )

    # ------------------------------------------------------------------
    # Private helpers (mockable in tests)
    # ------------------------------------------------------------------

    def voice_name(self, voice_profile_ref: str) -> str:
        parts = Path(voice_profile_ref).parts
        if parts and parts[0] == "voices":
            parts = parts[1:]
        return str(Path(*parts))

    def _check_voice(self, ref: str) -> Path:
        name = self.voice_name(ref)
        for ext in _VOICE_EXTENSIONS:
            path = self._voice_dir / f"{name}{ext}"
            if path.exists():
                return path
        expected = self._voice_dir / f"{name}.wav"
        raise RuntimeError(
            f"Voice profile not found for '{ref}'. "
            f"Expected: {expected}. "
            "Complete Track B (synthetic voice design + reference recording) "
            "before running synthesize_voice."
        )

    async def _call_tts(self, text: str, voice_path: Path, language_code: str) -> bytes:
        """POST to the Fish Audio S2 server; return raw WAV bytes."""
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as client:
            with voice_path.open("rb") as ref_file:
                resp = await client.post(
                    f"{self._base_url}/synthesize",
                    data={
                        "text": text,
                        "language": language_code,
                        "format": "wav",
                    },
                    files={"reference_audio": (voice_path.name, ref_file, "audio/wav")},
                )
            resp.raise_for_status()
        return resp.content

    def _save_audio(self, wav_bytes: bytes, out_dir: Path, text: str) -> tuple[Path, float]:
        stem = hashlib.sha1(text.encode()).hexdigest()[:12]
        path = out_dir / f"{stem}.wav"
        path.write_bytes(wav_bytes)
        return path, self._wav_duration(path, text)

    def _wav_duration(self, path: Path, text: str = "") -> float:
        try:
            with wave.open(str(path)) as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            words = len(text.split()) if text else 1
            return max(1.0, words / _FALLBACK_WPS)
