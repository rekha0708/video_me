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

# Chatterbox exaggeration nudges per expression keyword (0 = flat, 1 = maximum).
# Base value is the constructor default (0.5). Only override when the keyword matches.
_EXPRESSION_EXAGGERATION: dict[str, float] = {
    "excited": 0.7,
    "surprised": 0.7,
    "frightened": 0.8,
    "crying": 0.8,
    "sad": 0.6,
    "whispering": 0.3,
    "whisper": 0.3,
}

# Fallback speech rate when the response isn't a parseable WAV (words per second).
_FALLBACK_WPS: float = 3.0


class TtsAdapter(SynthesizeVoice):
    """
    synthesize_voice adapter: per-line TTS via a Chatterbox-compatible HTTP API.

    The service must expose:
      GET  /health            → {"status": "ok"}
      POST /synthesize        → multipart/form-data: text, reference_audio (file),
                                language, exaggeration  → raw WAV bytes

    **Track B dependency**: each cast member needs a reference WAV (or MP3/FLAC) at
    ``voice_dir/<name>.(wav|mp3|flac)`` before this adapter can run.
    The adapter raises a clear Track B error if the file is missing.

    One instance per job — ``work_dir`` should be job-scoped.

    Args:
        work_dir:     Output directory for synthesized WAV files.
        base_url:     TTS service root (e.g. http://localhost:8020).
        voice_dir:    Local directory containing per-member reference audio files.
        language:     BCP-47 language code passed to the TTS service.
        exaggeration: Baseline expressiveness strength (0 flat → 1 maximum).
                      Per-expression overrides in _EXPRESSION_EXAGGERATION take
                      precedence when the expression keyword matches.
    """

    version = "1.0.0"

    def __init__(
        self,
        work_dir: Path,
        base_url: str = "http://localhost:8020",
        voice_dir: Path = Path("voices"),
        language: str = "en",
        exaggeration: float = 0.5,
    ) -> None:
        self.work_dir = work_dir
        self._base_url = base_url.rstrip("/")
        self._voice_dir = voice_dir
        self._language = language
        self._exaggeration = exaggeration

    async def health(self) -> HealthStatus:
        try:
            import httpx
        except ImportError:
            return HealthStatus(
                status="down",
                reason="httpx not installed. Run: pip install httpx",
            )
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{self._base_url}/health")
                resp.raise_for_status()
            return HealthStatus(status="ok")
        except Exception as exc:
            return HealthStatus(
                status="down",
                reason=f"TTS service unreachable at {self._base_url}: {exc}",
            )

    async def estimate_cost(self, req: VoiceRequest) -> CostEstimate:
        return CostEstimate(
            amount=0.0,
            notes="Self-hosted TTS service; GPU compute cost via rented hardware.",
        )

    async def run(self, req: VoiceRequest) -> AudioTrack:
        voice_path = self._check_voice(req.voice_profile_ref)

        out_dir = self.work_dir / req.speaker_id
        out_dir.mkdir(parents=True, exist_ok=True)

        exaggeration = self._exaggeration_for(req.expression)

        log_event(
            logger,
            "synthesize_voice_started",
            speaker_id=req.speaker_id,
            voice_ref=req.voice_profile_ref,
            text_chars=len(req.text),
            expression=req.expression,
            exaggeration=exaggeration,
        )

        wav_bytes = await self._call_tts(req.text, voice_path, exaggeration)
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
        """
        Derive the path stem for a voice_profile_ref.
        "voices/pig_kids_placeholder/c1" → "pig_kids_placeholder/c1"
        """
        parts = Path(voice_profile_ref).parts
        if parts and parts[0] == "voices":
            parts = parts[1:]
        return str(Path(*parts))

    def _check_voice(self, ref: str) -> Path:
        """
        Return the reference audio path for this ref.
        Raises RuntimeError with a Track B prompt if the file is absent.
        """
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

    def _exaggeration_for(self, expression: str | None) -> float:
        """Return exaggeration level for this expression; fall back to base value."""
        if expression is None:
            return self._exaggeration
        return _EXPRESSION_EXAGGERATION.get(expression.lower(), self._exaggeration)

    async def _call_tts(
        self, text: str, voice_path: Path, exaggeration: float
    ) -> bytes:
        """POST to the TTS service; return raw WAV bytes."""
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as client:
            with voice_path.open("rb") as ref_file:
                resp = await client.post(
                    f"{self._base_url}/synthesize",
                    data={
                        "text": text,
                        "language": self._language,
                        "exaggeration": str(exaggeration),
                    },
                    files={"reference_audio": (voice_path.name, ref_file, "audio/wav")},
                )
            resp.raise_for_status()
        return resp.content

    def _save_audio(
        self, wav_bytes: bytes, out_dir: Path, text: str
    ) -> tuple[Path, float]:
        """Write WAV bytes to disk; return (path, duration_sec)."""
        stem = hashlib.sha1(text.encode()).hexdigest()[:12]
        path = out_dir / f"{stem}.wav"
        path.write_bytes(wav_bytes)
        return path, self._wav_duration(path, text)

    def _wav_duration(self, path: Path, text: str = "") -> float:
        """Read duration from WAV header; fall back to word-count estimate."""
        try:
            with wave.open(str(path)) as wf:
                return wf.getnframes() / wf.getframerate()
        except Exception:
            words = len(text.split()) if text else 1
            return max(1.0, words / _FALLBACK_WPS)
