from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from core.models.profile import Cast, ChannelProfile

ModelT = TypeVar("ModelT", bound=BaseModel)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VIDEO_ME_", env_file=".env", extra="ignore")

    app_name: str = "video_me"
    environment: str = "local"
    data_dir: Path = Path(".local")
    artifact_dir: Path = Path(".local/artifacts")
    sqlite_path: Path = Path(".local/video_me.db")
    job_store: Literal["sqlite", "postgres"] = "sqlite"
    artifact_store: Literal["local", "s3"] = "local"
    postgres_dsn: str = "postgresql://video_me:video_me_dev@localhost:5432/video_me"
    s3_endpoint_url: str = "http://localhost:9000"
    s3_bucket: str = "video-me-artifacts"
    s3_access_key_id: str = "video_me"
    s3_secret_access_key: str = "video_me_dev_password"
    s3_region: str = "us-east-1"
    workflow_engine: str = "asyncio"
    max_regenerations: int = 3
    lora_dir: Path = Path("loras")
    voice_dir: Path = Path("voices")
    review_dir: Path = Path("review")
    local_video_dir: Path = Path("/workspace/downloads")
    llm_model: str = "qwen3.6:35b"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    critique_model: str = "qwen3.6:35b"
    critique_base_url: str = "http://localhost:11434/v1"
    critique_api_key: str = "ollama"
    # --- analyze_visuals: describe source-video settings (same multimodal model) ---
    analyze_visuals_model: str = "qwen3.6:35b"
    analyze_visuals_base_url: str = "http://localhost:11434/v1"
    analyze_visuals_api_key: str = "ollama"
    visual_max_frames: int = 8
    chat_model: str = "qwen3.6:35b"
    chat_base_url: str = "http://localhost:11434/v1"
    chat_api_key: str = "ollama"
    # --- render_character backend ("a1111", "comfyui_flux", or "musubi_flux") ---
    render_adapter: Literal["a1111", "comfyui_flux", "musubi_flux"] = "musubi_flux"
    sd_base_url: str = "http://localhost:7860"       # AUTOMATIC1111 (kept for fallback)
    comfyui_base_url: str = "http://localhost:8188"  # ComfyUI (legacy LTX + comfyui_flux fallback)

    # --- generate_video backend ("wan_s2v", "wan", "wan_lightx2v", or legacy "ltx") ---
    video_adapter: Literal["wan_s2v", "wan", "wan_lightx2v", "ltx"] = "wan_s2v"
    wan_base_url: str = "http://localhost:8030"      # Wan 2.2 deferred-load server (fallback)
    wan_s2v_base_url: str = "http://localhost:8031"  # Wan 2.2 Speech-to-Video server (singing/default)
    wan_lightx2v_base_url: str = "http://localhost:8032"  # LightX2V 4-step Wan I2V (experimental)
    ltx_base_url: str = "http://localhost:8188"      # LTX-Video 2.3 via ComfyUI (legacy)
    # Wan deferred-loading sequence (core/gpu_sequencer.py): gap between unloading
    # the render-phase models and loading Wan, and the readiness-poll ceiling.
    wan_load_gap_sec: int = 30
    wan_load_timeout_sec: int = 1800

    # --- synthesize_voice backend ("chatterbox" or "fish_s2") ---
    tts_adapter: Literal["chatterbox", "fish_s2"] = "fish_s2"
    tts_base_url: str = "http://localhost:8020"       # Chatterbox (fallback)
    fish_s2_base_url: str = "http://localhost:8025"   # Fish Audio S2 (default)
    fish_s2_load_gap_sec: int = 5
    # ONE budget for the entire eager-load-to-ready wait, used by both
    # ensure_fish_s2_process_running (waiting for the process to accept
    # connections at all) and _prepare_managed_adapter's wait_until_loaded
    # (waiting for model_loaded=true) — deliberately the same setting, not two
    # independently-guessable ones. fish_s2_server.py's lifespan() loads the
    # model EAGERLY at startup and blocks the whole ASGI app on it — /health
    # can't respond at all, let alone report model_loaded=true, until the load
    # finishes — so "process reachable" and "model loaded" are literally the
    # same real-world event here, not two separate waits (confirmed in the
    # code and in log ordering: "Application startup complete" only ever
    # prints after "Models warmed up"). A previous version of this setting
    # split these into two values (240s / 30s) picked without checking that;
    # the second one caused two production job failures in one session.
    # Real data — 5 successful loads timestamped this session (2026-07-10):
    # 63.8s, 63.8s, 94.3s, 122.6s, 80.1s -> min 63.8s, max 122.6s, mean 88.8s.
    # 600s is a stated ~5x margin over the verified max, not a fresh guess —
    # errs toward waiting (a slow load that succeeds costs nothing) rather
    # than a tight bound that risks another false-timeout job failure.
    fish_s2_load_timeout_sec: int = 600
    # Fish S2's own CUDA allocator retains memory across synthesis calls that
    # POST /unload + torch.cuda.empty_cache() cannot reclaim (observed ~63GB
    # resident vs ~20GB fresh-process baseline after one job's worth of TTS
    # calls). The worker kills the whole process after every job and respawns
    # it fresh only when the next job actually needs voice synthesis — these
    # settings mirror the exact command start_services.sh uses to launch it.
    fish_s2_venv_python: str = "/workspace/.venv_fish_s2/bin/uvicorn"
    fish_s2_speech_dir: str = "/workspace/fish-speech"
    fish_s2_log_path: str = "/workspace/logs/fish_s2.log"

    # --- language selection ---
    target_language: str = "en"  # "en" | "hi" | "both"

    # --- lip-sync repair backend for non-native video adapters ("latentsync", "musetalk", or "none") ---
    lipsync_adapter: Literal["latentsync", "musetalk", "none"] = "latentsync"
    lipsync_base_url: str = "http://localhost:8040"   # legacy alias for MuseTalk
    musetalk_base_url: str = "http://localhost:8040"
    latentsync_base_url: str = "http://localhost:8041"
    latentsync_inference_steps: int = 20
    latentsync_guidance_scale: float = 1.5
    whisper_model_size: str = "large-v3"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    whisper_download_root: str = ""  # faster-whisper cache dir; setup_gpu writes /workspace/.cache/huggingface/hub
    whisper_local_files_only: bool = True
    whisper_model_revision: str = ""  # optional HF commit/tag to pin the cached faster-whisper snapshot
    whisper_language: str = "en"  # "en"/"hi" force source language; "auto" lets Whisper guess
    whisper_vad_filter: bool = False  # keep sung vocals/lyrics; VAD can be over-aggressive on music
    whisper_isolate_vocals: bool = False  # Demucs vocal separation before Whisper; only applied when audio_profile="singing"
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    render_allow_placeholder_lora: bool = False

    # --- plan critique loop ---
    max_plan_iterations: int = 3          # max LLM critique re-plans before failing
    auto_approve_plan: bool = False       # set True in CI / smoke tests to skip approval UI
    auto_approve_transcript: bool = False  # skip the story/transcribe review gate (dashboard jobs)

    # --- shot duration ---
    max_shot_duration_sec: float = 8.0   # ceiling per shot (words/2, clamped to [5, this])
    transcript_min_coverage_ratio: float = 0.2  # fail only catastrophic short transcripts on real media

    # --- AV sync / lip-sync policy ---
    lipsync_failure_policy: Literal["fallback_raw", "fail"] = "fallback_raw"
    lipsync_max_retries: int = 0
    av_sync_duration_tolerance_sec: float = 0.35
    av_sync_failure_policy: Literal["warn", "fail"] = "warn"
    wan_s2v_fps: int = 16

    # --- human approval web UI (storyboard) ---
    approval_port: int = 8765
    approval_timeout_hours: float = 24.0

    # --- image candidate generation + VLM critique ---
    image_candidates: int = 1            # images generated per shot for critique
    image_critique_model: str = "qwen3.6:35b"
    image_critique_base_url: str = "http://localhost:11434/v1"
    image_critique_api_key: str = "ollama"
    feedback_log_dir: Path = Path("assets/kids_duo")  # resolved per-cast in load_app_config()

    # --- config paths (overridable via env) ---
    cast_path: Path = Path("config/casts/kids_duo.yaml")
    channel_path: Path = Path("config/channels/education_kids.yaml")

    # --- human approval web UI (image grid) ---
    # Reuses approval_port — the two gates run sequentially so no conflict.
    auto_approve_images: bool = False    # set True in CI / smoke tests

    @field_validator("whisper_language", mode="before")
    @classmethod
    def _default_blank_whisper_language_to_english(cls, value):
        if value is None:
            return "en"
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or "en"
        return value


class AppConfig(BaseModel):
    settings: Settings = Field(default_factory=Settings)
    channel_profile: ChannelProfile
    cast: Cast


def load_yaml_model(path: Path, model: type[ModelT]) -> ModelT:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return model.model_validate(payload)


def load_app_config(
    channel_path: Path | None = None,
    cast_path: Path | None = None,
) -> AppConfig:
    settings = Settings()
    resolved_channel = channel_path or settings.channel_path
    resolved_cast = cast_path or settings.cast_path
    cast = load_yaml_model(resolved_cast, Cast)
    if settings.feedback_log_dir == Path("assets/kids_duo"):
        settings.feedback_log_dir = Path(f"assets/{cast.id}")
    return AppConfig(
        settings=settings,
        channel_profile=load_yaml_model(resolved_channel, ChannelProfile),
        cast=cast,
    )
