from pathlib import Path
from typing import Literal, TypeVar

import yaml
from pydantic import BaseModel, Field
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
    llm_model: str = "qwen3.6:35b"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    critique_model: str = "llava:7b"
    critique_base_url: str = "http://localhost:11434/v1"
    critique_api_key: str = "ollama"
    # --- render_character backend ("a1111" or "comfyui_flux") ---
    render_adapter: Literal["a1111", "comfyui_flux"] = "comfyui_flux"
    sd_base_url: str = "http://localhost:7860"       # AUTOMATIC1111 (kept for fallback)
    comfyui_base_url: str = "http://localhost:8188"  # ComfyUI (Flux + LTX)

    # --- generate_video backend ("wan" or "ltx") ---
    video_adapter: Literal["wan", "ltx"] = "ltx"
    wan_base_url: str = "http://localhost:8030"      # Wan 2.2 resident server (kept for fallback)
    ltx_base_url: str = "http://localhost:8188"      # LTX-Video 2.3 via ComfyUI (default same host)

    tts_base_url: str = "http://localhost:8020"
    lipsync_base_url: str = "http://localhost:8040"
    whisper_model_size: str = "medium"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "int8"
    ffmpeg_bin: str = "ffmpeg"
    ffprobe_bin: str = "ffprobe"
    render_allow_placeholder_lora: bool = False

    # --- plan critique loop ---
    max_plan_iterations: int = 3          # max LLM critique re-plans before failing
    auto_approve_plan: bool = False       # set True in CI / smoke tests to skip approval UI

    # --- human approval web UI ---
    approval_port: int = 8765
    approval_timeout_hours: float = 24.0


class AppConfig(BaseModel):
    settings: Settings = Field(default_factory=Settings)
    channel_profile: ChannelProfile
    cast: Cast


def load_yaml_model(path: Path, model: type[ModelT]) -> ModelT:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return model.model_validate(payload)


def load_app_config(
    channel_path: Path = Path("config/channels/education_kids.yaml"),
    cast_path: Path = Path("config/casts/kids_duo.yaml"),
) -> AppConfig:
    return AppConfig(
        channel_profile=load_yaml_model(channel_path, ChannelProfile),
        cast=load_yaml_model(cast_path, Cast),
    )
