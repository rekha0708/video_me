# Pipeline stages — model / service / VRAM

End-to-end reference for the **default stack** (musubi Flux image + ComfyUI/LTX
video + Fish S2 TTS), on the G200 (143 GB). Kept in sync with `core/workflow.py`
and `core/config.py`. Last updated 2026-07-05.

One model does all reasoning **and** vision: **qwen3.6:35b** (~30 GB, natively
multimodal, resident in Ollama). Every LLM/VLM stage below shares that one
resident model — the 30 GB is **not** additive across stages.

There are no per-stage "agents" — it's a single async pipeline process
(`run_pipeline_job`). The `.claude/agents/*` are operator tools, not runtime.

## Stage table

| # | Stage | Model / service | ~VRAM | Notes |
|---|-------|-----------------|-------|-------|
| 1 | fetch_media | yt-dlp + ffmpeg (CPU) | 0 | download + extract audio; **skipped** for story input |
| 2 | transcribe | faster-whisper | ~1–2 GB | audio → timed transcript; **seeded** (not run) for story input |
| 3 | analyze_content | qwen3.6:35b | 30 (shared) | transcript → topic + learning objective |
| 4 | **analyze_visuals** | qwen3.6:35b (multimodal) | 30 (shared) | ≤8 source frames → per-segment settings/props; best-effort, empty for story/remote |
| 5 | check_rights | — | 0 | gate; BLOCKS if `rights_cleared=False` |
| 6 | adapt_script | qwen3.6:35b | 30 (shared) | original script; **scene settings grounded in #4** |
| 7 | plan_shots | qwen3.6:35b | 30 (shared) | one shot per line + camera/action/duration (5–8 s) |
| 8 | critique_plan (≤3×) | qwen3.6:35b | 30 (shared) | score 5 dims; re-plan if any < 0.75 |
| — | **plan approval gate** | web UI :8765/dashboard | 0 | human approves storyboard |
| 9 | render_character ×N (per shot) | **Flux 2.0 Dev** via musubi (subprocess) | ~20 GB | image per shot from `visual_descriptor + setting + camera` + LoRA; freed after each image; Ollama unloaded first. LoRA/weight/steps/trigger from `config/casts/<cast>/params.py`. **Skipped** in story_images mode. |
| 10 | critique_images | qwen3.6:35b | 30 | pick best candidate; self-learning feedback log |
| — | **image approval gate** | web UI | 0 | human confirms/overrides per shot |
| — | *gpu_sequencer* | — | — | wan path only: unload Ollama → 30 s → POST /load → poll |
| 11 | synthesize_voice (per shot) | **Fish Audio S2** | ~20 GB | TTS EN/HI; voice ref from cast params/YAML |
| 12 | generate_video (per shot) | **LTX-2.3 22B** via ComfyUI *(default)* | ~44 GB | i2v from image + `setting` prompt + audio; **native lip-sync** |
| 12b | generate_video *(fallback)* | **Wan 2.2** :8030 | ~52 GB | silent i2v; deferred-loaded; then MuseTalk lip_sync (subprocess, broken on cartoon faces) |
| 13 | assemble_video | ffmpeg (CPU) | 0 | concat, scale 1080×1920, captions, AI-disclosure label |
| 14 | critique (video, Phase 2) | qwen3.6:35b | 30 | rubric on sampled output frames → pass/regenerate/reject |
| 15 | publish | file copy (CPU) | 0 | → `review/<ts>_<lang>_<stem>/` + metadata.json |

## VRAM peak

Default stack, worst simultaneous residency:
`qwen(30) + LTX(44) + Flux(20) + Fish(20) ≈ 114 / 143 GB` (~29 GB headroom).

The **gpu_sequencer** guarantees **Wan(52) and Flux(20) never coexist** — loading
both was the render_character OOM (commit 58ce9d8) that motivated Wan deferred
loading. Wan is unloaded during render and loaded only after image approval.

## Service ports

| Service | Port | Required (default)? |
|---------|------|---------------------|
| Ollama (qwen3.6:35b) | 11434 | ✅ always |
| ComfyUI (LTX video) | 8188 | ✅ default |
| Fish Audio S2 (TTS) | 8025 | ✅ default |
| musubi-tuner (Flux) | — (subprocess) | ✅ default |
| Wan 2.2 | 8030 | ⚠️ `VIDEO_ME_VIDEO_ADAPTER=wan` |
| MuseTalk | 8040 | ⚠️ wan path only |
| Chatterbox TTS | 8020 | ⚠️ `TTS_ADAPTER=chatterbox` |
| AUTOMATIC1111 | 7860 | ⚠️ `RENDER_ADAPTER=a1111` |

Start everything + health-check: `bash scripts/start_services.sh`.

## Known failure modes (see also the "where it breaks" notes)

- **Track B not done** → render_character raises before any GPU call (LoRA
  missing/placeholder). Current #1 blocker for a real run.
- **analyze_visuals silently empty** if the Ollama model isn't serving vision or
  ffmpeg is missing → falls back to invented settings (no crash). Verify the
  `analyze_visuals.json` artifact / "Source Video Settings" card on the first run.
- **Per-shot render cost**: N_shots × N_candidates Flux images — lower
  `image_candidates` for real runs.
- **Multi-character shot**: only the speaker (`characters_on_screen[0]`) is
  rendered; a second character does not appear. Not handled today.
- **Wan `/unload` vs in-flight inference**: unload waits out inference; the
  adapter's 120 s timeout raises rather than OOM the next render.
