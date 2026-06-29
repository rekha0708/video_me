---
name: project-video-me
description: "Core context for the video_me kids' educational video pipeline — stack, services, LoRA training, and asset locations"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0f42bdac-7ce1-4470-b65d-7073873d419a
---

Kids' educational video pipeline that transforms a source video into an animated short starring Max and Zoe.

**Stack:**
- LLM/VLM: qwen3.6:35b via Ollama (port 11434)
- Image gen: **musubi-tuner `flux_2_generate_image.py`** (direct subprocess, NOT ComfyUI — ComfyUI lacks Mistral 3 loader node)
- Video gen: LTX-2.3 22B via ComfyUI (port 8188, native lip-sync)
- TTS: Fish Audio S2 Pro (port 8025, EN + HI + 80 languages)
- Fallback video: Wan 2.2 I2V (port 8030)
- LoRA training: musubi-tuner (kohya-ss/musubi-tuner), Flux 2.0 native

**Key paths:**
- Pipeline: `/workspace/video_me/`
- ComfyUI: `/workspace/ComfyUI/`
- Flux 2.0 model: `/workspace/ComfyUI/models/diffusion_models/flux2-dev.safetensors`
- VAE: `/workspace/ComfyUI/models/diffusion_models/ae.safetensors`
- Mistral 3 text encoder: `/workspace/FLUX2-text-encoder/`
- musubi-tuner: `/workspace/musubi-tuner/`
- LoRAs: `/workspace/video_me/loras/`
- Voice refs: `/workspace/video_me/voices/kids_duo/max.wav` + `zoe.wav`
- Wan2.2 repo: `/workspace/Wan2.2/Wan2.2/` (nested — inner dir has the `wan/` package)
- Wan venv: `/workspace/.venv_wan/`
- Fish S2 venv: `/workspace/.venv_fish_s2/`
- Downloads: `/workspace/downloads/`
- Logs: `/workspace/logs/`

**Services start command:** `bash scripts/start_services.sh` (run after every pod restart; auto-reinstalls Ollama which is wiped on restart)

**GitHub:** https://github.com/rekha0708/video_me

**Why:** RunPod H200 (143 GB VRAM). Pod restarts wipe base Linux; `/workspace/` persists on network volume.
