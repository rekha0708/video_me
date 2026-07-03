---
name: feedback-render-adapter
description: ComfyUI cannot do local Flux 2.0 inference — use musubi_flux adapter instead
metadata:
  type: feedback
---

Use `musubi_flux` (not `comfyui_flux`) as the render_character adapter for Flux 2.0.

**Why:** ComfyUI's `Flux2*` nodes are cloud BFL API nodes requiring a paid API key. ComfyUI also has no Mistral 3 text encoder loader node — it only ships with CLIP+T5 (Flux 1.x encoders). Flux 2.0 requires Mistral 3. Confirmed again on 2026-07-03: `.env` had drifted to `VIDEO_ME_RENDER_ADAPTER=comfyui_flux`, and the ComfyUI Flux workflow (`assets/comfyui_workflows/flux_lora_txt2img.json`) is a leftover Flux-1.x template (DualCLIPLoader + T5/CLIP) with only the UNET filename swapped to a Flux 2.0 checkpoint — fundamentally incompatible, plus the CLIP/T5 files it references don't even exist in `models/clip/`.

**How to apply:** Check `.env`'s `VIDEO_ME_RENDER_ADAPTER` is `musubi_flux` — it can silently drift back to `comfyui_flux` (config.py's own default is `musubi_flux`, but `.env` overrides it). The adapter (`adapters/render_character/musubi_flux_adapter.py`) calls `/workspace/musubi-tuner/src/musubi_tuner/flux_2_generate_image.py` as an async subprocess with `--fp8 --fp8_scaled --attn_mode flash`. ComfyUI is still used for LTX-2.3 video generation (a different stage).

**musubi-tuner needs its own dedicated venv** — `/workspace/.venv_musubi` (`python3 -m venv --system-site-packages`, inherits torch 2.8.0+cu128, then `pip install -e /workspace/musubi-tuner` + `pip install flash-attn --no-build-isolation`). The adapter invokes `_MUSUBI_PYTHON` (that venv's python), NOT `sys.executable` — using `sys.executable` would resolve to the orchestration `.venv`, which per [[project_video_me]] must stay lightweight and never gets musubi_tuner installed. `setup_gpu.sh`'s `setup_musubi_tuner()` and `start_services.sh` both set up / self-heal this venv now (fixed 2026-07-03; the old version of `setup_gpu.sh` installed into system pip3, which vanished on a pod restart — see [[feedback_service_startup_gaps]]).

**`--text_encoder` must point at the first shard file**, not the directory: `/workspace/FLUX2-text-encoder/text_encoder/model-00001-of-00010.safetensors`. musubi-tuner's `load_split_weights()` derives sibling shard filenames from this one by pattern-matching `NNNNN-of-NNNNN.safetensors` — passing the bare directory raises `IsADirectoryError`. The tokenizer is pulled separately from HF Hub (`mistralai/Mistral-Small-3.1-24B-Instruct-2503`), not from the local `tokenizer/` subdirectory.
