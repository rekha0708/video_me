---
name: feedback-render-adapter
description: ComfyUI cannot do local Flux 2.0 inference — use musubi_flux adapter instead
metadata:
  type: feedback
---

Use `musubi_flux` (not `comfyui_flux`) as the render_character adapter for Flux 2.0.

**Why:** ComfyUI's `Flux2*` nodes are cloud BFL API nodes requiring a paid API key. ComfyUI also has no Mistral 3 text encoder loader node — it only ships with CLIP+T5 (Flux 1.x encoders). Flux 2.0 requires Mistral 3.

**How to apply:** `VIDEO_ME_RENDER_ADAPTER=musubi_flux` (now the default in config.py). The adapter calls `/workspace/musubi-tuner/src/musubi_tuner/flux_2_generate_image.py` as an async subprocess with `--fp8 --fp8_scaled --attn_mode flash`. ComfyUI is still used for LTX-2.3 video generation (a different stage).
