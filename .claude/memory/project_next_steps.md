---
name: project-next-steps
description: Pipeline end-to-end run status and render adapter fix (as of 2026-06-29)
metadata:
  type: project
---

Pipeline first run completed LLM stages successfully. Job ID: `20260629-074235-za3`.

**LLM stages all ✅:**
- fetch_media (1s), transcribe (40s), analyze_content (84s), adapt_script (45s)
- plan_shots: 14 shots, critique passed first attempt (all scores ≥0.80, kids_safety 1.0)
- Human storyboard approval: approved

**ComfyUI render failure + fix:**
ComfyUI `Flux2*` nodes are cloud API nodes (require BFL API key), not local inference.
ComfyUI also lacks a Mistral 3 text encoder loader — it only has CLIP+T5 (Flux 1.x).

Fix: wrote `adapters/render_character/musubi_flux_adapter.py` — calls musubi-tuner's
`flux_2_generate_image.py` directly as a subprocess. Default adapter changed to `musubi_flux`.

**Current state:** pipeline resumed at render_character (shot 1/14), using musubi_flux adapter.

**Run command:**
```bash
# New run:
VIDEO_ME_RENDER_ADAPTER=musubi_flux .venv/bin/python run_pipeline.py <video> --rights-cleared
# Resume:
.venv/bin/python run_pipeline.py --resume-job <JOB_ID> --rights-cleared
```

**Human approval gates:** storyboard at localhost:8765 (done), image grid (coming after renders)

**Why:** [[project-video-me]], [[project-lora-training]]
