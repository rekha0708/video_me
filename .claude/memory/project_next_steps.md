---
name: project-next-steps
description: Immediate next steps after Max LoRA training completes (as of 2026-06-29)
metadata: 
  node_type: memory
  type: project
  originSessionId: 0f42bdac-7ce1-4470-b65d-7073873d419a
---

Max LoRA training completing ~8:10 AM 2026-06-29. After it finishes:

1. Verify `loras/kids_duo_max.safetensors` is ~745 MB (not the old 37 MB stale file)
2. Run `python -m scripts.check_track_b` — both LoRAs + voice refs should be READY
3. Run pipeline end-to-end with source video at `/workspace/downloads/learn_body_parts_with_rosie_fun_kids_act.mp4`
4. Human approval gates at `localhost:8765` — storyboard first, then image grid

**Source video confirmed:** `/workspace/downloads/learn_body_parts_with_rosie_fun_kids_act.mp4` (13 MB, rights_cleared=True per user)

**Services already up:** Ollama ✓, ComfyUI ✓, Fish S2 ✓

**Why:** [[project-video-me]], [[project-lora-training]]
