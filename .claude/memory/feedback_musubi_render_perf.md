---
name: feedback-musubi-render-perf
description: Why render_character takes ~4min/candidate and shows 0% GPU util with climbing VRAM
metadata: 
  node_type: memory
  type: feedback
  originSessionId: dea3ed26-190c-4d16-812a-c23f9f1ff2c9
---

`adapters/render_character/musubi_flux_adapter.py` spawns a **brand-new subprocess per candidate
image** (`_generate()`, called once per `num_images` in a loop). Each subprocess reloads from disk:
Flux 2.0 DiT (`flux2-dev.safetensors`, 64.4 GB bf16) + Mistral 3 text encoder shard(s), then
`--fp8_scaled` walks the entire state dict tensor-by-tensor in Python
(`musubi-tuner/.../fp8_optimization_utils.py:optimize_state_dict_with_fp8`) to quantize to fp8.

**Why this looks "stuck" (0% GPU-util, VRAM slowly climbing, ~57% CPU):** the ~4min/candidate cost
is disk I/O + Python-level per-tensor marshalling, not diffusion compute — the actual 20-step
1024×1024 generation is fast on an H200. `nvidia-smi --query-gpu=utilization.gpu,...` sampled every
2s during a real run showed 0% util almost the entire time, with VRAM climbing steadily as tensors
streamed in. This is expected/correct behavior, not a hang — confirmed by watching it complete.

**No cheap one-flag fix exists.** The real gap is no model residency: nothing caches the
loaded+quantized weights across calls, unlike Ollama/ComfyUI which stay resident between requests.
Two independent levers if this becomes a priority to fix:
1. musubi-tuner's `--save_merged_model` to pre-bake an fp8 checkpoint once (~32GB instead of
   re-quantizing 64GB bf16 every call) — cuts load time, still per-subprocess.
2. Make the adapter keep a resident process/server (like ComfyUI/Ollama) instead of spawning one
   per image — the real fix, but a bigger architecture change vs. the current intentional
   subprocess design.

**Why:** [[project-next-steps]] — found while investigating why render_character was slow during
job `20260704-020418-33o` on 2026-07-04, before that job later failed with an unrelated VRAM OOM
(see [[project-wan22]]).
