---
name: feedback-training
description: Optimizer and precision choices for Flux 2.0 LoRA training on H200
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0f42bdac-7ce1-4470-b65d-7073873d419a
---

Use full `adamw` (not `adamw8bit`) for LoRA training on H200.

**Why:** H200 has 143 GB VRAM — adamw8bit saves only ~6 GB with zero speed benefit. The memory saving only matters when running multiple simultaneous training jobs. User confirmed preference after asking about it.

**How to apply:** In all future musubi-tuner training commands, pass `--optimizer_type adamw`. Only switch to adamw8bit if running two jobs at the same time.

---

Keep `--fp8_base --fp8_scaled` for Flux 2.0 LoRA training.

**Why:** fp8 gives both ~31 GB VRAM savings AND 10-30% faster forward pass via H200 fp8 tensor cores. Not just a memory optimization — also a speed win. LoRA weights still train in bf16 regardless.

**How to apply:** Always include `--fp8_base --fp8_scaled` in musubi-tuner training commands for Flux 2.0.
