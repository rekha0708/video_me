---
name: project-lora-training
description: "LoRA training setup for Max and Zoe characters — framework, config, VRAM usage, lessons learned"
metadata: 
  node_type: memory
  type: project
  originSessionId: 0f42bdac-7ce1-4470-b65d-7073873d419a
---

Max and Zoe Flux 2.0 LoRAs trained with musubi-tuner. Both trained to 1200 steps / 4 epochs.

**Trained LoRAs (as of 2026-06-29):**
- `loras/kids_duo_zoe.safetensors` — 745 MB ✅ complete
- `loras/kids_duo_max.safetensors` — training finishing ~8:10 AM (step ~1130/1200)

**Training config:**
- Framework: musubi-tuner `flux_2_train_network.py` (NOT sd-scripts — incompatible with Flux 2.0)
- Steps: 1200 total (300 steps/epoch × 4 epochs)
- Batch size: 4; 1200 images/epoch ÷ 4 = 300 steps
- Network: `networks.lora_flux_2`, dim=32, alpha=16
- Base precision: fp8 (`--fp8_base --fp8_scaled`) — saves ~31 GB VRAM AND speeds up forward pass
- Mixed precision: bf16
- Optimizer: adamw8bit (current runs) → **use `adamw` for future runs** — H200 has plenty of VRAM, no benefit from 8bit
- Flash attention enabled

**VRAM breakdown at training (~83 GB / 143 GB):**
- Flux 2.0 DiT in fp8: ~30 GB
- Data loaders: ~20 GB
- Activations (gradient checkpointing): ~15 GB
- AdamW states: ~2 GB
- Rest: CUDA overhead

**Critical lessons:**
- NEVER run Max and Zoe training simultaneously — causes CUDA OOM on H200
- `wait PID` doesn't work cross-shell; use `pgrep` polling instead
- Epoch 2 checkpoint saved at step 600 (~745 MB) — good size indicator
- Final output overwrites `kids_duo_max.safetensors`; old 37 MB file = invalid prior run

**Dataset configs:** `assets/kids_duo/training/musubi_dataset_max.toml` / `_zoe.toml`
**Cache (gitignored):** `assets/kids_duo/training/cache/`
**Logs:** `/workspace/logs/lora_max.log`, `/workspace/logs/lora_zoe.log`

**Why:** [[project-video-me]]
