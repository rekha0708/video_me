# ComfyUI Stack Upgrade — Flux 2.0 Dev + LTX-2.3 22B

**Date:** 2026-06-28  
**Summary:** Upgraded to latest and most capable models — Flux 2.0 Dev (32B) for image generation and LTX-2.3 22B distilled v1.1 for video generation with native lip-sync.

---

## ✅ What Changed

### **1. Flux 1.x → Flux 2.0 Dev (32B params)**

**Previous:** Flux.1-dev (12B params)  
**New:** Flux 2.0 Dev (32B params, released Nov 2025)

**Improvements:**
- 32B parameters (vs 12B) — significantly more capable
- Better photorealism and lighting
- Improved hand and fabric detail
- Better multi-image consistency
- Professional-class text rendering

**Model files:**
```
models/diffusion_models/flux.2-dev.safetensors  (~58 GB)
models/text_encoders/clip_l.safetensors         (~1 GB)
models/text_encoders/t5xxl_fp8_e4m3fn.safetensors (~4.5 GB)
models/vae/ae.safetensors                       (~335 MB)
Total: ~64 GB
```

**VRAM:** ~20 GB (up from ~12 GB with Flux 1.x)

---

### **2. LTX-Video v0.9.5 → LTX-2.3 22B distilled v1.1**

**Previous:** LTX-Video 2B v0.9.5 (6.34 GB, old model from 2025)  
**New:** LTX-2.3 22B distilled v1.1 (42 GB, released March 2026)

**Improvements:**
- 22B parameters (vs 2B) — 11× model size
- 8-step distilled — fast inference (~1 min/shot)
- Native audio-video sync in single diffusion pass
- Better prompt following and detail generation
- Improved aesthetics and audio quality (v1.1)
- Supports up to 20s video @ 24 FPS

**Model file:**
```
models/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors (~42 GB)
```

**VRAM:** ~44 GB (vs ~12 GB with old 2B model)

---

### **3. LTX Custom Nodes Added**

**New:** ComfyUI-LTXVideo custom nodes installed automatically

Required for LTX-2.3 support — provides:
- `LTXVModelLoader`
- `LTXVSampler`
- `LTXVScheduler`
- `LTXVConditioning`
- Audio-video sync nodes

---

### **4. Workflow JSON Templates Created**

**New files added:**
- `assets/comfyui_workflows/flux_lora_txt2img.json` — Flux 2.0 + LoRA txt2img
- `assets/comfyui_workflows/ltx_i2v.json` — LTX-2.3 image-to-video with audio

Adapters now load real workflow templates instead of using hard-coded fallbacks.

---

## 📊 VRAM Budget Impact

| Component | Old | New | Δ |
|-----------|-----|-----|---|
| qwen3.6:35b (LLM+VLM) | 30 GB | 30 GB | — |
| Flux model | 12 GB | 20 GB | +8 GB |
| LTX model | 12 GB | 44 GB | +32 GB |
| Fish S2 TTS | 20 GB | 20 GB | — |
| **Peak total** | **~74 GB** | **~114 GB** | **+40 GB** |

**Target GPU:** NVIDIA G200 (143 GB) — **29 GB headroom** ✅

---

## 🔧 Files Updated

### **Setup & Services:**
- `scripts/setup_gpu.sh` — downloads Flux 2.0 + LTX-2.3 + custom nodes
- `scripts/start_services.sh` — updated descriptions

### **Documentation:**
- `CLAUDE.md` — updated model references throughout
- `README.md` — updated all Flux/LTX mentions
- `assets/comfyui_workflows/README.md` — updated workflow descriptions

### **Workflow templates (NEW):**
- `assets/comfyui_workflows/flux_lora_txt2img.json`
- `assets/comfyui_workflows/ltx_i2v.json`

---

## 🚀 Next Steps

### **1. Retrain LoRAs for Flux 2.0**

Existing SD 1.5 and Flux 1.x LoRAs are **incompatible** with Flux 2.0.

```bash
# Update kohya_ss config to target flux.2-dev.safetensors
# Then retrain:
accelerate launch flux_train_network.py \
  --config_file assets/kids_duo/training/kohya_config.toml
```

Output: `loras/kids_duo_max.safetensors` (Flux 2.0 format)

### **2. Install ComfyUI + Models**

```bash
# Full setup (requires HF_TOKEN for Flux 2.0)
export HF_TOKEN=hf_...
bash scripts/setup_gpu.sh

# Total downloads: ~106 GB (Flux 64GB + LTX 42GB)
# Time estimate: ~2-4 hours on fast connection
```

### **3. Start Services**

```bash
bash scripts/start_services.sh
python -m scripts.check_runtime_readiness
```

---

## 📝 Notes

- **Flux 2.0 is gated** — requires HuggingFace account + license acceptance at https://huggingface.co/black-forest-labs/FLUX.2-dev
- **LTX-2.3 is Apache 2.0** — free for commercial use under $10M revenue
- **Workflow templates** are minimal examples — customize in ComfyUI UI and re-export as needed
- **Fallback adapters** (A1111, Wan, MuseTalk) still available via env vars

---

## 🔗 References

- Flux 2.0 announcement: https://bfl.ai/blog/flux-2
- LTX-2.3 HuggingFace: https://huggingface.co/Lightricks/LTX-2.3
- ComfyUI LTX nodes: https://github.com/Lightricks/ComfyUI-LTXVideo
