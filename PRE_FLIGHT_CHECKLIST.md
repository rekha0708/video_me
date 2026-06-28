# Pre-Flight Checklist — GPU Installation

**Date:** 2026-06-28  
**Target:** GPU server deployment of video_me pipeline  
**Stack:** Flux 2.0 Dev (32B) + LTX-2.3 22B distilled v1.1 + ComfyUI

---

## ✅ 1. Installation Steps Verification

### **Step-by-Step Installation Flow**

```bash
# ── On your local machine ──────────────────────────────────────────────────
# 1. Verify .env has your HF token
cat .env | grep HF_TOKEN
# Should show: HF_TOKEN=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

# 2. Push to git (already done ✅)
git push origin master

# ── On the GPU machine ────────────────────────────────────────────────────
# 3. Clone/pull the repo
cd /workspace
git clone https://github.com/rekha0708/video_me.git  # or git pull if exists
cd video_me

# 4. Load environment variables
source .env
echo $HF_TOKEN  # Verify token is loaded

# 5. Run full setup (installs everything, downloads ~106 GB)
bash scripts/setup_gpu.sh
# Estimated time: 2-4 hours on fast connection

# 6. Start all services
bash scripts/start_services.sh

# 7. Verify services are healthy
python -m scripts.check_runtime_readiness

# 8. Verify Track B assets (LoRAs + voices)
python -m scripts.check_track_b

# 9. Run test suite
python -m pytest -q

# 10. Test first pipeline run (with a short video)
python -c "
import asyncio
from core.config import load_app_config
from core.workflow import run_with_critique
config = load_app_config()
job = asyncio.run(run_with_critique(
    source_url='YOUR_TEST_URL',
    rights_cleared=True,
    app_config=config
))
print(f'Job status: {job.status}')
"
```

---

## ✅ 2. Repository Versions — Latest & Correct

### **Model Versions (VERIFIED ✅)**

| Component | Version | Release Date | Source | Status |
|-----------|---------|--------------|--------|--------|
| **Flux 2.0 Dev** | 32B params | Nov 25, 2025 | `black-forest-labs/FLUX.2-dev` | ✅ Latest |
| **LTX-2.3 distilled** | 22B v1.1 | ~Mar 2026 | `Lightricks/LTX-2.3` | ✅ Latest |
| **qwen3.6:35b** | MoE 35B | 2025 | Ollama | ✅ Latest |
| **Fish Audio S2** | Latest | 2025-2026 | `fishaudio/fish-speech` | ✅ Latest |

### **Repository URLs (VERIFIED ✅)**

```bash
# ComfyUI
https://github.com/comfyanonymous/ComfyUI.git
# Clone command in setup_gpu.sh: ✅ CORRECT (line 278)

# LTX-Video custom nodes
https://github.com/Lightricks/ComfyUI-LTXVideo.git
# Clone command in setup_gpu.sh: ✅ CORRECT (line 303)

# Fish Audio S2
https://github.com/fishaudio/fish-speech.git
# Clone command in setup_gpu.sh: ✅ CORRECT (line 455)

# ComfyUI Manager (optional)
https://github.com/ltdrdata/ComfyUI-Manager.git
# Clone command in setup_gpu.sh: ✅ CORRECT (line 295)
```

### **Model Download Commands (VERIFIED ✅)**

```bash
# Flux 2.0 Dev (requires HF_TOKEN)
huggingface-cli download black-forest-labs/FLUX.2-dev \
  flux.2-dev.safetensors --token $HF_TOKEN
# setup_gpu.sh line 325-327: ✅ CORRECT

# LTX-2.3 22B distilled v1.1
huggingface-cli download Lightricks/LTX-2.3 \
  ltx-2.3-22b-distilled-1.1.safetensors
# setup_gpu.sh line 380-382: ✅ CORRECT

# T5 XXL FP8 text encoder
curl -fL "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors"
# setup_gpu.sh line 345-346: ✅ CORRECT

# CLIP-L text encoder
curl -fL "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"
# setup_gpu.sh line 356-357: ✅ CORRECT

# Flux VAE
curl -fL "https://huggingface.co/black-forest-labs/FLUX.1-schnell/resolve/main/ae.safetensors"
# setup_gpu.sh line 368-369: ✅ CORRECT
```

---

## ✅ 3. Service Health Checks

### **Required Services (Default Stack)**

| Service | Port | Health Endpoint | Expected in setup_gpu.sh | Status |
|---------|------|-----------------|--------------------------|--------|
| **Ollama** | 11434 | `/api/tags` | ✅ Line 237-269 | ✅ |
| **ComfyUI** | 8188 | `/system_stats` | ✅ Line 272-399 | ✅ |
| **Fish S2** | 8025 | `/health` | ✅ Line 447-474 | ✅ |

### **Fallback Services (Opt-in)**

| Service | Port | Install Flag | Expected in setup_gpu.sh | Status |
|---------|------|--------------|--------------------------|--------|
| **A1111** | 7860 | `--with-a1111` | ✅ Line 402-444 | ✅ |
| **Chatterbox** | 8020 | `--with-chatterbox` | ✅ Line 477-481 | ✅ |
| **Wan 2.2** | 8030 | `--with-wan` | ✅ Line 484-520 | ✅ |
| **MuseTalk** | 8040 | `--with-wan` | ✅ Line 523-617 | ✅ |

### **Service Startup (start_services.sh)**

✅ All services properly configured:
- **Ollama reinstall check** (line 45-62) — handles pod restart
- **ComfyUI startup** (line 64-76)
- **Fish S2 startup** (line 78-93)
- **Fallback services** (line 95-161) — conditional on directory existence
- **Health verification** (line 164-182)

---

## ✅ 4. Test Coverage for New Models

### **Adapters Using New Models**

| Adapter | Model | Test File | Test Coverage | Status |
|---------|-------|-----------|---------------|--------|
| `ComfyUIFluxAdapter` | Flux 2.0 Dev | `tests/test_render_character.py` | Mocked HTTP calls | ✅ |
| `LtxAdapter` | LTX-2.3 22B | `tests/test_generate_video.py` | Mocked HTTP calls | ✅ |
| `LlmAnalyzeAdapter` | qwen3.6:35b | `tests/test_analyze_content.py` | Mocked LLM | ✅ |
| `VlmImageCritiqueAdapter` | qwen3.6:35b | `tests/test_critique.py` | Mocked VLM | ✅ |

### **Workflow Template Tests**

```bash
# Verify workflow JSON exists
ls -lh assets/comfyui_workflows/
# Expected:
# flux_lora_txt2img.json  ✅
# ltx_i2v.json            ✅
```

### **Critical Test Verification**

Run before GPU deployment:

```bash
# 1. Full test suite (313 tests, all should pass)
python -m pytest -q

# 2. Specific adapter tests
python -m pytest tests/test_render_character.py -v
python -m pytest tests/test_generate_video.py -v
python -m pytest tests/test_workflow.py -v

# 3. Runtime readiness (code-only, no services)
python -m scripts.check_runtime_readiness --code-test --skip-services
```

---

## ✅ 5. Download Size & Time Estimates

### **Total Downloads**

| Component | Size | Download via | Requires HF_TOKEN |
|-----------|------|--------------|-------------------|
| **Flux 2.0 UNET** | ~58 GB | `huggingface-cli` | ✅ YES |
| **T5 XXL FP8** | ~4.5 GB | `curl` | ❌ No |
| **CLIP-L** | ~1 GB | `curl` | ❌ No |
| **Flux VAE** | ~335 MB | `curl` | ❌ No |
| **LTX-2.3 22B** | ~42 GB | `huggingface-cli` | ⚠️ Recommended |
| **qwen3.6:35b** | ~20 GB | `ollama pull` | ❌ No |
| **ComfyUI + deps** | ~500 MB | `git clone` + `pip` | ❌ No |
| **Fish S2 + deps** | ~300 MB | `git clone` + `pip` | ❌ No |
| **TOTAL** | **~127 GB** | Mixed | Flux requires token |

### **Time Estimates (1 Gbps connection)**

- **Download time:** ~20-30 minutes (full speed)
- **Realistic time:** 2-4 hours (including pip installs, git clones, Ollama model pull)
- **First run:** Add 5-10 minutes for ComfyUI to index custom nodes

---

## ✅ 6. VRAM Budget Check

### **Peak VRAM Usage (Sequential)**

| Component | VRAM | When Active |
|-----------|------|-------------|
| **Ollama (qwen3.6:35b)** | ~30 GB | During LLM/VLM stages (analyze, adapt, plan, critique) |
| **ComfyUI Flux 2.0** | ~20 GB | During render_character (image gen) |
| **ComfyUI LTX-2.3** | ~44 GB | During generate_video (video gen) |
| **Fish Audio S2** | ~8 GB | During synthesize_voice (TTS) |

**Peak (worst-case simultaneous):** ~102 GB (if Ollama stays loaded during ComfyUI)
**Typical peak:** ~74 GB (Ollama unloads when ComfyUI stages run)

**Target GPU:** NVIDIA G200 (143 GB VRAM) → **41-69 GB headroom** ✅

### **Fallback Services VRAM (if enabled)**

| Service | VRAM | Notes |
|---------|------|-------|
| A1111 SD 1.5 | ~4 GB | Old render fallback |
| Wan 2.2 | ~14 GB | Old video fallback |
| MuseTalk | ~6 GB | Lip-sync for Wan path |
| Chatterbox | ~2 GB | Old TTS fallback |

---

## ✅ 7. Environment Variables Check

### **Critical Variables in .env**

```bash
# MUST be set (already in committed .env ✅)
HF_TOKEN=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

# GPU settings (default in .env ✅)
VIDEO_ME_WHISPER_DEVICE=cuda
VIDEO_ME_WHISPER_COMPUTE_TYPE=float16

# Adapter selection (default stack ✅)
VIDEO_ME_RENDER_ADAPTER=comfyui_flux
VIDEO_ME_VIDEO_ADAPTER=ltx
VIDEO_ME_TTS_ADAPTER=fish_s2

# Service URLs (default ✅)
VIDEO_ME_LLM_BASE_URL=http://localhost:11434/v1
VIDEO_ME_COMFYUI_BASE_URL=http://localhost:8188
VIDEO_ME_FISH_S2_BASE_URL=http://localhost:8025

# Models (default ✅)
VIDEO_ME_LLM_MODEL=qwen3.6:35b
VIDEO_ME_CRITIQUE_MODEL=qwen3.6:35b
```

### **Verification Command**

```bash
source .env
echo "HF_TOKEN: ${HF_TOKEN:0:10}..."  # Show first 10 chars
echo "Render adapter: $VIDEO_ME_RENDER_ADAPTER"
echo "Video adapter: $VIDEO_ME_VIDEO_ADAPTER"
echo "TTS adapter: $VIDEO_ME_TTS_ADAPTER"
echo "LLM model: $VIDEO_ME_LLM_MODEL"
```

Expected output:
```
HF_TOKEN: hf_IAoRldp...
Render adapter: comfyui_flux
Video adapter: ltx
TTS adapter: fish_s2
LLM model: qwen3.6:35b
```

---

## ✅ 8. Known Issues & Mitigations

### **Issue 1: HF Token Expires in 2 Days**

**Impact:** Flux 2.0 download will fail after 2026-06-30
**Mitigation:**
```bash
# Generate new token at: https://huggingface.co/settings/tokens
# Update .env on GPU server:
nano .env  # Update HF_TOKEN=...
source .env
```

### **Issue 2: Ollama Binary Wiped on Pod Restart**

**Impact:** `ollama` command not found after restart
**Mitigation:** `start_services.sh` automatically reinstalls (line 49-52) ✅

### **Issue 3: LoRAs Need Retraining for Flux 2.0**

**Impact:** Existing SD 1.5 LoRAs won't work with Flux 2.0
**Status:** Placeholder LoRAs in place (bootstrap SD 1.5 format)
**Next step:** Retrain with `flux_train_network.py` after setup

### **Issue 4: ComfyUI Takes ~30s to Start**

**Impact:** First health check may fail
**Mitigation:** `start_services.sh` waits up to 80s (line 168) ✅

### **Issue 5: LTX Audio Misalignment on Non-Standard Resolutions**

**Impact:** Lip-sync issues if using non-native resolutions
**Mitigation:** `LtxAdapter` defaults to 1280×720 (native) ✅
**Documented:** Line 43-44 of `adapters/generate_video/ltx_adapter.py`

---

## ✅ 9. Final Pre-Flight Checklist

**Before starting GPU installation, verify:**

- ☑ `.env` file has valid HF_TOKEN (expires 2026-06-30)
- ☑ Repository pushed to `master` branch (commit `8e0c313`)
- ☑ GPU machine has CUDA + NVIDIA driver installed
- ☑ Network volume mounted at `/workspace` (for RunPod)
- ☑ At least 150 GB free disk space (127 GB downloads + workspace)
- ☑ Internet connection stable (2-4 hour download window)
- ☑ Access to HuggingFace (not blocked by firewall)
- ☑ GitHub access for repo clone
- ☑ Target GPU has ≥100 GB VRAM (G200 143 GB recommended)

**After installation, verify:**

- ☑ All services respond to health checks
- ☑ `python -m scripts.check_runtime_readiness` passes
- ☑ `python -m scripts.check_track_b` passes
- ☑ `python -m pytest -q` shows 313 tests passing
- ☑ LoRA files exist in `loras/` (placeholder bootstrap OK for now)
- ☑ Voice files exist in `voices/` (bootstrap gTTS WAVs OK for now)

---

## 🚀 Quick Start Command Block

**Copy-paste this entire block on the GPU machine:**

```bash
#!/bin/bash
set -euo pipefail

# 1. Clone repo
cd /workspace
git clone https://github.com/rekha0708/video_me.git || (cd video_me && git pull)
cd video_me

# 2. Load environment
source .env
echo "HF_TOKEN loaded: ${HF_TOKEN:0:10}..."

# 3. Run setup (2-4 hours)
bash scripts/setup_gpu.sh

# 4. Start services
bash scripts/start_services.sh

# 5. Verify
python -m scripts.check_runtime_readiness
python -m scripts.check_track_b
python -m pytest -q

echo "✅ Installation complete! Ready to run pipeline."
```

---

**📋 All checks pass? You're ready to deploy! 🎬**

