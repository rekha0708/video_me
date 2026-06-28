# TCP Ports — video_me Pipeline

Complete list of all TCP ports used by the video_me pipeline services and web UIs.

---

## 📊 **Port Usage Summary**

| Port | Service | Protocol | Required? | Purpose |
|------|---------|----------|-----------|---------|
| **11434** | Ollama | HTTP | ✅ **Always** | LLM + VLM (qwen3.6:35b) for all text/image/video stages |
| **8188** | ComfyUI | HTTP/WebSocket | ✅ **Default** | Flux 2.0 Dev image gen + LTX-2.3 22B video gen |
| **8025** | Fish Audio S2 | HTTP | ✅ **Default** | Voice synthesis (EN + HI + 80 languages) |
| **8765** | Human Approval UI | HTTP | ✅ **Always** | Web UI for storyboard + image approval (shared port) |
| **8020** | Chatterbox TTS | HTTP | ⚠️ Fallback | Voice synthesis (EN only, `TTS_ADAPTER=chatterbox`) |
| **7860** | AUTOMATIC1111 | HTTP | ⚠️ Fallback | SD 1.5 image gen (`RENDER_ADAPTER=a1111`) |
| **8030** | Wan 2.2 | HTTP | ⚠️ Fallback | Image-to-video (`VIDEO_ADAPTER=wan`) |
| **8040** | MuseTalk | HTTP | ⚠️ Fallback | Lip-sync for Wan path (`VIDEO_ADAPTER=wan`) |

---

## ✅ **Required Ports (Default Stack)**

### **Must Be Exposed for Pipeline to Work:**

```bash
# 3 required services + 1 approval UI
11434   # Ollama (LLM + VLM)
8188    # ComfyUI (Flux 2.0 + LTX-2.3)
8025    # Fish Audio S2 (TTS)
8765    # Human approval web UI (storyboard + images)
```

### **Firewall Rules (if needed):**
```bash
# Allow inbound on required ports
sudo ufw allow 11434/tcp comment "Ollama LLM+VLM"
sudo ufw allow 8188/tcp comment "ComfyUI Flux+LTX"
sudo ufw allow 8025/tcp comment "Fish Audio S2 TTS"
sudo ufw allow 8765/tcp comment "Human approval UI"
```

---

## ⚠️ **Fallback Ports (Optional)**

### **Only Needed if Using Fallback Adapters:**

```bash
# Fallback services (start only if configured)
8020    # Chatterbox TTS (if TTS_ADAPTER=chatterbox)
7860    # AUTOMATIC1111 (if RENDER_ADAPTER=a1111)
8030    # Wan 2.2 server (if VIDEO_ADAPTER=wan)
8040    # MuseTalk (if VIDEO_ADAPTER=wan)
```

### **When to Use Fallback Ports:**
- **8020 (Chatterbox):** English-only TTS, simpler setup than Fish S2
- **7860 (A1111):** SD 1.5 fallback if Flux 2.0 has issues
- **8030 + 8040 (Wan + MuseTalk):** Old video generation path (slower, 2-stage)

---

## 🌐 **Detailed Port Descriptions**

### **Port 11434 — Ollama (LLM + VLM)**
- **Service:** Ollama HTTP API
- **Model:** qwen3.6:35b (MoE 35B, natively multimodal)
- **Used by:**
  - `analyze_content` — content analysis
  - `adapt_script` — script transformation
  - `plan_shots` — storyboard generation
  - `critique_plan` — plan quality scoring
  - `critique_images` — image candidate selection
  - `critique` — video frame analysis
- **Health check:** `GET http://localhost:11434/api/tags`
- **VRAM:** ~30 GB
- **Start command:** `OLLAMA_MODELS=/workspace/ollama ollama serve`

---

### **Port 8188 — ComfyUI (Flux 2.0 + LTX-2.3)**
- **Service:** ComfyUI HTTP + WebSocket API
- **Models:**
  - Flux 2.0 Dev (32B) — image generation
  - LTX-2.3 22B distilled — video generation
- **Used by:**
  - `render_character` — generate character images with LoRA
  - `generate_video` — image-to-video with native lip-sync
- **Health check:** `GET http://localhost:8188/`
- **VRAM:** ~20 GB (Flux) + ~44 GB (LTX) = 64 GB (sequential)
- **Start command:** `python3 /workspace/ComfyUI/main.py --listen 0.0.0.0 --port 8188`
- **WebSocket:** Used for workflow execution progress updates

---

### **Port 8025 — Fish Audio S2 (TTS)**
- **Service:** Fish Audio S2 HTTP API
- **Languages:** English, Hindi, + 78 more
- **Used by:** `synthesize_voice` — text-to-speech with voice cloning
- **Health check:** `GET http://localhost:8025/health`
- **VRAM:** ~20 GB
- **Start command:** `uvicorn services.fish_s2_server:app --host 0.0.0.0 --port 8025`
- **Voice cloning:** Uses reference WAV files from `voices/kids_duo/`

---

### **Port 8765 — Human Approval UI (Shared)**
- **Service:** FastAPI web UI (served by adapters)
- **Used by:**
  1. `web_approval_adapter` — storyboard approval after plan critique
  2. `image_approval_adapter` — image grid approval after candidate selection
- **Access:** `http://localhost:8765` (opens automatically in browser)
- **Endpoints:**
  - `GET /` — Approval UI HTML
  - `POST /approve` — Approve storyboard/images
  - `POST /reject` — Reject with notes (triggers re-plan for storyboard)
  - `POST /override` — Override image selection (image grid only)
- **Timeout:** 24 hours (configurable via `VIDEO_ME_APPROVAL_TIMEOUT_HOURS`)
- **CI bypass:**
  - `VIDEO_ME_AUTO_APPROVE_PLAN=true` — skip storyboard approval
  - `VIDEO_ME_AUTO_APPROVE_IMAGES=true` — skip image approval

---

### **Port 8020 — Chatterbox TTS (Fallback)**
- **Service:** Chatterbox TTS HTTP API
- **Languages:** English only
- **Used by:** `synthesize_voice` (if `VIDEO_ME_TTS_ADAPTER=chatterbox`)
- **Health check:** `GET http://localhost:8020/health`
- **VRAM:** ~2 GB
- **Start command:** `uvicorn services.chatterbox_server:app --host 0.0.0.0 --port 8020`

---

### **Port 7860 — AUTOMATIC1111 (Fallback)**
- **Service:** AUTOMATIC1111 Stable Diffusion WebUI API
- **Model:** Stable Diffusion 1.5
- **Used by:** `render_character` (if `VIDEO_ME_RENDER_ADAPTER=a1111`)
- **Health check:** `GET http://localhost:7860/sdapi/v1/sd-models`
- **VRAM:** ~12 GB
- **Start command:** `bash webui.sh --api --listen --port 7860`

---

### **Port 8030 — Wan 2.2 (Fallback)**
- **Service:** Wan 2.2 image-to-video HTTP API
- **Model:** Wan 2.2 (12B)
- **Used by:** `generate_video` (if `VIDEO_ME_VIDEO_ADAPTER=wan`)
- **Health check:** `GET http://localhost:8030/health`
- **VRAM:** ~12 GB
- **Start command:** `uvicorn services.wan_server:app --host 0.0.0.0 --port 8030`
- **Note:** Requires MuseTalk (port 8040) for lip-sync

---

### **Port 8040 — MuseTalk (Fallback)**
- **Service:** MuseTalk lip-sync HTTP API
- **Used by:** `lip_sync` (if `VIDEO_ADAPTER=wan`)
- **Health check:** `GET http://localhost:8040/health`
- **VRAM:** ~8 GB
- **Start command:** `uvicorn services.musetalk_server:app --host 0.0.0.0 --port 8040`
- **Note:** Only needed when using Wan 2.2 video path

---

## 🔧 **Configuration via Environment Variables**

### **Change Service Ports:**
```bash
# .env or shell
VIDEO_ME_APPROVAL_PORT=8765              # Human approval UI
VIDEO_ME_LLM_BASE_URL=http://localhost:11434/v1
VIDEO_ME_COMFYUI_BASE_URL=http://localhost:8188
VIDEO_ME_FISH_S2_BASE_URL=http://localhost:8025
VIDEO_ME_TTS_BASE_URL=http://localhost:8020        # Chatterbox fallback
VIDEO_ME_SD_BASE_URL=http://localhost:7860         # A1111 fallback
VIDEO_ME_WAN_BASE_URL=http://localhost:8030        # Wan fallback
VIDEO_ME_LIPSYNC_BASE_URL=http://localhost:8040    # MuseTalk fallback
```

### **Remote Services:**
```bash
# Use remote Ollama instance
VIDEO_ME_LLM_BASE_URL=http://192.168.1.100:11434/v1

# Use remote ComfyUI instance
VIDEO_ME_COMFYUI_BASE_URL=http://192.168.1.101:8188
```

---

## 🐳 **Docker Port Mapping**

If running services in Docker containers:

```yaml
# docker-compose.yml
services:
  ollama:
    ports:
      - "11434:11434"
  
  comfyui:
    ports:
      - "8188:8188"
  
  fish_s2:
    ports:
      - "8025:8025"
```

---

## ✅ **Quick Port Check**

```bash
# Check which ports are listening
netstat -tlnp | grep -E '(11434|8188|8025|8765|8020|7860|8030|8040)'

# Or with lsof
lsof -i :11434  # Ollama
lsof -i :8188   # ComfyUI
lsof -i :8025   # Fish S2
lsof -i :8765   # Approval UI
```

---

## 🔒 **Security Notes**

### **Localhost-Only (Default):**
All services bind to `0.0.0.0` but should be behind a firewall or accessed via SSH tunnel in production.

### **SSH Tunnel (Remote Access):**
```bash
# Forward approval UI to local machine
ssh -L 8765:localhost:8765 user@gpu-server

# Access at http://localhost:8765 on your local machine
```

### **No Authentication:**
Services have no built-in authentication. Use firewall rules or reverse proxy with auth if exposing to network.

---

## 📋 **TL;DR — Port Checklist**

### **Minimal Setup (Default Stack):**
```bash
✅ 11434  # Ollama
✅ 8188   # ComfyUI
✅ 8025   # Fish S2
✅ 8765   # Approval UI
```

### **Fallback Setup (If Needed):**
```bash
⚠️ 8020   # Chatterbox (if TTS_ADAPTER=chatterbox)
⚠️ 7860   # A1111 (if RENDER_ADAPTER=a1111)
⚠️ 8030   # Wan (if VIDEO_ADAPTER=wan)
⚠️ 8040   # MuseTalk (if VIDEO_ADAPTER=wan)
```

**All services run on localhost by default. External access requires firewall rules or SSH tunnels.**
