# Environment Setup Guide — video_me

This guide walks you through setting up your `.env` file for the video_me pipeline.

---

## 🚀 Quick Start

```bash
# 1. Copy the template
cp .env.example .env

# 2. Edit .env and add your HuggingFace token
nano .env  # or use your preferred editor

# 3. Find this line and replace with your token:
HF_TOKEN=hf_REPLACE_WITH_YOUR_TOKEN_HERE

# 4. Save and exit

# 5. Verify it loaded
source .env
echo $HF_TOKEN
# Should show: hf_...
```

---

## 🔑 Step 1: Get HuggingFace Token

### **Why you need it:**
Flux 2.0 Dev is a gated model. You need a HuggingFace token to download it.

### **How to get it:**

1. **Accept the Flux 2.0 license** (one-time):
   - Go to: https://huggingface.co/black-forest-labs/FLUX.2-dev
   - Click **"Agree and access repository"**
   - Read and accept the license

2. **Create a token** (one-time):
   - Go to: https://huggingface.co/settings/tokens
   - Click **"Create new token"** → **"Fine-grained"**
   - **Name:** `video_me_model_download`
   - **Permissions:** Check these boxes:
     ```
     ✅ Read access to contents of all public gated repos you can access
     ```
   - Click **"Create token"**
   - **Copy the token** (starts with `hf_...`)

3. **Add to .env file:**
   ```bash
   HF_TOKEN=hf_your_actual_token_here_1234567890abcdef
   ```

---

## ⚙️ Step 2: Configure Settings

The `.env` file has sensible defaults for GPU machines. You typically only need to change:

### **Required Changes:**
```bash
# MUST CHANGE: Add your actual HuggingFace token
HF_TOKEN=hf_your_token_here
```

### **Optional Changes:**

#### **For CPU-only testing:**
```bash
VIDEO_ME_WHISPER_DEVICE=cpu
VIDEO_ME_WHISPER_COMPUTE_TYPE=int8
```

#### **For different language:**
```bash
VIDEO_ME_TARGET_LANGUAGE=hi     # Hindi
VIDEO_ME_TARGET_LANGUAGE=both   # English + Hindi (runs twice)
```

#### **For CI/automated testing:**
```bash
VIDEO_ME_AUTO_APPROVE_PLAN=true
VIDEO_ME_AUTO_APPROVE_IMAGES=true
```

#### **For production (PostgreSQL + S3):**
```bash
VIDEO_ME_JOB_STORE=postgres
VIDEO_ME_ARTIFACT_STORE=s3
VIDEO_ME_POSTGRES_DSN=postgresql://user:pass@host:5432/video_me
VIDEO_ME_S3_ENDPOINT_URL=https://s3.amazonaws.com
VIDEO_ME_S3_BUCKET=your-bucket-name
VIDEO_ME_S3_ACCESS_KEY_ID=your_access_key
VIDEO_ME_S3_SECRET_ACCESS_KEY=your_secret_key
```

---

## 📋 Configuration Reference

### **Service URLs**

| Variable | Default | Purpose |
|----------|---------|---------|
| `VIDEO_ME_COMFYUI_BASE_URL` | `http://localhost:8188` | ComfyUI (Flux 2.0 + LTX-2.3) |
| `VIDEO_ME_FISH_S2_BASE_URL` | `http://localhost:8025` | Fish Audio S2 TTS |
| `VIDEO_ME_LLM_BASE_URL` | `http://localhost:11434/v1` | Ollama LLM/VLM |

### **Adapter Selection**

| Variable | Default | Options |
|----------|---------|---------|
| `VIDEO_ME_RENDER_ADAPTER` | `comfyui_flux` | `comfyui_flux`, `a1111` |
| `VIDEO_ME_VIDEO_ADAPTER` | `ltx` | `ltx`, `wan` |
| `VIDEO_ME_TTS_ADAPTER` | `fish_s2` | `fish_s2`, `chatterbox` |

### **Model Configuration**

| Variable | Default | Purpose |
|----------|---------|---------|
| `VIDEO_ME_LLM_MODEL` | `qwen3.6:35b` | Text generation (analyze, adapt, plan) |
| `VIDEO_ME_CRITIQUE_MODEL` | `qwen3.6:35b` | Image + video critique |
| `VIDEO_ME_WHISPER_MODEL_SIZE` | `medium` | Transcription quality |

---

## ✅ Step 3: Verify Configuration

```bash
# Load environment variables
source .env

# Check HF token is set
echo $HF_TOKEN
# Should show: hf_...

# Run runtime readiness check (after services are started)
python -m scripts.check_runtime_readiness

# Expected output:
# ✅ Ollama LLM/VLM API: OK
# ✅ ComfyUI (Flux 2.0 + LTX-2.3): OK
# ✅ Fish Audio S2: OK
```

---

## 🔒 Security Best Practices

### **✅ DO:**
- ✅ Keep `.env` file **git-ignored** (already configured)
- ✅ Use **read-only** HF tokens (not write)
- ✅ Set **token expiration** (30-90 days)
- ✅ Rotate tokens regularly
- ✅ Use different tokens for different projects

### **❌ DON'T:**
- ❌ Commit `.env` to version control
- ❌ Share `.env` file publicly
- ❌ Use tokens with write permissions
- ❌ Hardcode secrets in code
- ❌ Reuse tokens across multiple machines

---

## 🆘 Troubleshooting

### **Problem: Token not recognized**

```bash
# Check token is set
echo $HF_TOKEN
# If empty, source the .env file:
source .env
```

### **Problem: Permission denied downloading Flux 2.0**

**Solutions:**
1. Accept license: https://huggingface.co/black-forest-labs/FLUX.2-dev
2. Check token has gated repo access (see Step 1)
3. Verify token is valid: https://huggingface.co/settings/tokens

### **Problem: Can't find .env file**

```bash
# Check if it exists
ls -la .env

# If not, create from template
cp .env.example .env
nano .env
```

---

## 📦 What Happens During Setup

When you run `bash scripts/setup_gpu.sh`:

1. **Reads HF_TOKEN** from environment (loaded from `.env`)
2. **Downloads Flux 2.0 Dev** (~64 GB) — requires token
3. **Downloads LTX-2.3** (~42 GB) — token optional but recommended
4. **Installs ComfyUI** + custom nodes
5. **Sets up services**

Total download: **~106 GB**  
Time estimate: **2-4 hours** on fast connection

---

## 📝 Example .env File

See the created `.env` file in the project root for a fully-documented example with all available options.

---

## 🎯 Next Steps

After configuring `.env`:

1. ✅ **Run setup:**
   ```bash
   source .env
   bash scripts/setup_gpu.sh
   ```

2. ✅ **Start services:**
   ```bash
   bash scripts/start_services.sh
   ```

3. ✅ **Verify everything works:**
   ```bash
   python -m scripts.check_runtime_readiness
   python -m scripts.check_track_b
   ```

4. ✅ **Run first pipeline test** (see README.md)

---

**Questions?** See `UPGRADE_NOTES.md` for model details or `README.md` for full documentation.
