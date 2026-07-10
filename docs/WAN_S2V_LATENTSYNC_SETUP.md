# Wan2.2 S2V and LatentSync Setup

This project now treats **Wan2.2 S2V** as the main singing-video path.
Use **Wan2.2 I2V + LatentSync** only as a fallback/comparison path when you
want to generate visual motion first and repair lips afterward.

Official references:
- Wan2.2 README: https://github.com/Wan-Video/Wan2.2
- Wan2.2 S2V weights: https://huggingface.co/Wan-AI/Wan2.2-S2V-14B
- LatentSync README: https://github.com/bytedance/LatentSync
- LatentSync 1.6 weights: https://huggingface.co/ByteDance/LatentSync-1.6

## Recommendation

For singing, choose:

```bash
VIDEO_ME_VIDEO_ADAPTER=wan_s2v
VIDEO_ME_WAN_S2V_BASE_URL=http://localhost:8031
VIDEO_ME_TTS_ADAPTER=fish_s2
```

The workflow renders the still image, generates or slices the audio, unloads
Fish S2 if needed, then sends **image + audio** to Wan S2V. The separate
`lip_sync` stage is skipped because the video model is audio-conditioned.

For comparison/repair runs, choose:

```bash
VIDEO_ME_VIDEO_ADAPTER=wan
VIDEO_ME_LIPSYNC_ADAPTER=latentsync
VIDEO_ME_WAN_BASE_URL=http://localhost:8030
VIDEO_ME_LATENTSYNC_BASE_URL=http://localhost:8041
```

That path generates silent Wan I2V first, then sends the clip plus audio to
LatentSync. It can improve ordinary speech clips, but it is less ideal than
S2V for singing because repair happens after the motion has already been
generated.

## One-command setup

Default singing stack:

```bash
bash scripts/setup_gpu.sh
bash scripts/start_services.sh
python -m scripts.check_runtime_readiness
```

Add Wan I2V + LatentSync fallback:

```bash
bash scripts/setup_gpu.sh --with-wan-i2v --with-latentsync
```

Add the legacy MuseTalk repair fallback too:

```bash
bash scripts/setup_gpu.sh --with-musetalk
```

## Manual Wan S2V setup

```bash
cd /workspace
git clone https://github.com/Wan-Video/Wan2.2.git Wan2.2
python3 -m venv --system-site-packages /workspace/.venv_wan
/workspace/.venv_wan/bin/pip install --upgrade pip
/workspace/.venv_wan/bin/pip install -r /workspace/Wan2.2/requirements.txt
/workspace/.venv_wan/bin/pip install -r /workspace/Wan2.2/requirements_s2v.txt
/workspace/.venv_wan/bin/pip install -e /workspace/Wan2.2 --no-deps
/workspace/.venv_wan/bin/pip install fastapi uvicorn python-multipart "huggingface_hub[cli]"
/workspace/.venv_wan/bin/huggingface-cli download Wan-AI/Wan2.2-S2V-14B \
  --local-dir /workspace/Wan2.2-S2V-14B
```

Start the wrapper:

```bash
cd /workspace/video_me
WAN_DIR=/workspace/Wan2.2 \
WAN_S2V_MODEL_DIR=/workspace/Wan2.2-S2V-14B \
/workspace/.venv_wan/bin/uvicorn services.wan_s2v_server:app \
  --host 0.0.0.0 --port 8031
```

Health check:

```bash
curl http://localhost:8031/health
```

## Manual LatentSync setup

```bash
cd /workspace
git clone https://github.com/bytedance/LatentSync.git LatentSync
python3.10 -m venv --system-site-packages /workspace/.venv_latentsync
/workspace/.venv_latentsync/bin/pip install --upgrade pip
/workspace/.venv_latentsync/bin/pip install -r /workspace/LatentSync/requirements.txt
/workspace/.venv_latentsync/bin/pip install fastapi uvicorn python-multipart "huggingface_hub[cli]"
/workspace/.venv_latentsync/bin/huggingface-cli download ByteDance/LatentSync-1.6 \
  latentsync_unet.pt --local-dir /workspace/LatentSync/checkpoints
/workspace/.venv_latentsync/bin/huggingface-cli download ByteDance/LatentSync-1.6 \
  whisper/tiny.pt --local-dir /workspace/LatentSync/checkpoints
```

Start the wrapper:

```bash
cd /workspace/video_me
LATENTSYNC_DIR=/workspace/LatentSync \
/workspace/.venv_latentsync/bin/uvicorn services.latentsync_server:app \
  --host 0.0.0.0 --port 8041
```

Health check:

```bash
curl http://localhost:8041/health
```

## Dashboard selection

In the new-job dashboard:

- **Video Model → Wan 2.2 S2V** for the main singing path.
- **Video Model → Wan 2.2 I2V** plus **Lip-sync Repair → LatentSync** for fallback comparisons.
- **Lip-sync Repair → MuseTalk** only for legacy/fast repair checks.

Retrying a completed or failed job can also override `video_adapter` and
`lipsync_adapter`, so you can compare S2V against Wan I2V + LatentSync using
the same cached rendered images.
