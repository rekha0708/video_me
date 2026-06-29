# DEPLOY — video_me GPU Installation

> **Source of truth is `CLAUDE.md`.** This is a deploy walkthrough, not a sign-off.
> Verify on the target box before claiming readiness.

**Default stack:** musubi-tuner Flux 2.0 (image, subprocess) + ComfyUI/LTX-2.3 (video, :8188)
+ Fish Audio S2 (TTS, :8025) + Ollama qwen3.6:35b (:11434).

---

## Open items before a real run

- ⬜ **Track B assets** — `loras/kids_duo_max.safetensors` / `kids_duo_zoe.safetensors` must be real trained Flux LoRAs. `python -m scripts.check_track_b` must report READY.
- ⬜ **Model downloads** — Flux 2.0 (musubi) + LTX-2.3 (ComfyUI) + text encoders. Confirm the HF repo ids in `setup_gpu.sh` against current upstream releases.
- ⬜ **Fish Audio S2 model** — confirm `services/fish_s2_server.py` loads your Fish/OpenAudio build.
- ⬜ **Tests** — 315 collected; re-run `pytest` on the target box and record the number (local Mac/py3.14 venv: 312 pass / 3 fail, the 3 being stale `json_repair` tests — see `CLAUDE.md`).
- ✅ **Adapter wiring** — config + `setup_gpu.sh` + `start_services.sh` aligned on `musubi_flux` / `ltx` / `fish_s2`.

---

## Pre-install requirements

**Hardware:** NVIDIA GPU with ≥100 GB VRAM (G200 143 GB recommended); CUDA driver (`nvidia-smi` works); ≥150 GB free disk.
**Network:** stable connection (multi-hour download window); HuggingFace + GitHub reachable.
**Accounts:** HuggingFace account; Flux 2.0 license accepted; **your own** HF token exported as `HF_TOKEN` (do not commit it — see Security below).
**Environment:** Ubuntu/Linux, Python 3.11+, network volume at `/workspace` (RunPod).

---

## Quick start (on the GPU machine)

```bash
set -euo pipefail

cd /workspace
git clone <your-repo-url> video_me || (cd video_me && git pull)
cd video_me

export HF_TOKEN=hf_...          # your token; never commit it

bash scripts/setup_gpu.sh       # installs musubi-tuner + ComfyUI/LTX + Fish S2 + Ollama
bash scripts/start_services.sh  # starts Ollama + ComfyUI + Fish S2

python -m scripts.check_runtime_readiness
python -m scripts.check_track_b
python -m pytest -q
```

Legacy fallbacks are opt-in: add `--with-a1111 / --with-chatterbox / --with-wan` to
`setup_gpu.sh`, set `VIDEO_ME_START_LEGACY=1` for `start_services.sh`, and select with
`VIDEO_ME_RENDER_ADAPTER` / `VIDEO_ME_VIDEO_ADAPTER` / `VIDEO_ME_TTS_ADAPTER`.

---

## Resource budget (approximate)

| Component | Download | VRAM | When |
|---|---|---|---|
| qwen3.6:35b (Ollama) | ~20 GB | ~30 GB | LLM/VLM stages |
| Flux 2.0 (musubi-tuner) | ~60 GB + encoders | ~20 GB | render_character |
| LTX-2.3 22B distilled (ComfyUI) | ~42 GB | ~44 GB | generate_video |
| Fish Audio S2 | ~300 MB | ~8 GB | synthesize_voice |
| ComfyUI + system/python deps | ~1 GB | — | framework |

Peak VRAM ~74 GB typical (Ollama evicts before GPU-heavy stages; see
`core/workflow.py:_unload_ollama_model`), ~102 GB worst case. On a 143 GB G200 that
leaves comfortable headroom. Sizes are estimates — confirm against actual downloads.

Opt-in fallbacks add: A1111 +~4 GB, Wan 2.2 +~30 GB, MuseTalk/Chatterbox small.

---

## Post-install tasks

1. **Train Flux 2.0 LoRAs** (Track B) — SD 1.5 weights are incompatible. See `LORA_TRAINING_AUTOMATED.md`.
2. **Record real voice references** — replace the gTTS bootstrap WAVs at `voices/kids_duo/{max,zoe}.wav` (10–30 s clear single-speaker, WAV/MP3/FLAC).
3. **Run the pipeline end-to-end** on a short test video — see `CLAUDE.md` "Running the pipeline".

---

## Known issues & mitigations

| Issue | Mitigation |
|---|---|
| `ollama` binary wiped on RunPod restart | Reinstall `curl -fsSL https://ollama.ai/install.sh \| sh`, then `start_services.sh`. |
| ComfyUI takes ~30–60 s to start | `start_services.sh` waits before the health check; re-run readiness if the first probe fails. |
| LTX audio misalignment on non-standard resolutions | `LtxAdapter` defaults to native 1280×720 — keep native sizes. |
| Flux 2.0 LoRAs not trained | render_character uses placeholder/missing LoRAs until Track B is done; `check_track_b` blocks real runs. |

---

## Final checklist

Before install: CUDA + driver present · `/workspace` volume mounted · ≥150 GB disk ·
HF token exported · Flux 2.0 license accepted · HuggingFace/GitHub reachable.

After install (all must pass on the GPU box):
- ☐ Services healthy (Ollama 11434, ComfyUI 8188, Fish S2 8025)
- ☐ `python -m scripts.check_runtime_readiness`
- ☐ `python -m scripts.check_track_b` (must be READY — finish Track B)
- ☐ `python -m pytest -q` (record the real number)

---

## Security

Do **not** commit `HF_TOKEN` or any secret to git or to these docs. Export it in the
shell (`export HF_TOKEN=...`) or put it in `.env`, which is git-ignored. **If a token was
ever committed (`.env` was tracked in earlier history), rotate it now** at
https://huggingface.co/settings/tokens.
