---
name: project-wan22
description: Wan 2.2 I2V installation quirks and path gotchas
metadata: 
  node_type: memory
  type: project
  originSessionId: 0f42bdac-7ce1-4470-b65d-7073873d419a
---

Wan 2.2 is installed as a fallback image-to-video adapter (port 8030).

**Path quirk:** Repo cloned into nested dir — `WAN_DIR` must be `/workspace/Wan2.2/Wan2.2` (not `/workspace/Wan2.2`). The inner dir is the repo root containing the `wan/` package.

**Install approach:** `pip install -e /workspace/Wan2.2/Wan2.2/ --no-deps` makes `import wan` work globally in `.venv_wan` without sys.path hacks.

**numpy:** Wan pins `numpy<2` but works fine with numpy 2.x. System scipy 1.18.0 requires `numpy>=2.0` — install `numpy>=2.0,<2.3` in the wan venv.

**Missing deps (not in Wan requirements.txt):** `decord`, `librosa`, `rotary-embedding-torch`, `peft`

**Model:** `/workspace/Wan2.2-I2V-A14B/`
**Venv:** `/workspace/.venv_wan/`

**Status as of 2026-07-04: video generation confirmed working end-to-end** (real generated frame verified). Getting there required fixing, in order:
1. `easydict`/`accelerate` missing + `huggingface_hub` unpinned (inherited an incompatible 1.22.0 from elsewhere on the box, breaks transformers 4.51.3/peft 0.19.x which need <1.0) — pin `huggingface_hub>=0.30.0,<1.0` in setup_gpu.sh.
2. Running Ollama+ComfyUI+Fish+Wan simultaneously OOMs on a single H200 (143GB) — must unload Ollama (`keep_alive=0`) before Wan inference, same as the real pipeline already does before its GPU-heavy stages.
3. A failed call (e.g. an OOM mid-transfer) can leave the **resident** Wan pipeline in a corrupted mixed CPU/GPU device state — `wan_server.py` has no recovery path, the whole process must be restarted.
4. `services/wan_server.py`'s `_inference()` was missing a batch dim before `save_video()` — `WanI2V.generate()` returns `(C,N,H,W)`, but `save_video()`'s `unbind(2)` needs `(B,C,N,H,W)` so dim 2 lands on frames not height. Fixed with `video_tensor[None]` (matches Wan2.2's own `generate.py`). Without this the output is a real, playable, but garbage-dimension MP4 — easy to miss if you only check the HTTP status.

**MuseTalk lip-sync (port 8040) — confirmed NOT viable for this project's content.** All dependencies now install cleanly (see feedback_dashboard_worker.md-adjacent lessons: `munkres`, `chumpy`, `json-tricks`, `xtcocotools` all need `--no-build-isolation` or are simply missing from the original install list) and PyTorch 2.8's `torch.load` `weights_only=True` default is patched via `services/musetalk_compat/sitecustomize.py` (injected via subprocess PYTHONPATH, not an in-place package edit). But MuseTalk's face detector (trained on photorealistic human faces) does not recognize our stylized cartoon character renders at all — confirmed via a live `/lipsync` call returning a byte-identical passthrough of the input. `video_adapter=wan` therefore produces real video with **no synced mouth movement**. LTX's native lip-sync (generative, not detection-based) remains the only working lip-sync path for this content.

**VRAM contention when `video_adapter=wan` is used alongside the rest of the pipeline (confirmed 2026-07-04, job `20260704-020418-33o`).** Unlike ComfyUI/LTX (which only loads its video model on-demand during `generate_video`), the Wan adapter runs as a **permanently-resident model server** — confirmed via `nvidia-smi --query-compute-apps`: ~53.6 GiB held continuously from server startup, on top of ComfyUI (~30 GiB, idle but still resident), Fish S2, and MuseTalk. `core/workflow.py` unloads `qwen3.6:35b` from Ollama once "before the shot loop" — but the per-shot loop's `image_critique_adapter` calls Ollama again (same model) between every character's render, auto-reloading ~20-30 GB back into VRAM (Ollama has no persistent unload, just load-on-demand). With `wan`'s extra resident ~52 GB baked in, this reload leaves no headroom for musubi-tuner's Flux 2.0 load: job crashed with `torch.OutOfMemoryError` loading Zoe's render (Max's succeeded first, before the critique-triggered reload). Total observed: 5 resident processes = ~116 GiB out of 143.8 GiB before Flux 2.0 even tries to allocate.
**Why this didn't happen on the default `ltx` stack:** CLAUDE.md's VRAM budget (~114 GB, 29 GB headroom) is calculated for `ltx`, which doesn't add a permanently-resident ~52 GB process the way `wan` does — so the same Ollama-reload-during-critique pattern didn't previously starve `render_character`.
**Unresolved as of 2026-07-04:** job `20260704-020418-33o` is `failed`, paused at render_character/zoe, decision pending — options discussed: (a) revert to default `ltx` (matches documented budget + working lip-sync), (b) stop the MuseTalk server (already confirmed non-functional, frees VRAM, but the `lip_sync` stage's health check would need to tolerate it being down), (c) retry as-is (risks repeating the OOM).

**Why:** [[project-video-me]], [[project-next-steps]]
