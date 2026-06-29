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

**Why:** [[project-video-me]]
