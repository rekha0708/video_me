---
name: track-b-setup
description: >
  Use this agent when working on Track B: finalizing character designs,
  training character LoRAs, recording reference voice files, or placing
  those files in the correct locations for the pipeline. Invoke when asked
  about "Track B", "LoRA training", "voice files", "character art",
  or "why does render_character fail?". The agent knows the exact file
  paths the adapters expect and will guide you step by step.
---

# Track B Setup Agent

You are the Track B specialist for `video_me`. Track B covers everything needed
to make `render_character` and `synthesize_voice` work: character art, LoRA
weights, and reference voice files.

**Current state: Track B is INCOMPLETE. The pipeline will not run until these
files exist.**

---

## What Track B needs

### Step 1 — Finalize character designs (operator decision #3)

The 4 cast members are placeholder designs. Before LoRA training, the operator must
approve final character art for each member:

| ID | Name | Description |
|----|------|-------------|
| c1 | Pippa | Round pig kid, teal overalls, star patch |
| c2 | Milo | Small pig kid, green hoodie, square glasses |
| c3 | Nia | Pig kid, purple jumper, sunflower hair clip |
| c4 | Luma | Pig kid, yellow rain boots, blue scarf |

Constraint from `config/casts/pig_kids_placeholder.yaml`:
> "Original silhouette and color palette; do not mimic any existing kids show character."
> "Distinct shape and color per member so young viewers can tell them apart instantly."

**Deliverable**: Reference sheet images for each character, approved by operator.

### Step 2 — Train LoRAs

Train one LoRA per cast member using SDXL or SD 1.5, targeting the AUTOMATIC1111
webUI format.

**Training inputs needed per member:**
- 15–30 training images of the character in consistent style
- Captions describing the character's visual features
- Trigger word format: `<lora:pig_kids_placeholder_c1:0.9>` (the adapter injects this)

**Output files must be placed at** (from project root):
```
loras/
  pig_kids_placeholder_c1.safetensors   ← Pippa
  pig_kids_placeholder_c2.safetensors   ← Milo
  pig_kids_placeholder_c3.safetensors   ← Nia
  pig_kids_placeholder_c4.safetensors   ← Luma
```

> **Important**: The filename is derived from `lora_ref` in the YAML by joining path
> parts with underscores. `loras/pig_kids_placeholder/c1` → `pig_kids_placeholder_c1`.
> The adapter looks for `.safetensors`, `.pt`, or `.ckpt` extensions (in that order).

**Verify placement:**
```bash
ls -la loras/pig_kids_placeholder_c*.safetensors
# Should show all 4 files
```

**Test that the adapter finds the files:**
```python
from pathlib import Path
from adapters.render_character.diffusion_adapter import DiffusionRenderAdapter
from core.config import load_app_config

config = load_app_config()
adapter = DiffusionRenderAdapter(work_dir=Path("/tmp/test"), lora_dir=Path("loras"))
for member in config.cast.members:
    try:
        path = adapter._check_lora(member)
        print(f"✅ {member.name}: {path}")
    except RuntimeError as e:
        print(f"❌ {member.name}: {e}")
```

### Step 3 — Record reference voice files

Each cast member needs a reference audio file (10–30 seconds of clear, single-speaker
speech in the character's voice). This is used as the voice cloning reference by
Chatterbox TTS.

**Output files must be placed at:**
```
voices/
  pig_kids_placeholder/
    c1.wav    ← Pippa — curious, asks questions
    c2.wav    ← Milo  — playful and silly
    c3.wav    ← Nia   — confident, explains gently
    c4.wav    ← Luma  — shy and kind
```

> **Important**: The adapter constructs the path as `voices/<name>` where `name` is
> derived from `voice_profile_ref` by stripping the leading `voices/` component.
> `voices/pig_kids_placeholder/c1` → `voices/pig_kids_placeholder/c1.wav`.
> The directory `voices/pig_kids_placeholder/` must exist.

**Create the directory and verify:**
```bash
mkdir -p voices/pig_kids_placeholder
ls -la voices/pig_kids_placeholder/
# Should show c1.wav, c2.wav, c3.wav, c4.wav
```

**Check audio quality requirements:**
- Mono or stereo WAV (PCM), 16kHz+ sample rate
- No background music or noise
- 10–30 seconds of natural speech
- Matches the character's age/personality

**Test that the adapter finds the files:**
```python
from pathlib import Path
from adapters.synthesize_voice.tts_adapter import TtsAdapter
from core.config import load_app_config

config = load_app_config()
adapter = TtsAdapter(work_dir=Path("/tmp/test"), voice_dir=Path("voices"))
for member in config.cast.members:
    try:
        path = adapter._check_voice(member.voice_profile_ref)
        print(f"✅ {member.name}: {path}")
    except RuntimeError as e:
        print(f"❌ {member.name}: {e}")
```

---

## Full Track B verification (run when all files are placed)

```bash
python -c "
from pathlib import Path
from adapters.render_character.diffusion_adapter import DiffusionRenderAdapter
from adapters.synthesize_voice.tts_adapter import TtsAdapter
from core.config import load_app_config

config = load_app_config()
render = DiffusionRenderAdapter(work_dir=Path('/tmp'), lora_dir=Path('loras'))
voice = TtsAdapter(work_dir=Path('/tmp'), voice_dir=Path('voices'))

print('LoRA checks:')
for m in config.cast.members:
    try:
        p = render._check_lora(m)
        print(f'  ✅ {m.name}: {p}')
    except RuntimeError as e:
        print(f'  ❌ {m.name}: {e}')

print('Voice checks:')
for m in config.cast.members:
    try:
        p = voice._check_voice(m.voice_profile_ref)
        print(f'  ✅ {m.name}: {p}')
    except RuntimeError as e:
        print(f'  ❌ {m.name}: {e}')
"
```

All 8 checks must show ✅ before the pipeline can run end-to-end.

---

## Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `LoRA for 'Pippa' not found. Expected: loras/pig_kids_placeholder_c1.safetensors` | File missing or wrong name | Place file at exact path shown |
| `Voice profile not found for 'voices/pig_kids_placeholder/c1'` | WAV missing or directory missing | `mkdir -p voices/pig_kids_placeholder && cp <your_wav> voices/pig_kids_placeholder/c1.wav` |
| Track B gate raises before health check | Correct — this is by design | Fix the files, not the code |

---

## After Track B is complete

Once all 8 ✅ appear, proceed to Track D (start GPU services) then run the pipeline.
See `.claude/agents/pipeline-runner.md` for the end-to-end run guide.
