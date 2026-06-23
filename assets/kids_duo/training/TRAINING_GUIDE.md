# Track B LoRA Training Guide — kids_duo (Leonardo.ai)

Two characters: **Max** (5-year-old boy) and **Zoe** (3-year-old girl).

---

## Important: two paths

Leonardo.ai's trained Custom Models are **platform-locked** — you can use them inside Leonardo
but can't download a `.safetensors` file. This means:

| | Path A — Replicate LoRA | Path B — Leonardo adapter |
|---|---|---|
| Generate images | Leonardo.ai | Leonardo.ai |
| Train LoRA | Replicate (cloud kohya) | Leonardo Custom Model |
| render_character | AUTOMATIC1111 (Track D) | Leonardo API (no A1111 needed) |
| Code changes | None | Update diffusion_adapter.py |
| Local GPU needed | No (Replicate handles it) | No |

**Recommendation:** Start with Path A if you want zero code changes and a simpler handoff.
Choose Path B if you want to eliminate the AUTOMATIC1111 Track D dependency entirely.

---

## Step 1 — Generate training images in Leonardo.ai (both paths)

1. Open Leonardo.ai → **Image Generation**
2. Select model: **Leonardo Anime XL** (preferred) or **DreamShaper v7**
3. Enable **Alchemy** toggle for higher quality
4. Set: Guidance Scale **7** · Steps **25** · Dimensions **768 × 768**
5. Paste the **Shared Negative Prompt** (top of `max_prompts.txt` / `zoe_prompts.txt`)

Run each of the 20 prompts from `max_prompts.txt`, then the 20 from `zoe_prompts.txt`.
Append the **Shared Positive Suffix** from the prompt file to every prompt.

Download all outputs. Cull to the **15–20 cleanest per character** — consistent outfit,
consistent hair, clean white background, no anatomy artifacts.

Place kept images:
```
assets/kids_duo/training/images/max/   ← max_001.png … max_020.png
assets/kids_duo/training/images/zoe/   ← zoe_001.png … zoe_020.png
```

---

## Path A — Train on Replicate, use with AUTOMATIC1111

### Step 2A — Train on Replicate

Go to [replicate.com/ostris/flux-dev-lora-trainer](https://replicate.com/ostris/flux-dev-lora-trainer).

Upload a zip of your `images/max/` folder (images + caption `.txt` files — see `caption_template_max.txt`).

Key settings:
| Setting | Value |
|---|---|
| `trigger_word` | `kids_duo_max` |
| `steps` | 1000 |
| `lora_rank` | 32 |
| `learning_rate` | 0.0004 |

Run. When complete, download the output as `kids_duo_max.safetensors`.

Repeat for Zoe: zip `images/zoe/`, trigger word `kids_duo_zoe`, output `kids_duo_zoe.safetensors`.

### Step 3A — Drop files and verify

```
loras/kids_duo_max.safetensors   ← download from Replicate
loras/kids_duo_zoe.safetensors   ← download from Replicate
```

```bash
python -m scripts.check_track_b
```

Render still requires AUTOMATIC1111 running (Track D).

---

## Path B — Train on Leonardo, update the render adapter

### Step 2B — Train Custom Model on Leonardo.ai

1. Go to Leonardo.ai → **Train a Custom Model**
2. Upload your curated images for Max
3. Set:
   - **Model type**: Character
   - **Instance prompt** (trigger word): `kids_duo_max`
   - **Resolution**: 768
4. Train. Note the **Model ID** from the URL when done (e.g. `abc123-...`)
5. Repeat for Zoe, note its Model ID

### Step 3B — Update diffusion_adapter.py

The current adapter calls AUTOMATIC1111's `/sdapi/v1/txt2img`. It needs a Leonardo variant
that calls `POST https://cloud.leonardo.ai/api/rest/v1/generations` and polls for results.

Tell your engineer (or ask Claude Code):

> "Add a `LeonardoRenderAdapter` in `adapters/render_character/leonardo_adapter.py` that
> calls the Leonardo.ai REST API instead of AUTOMATIC1111. Model IDs are stored in config
> (kids_duo_max_model_id, kids_duo_zoe_model_id). Wire it into AppConfig and load_app_config()."

The existing `DiffusionRenderAdapter` and its tests stay untouched — the Leonardo adapter
is additive. Swap which adapter is wired in `core/workflow.py`.

### Step 4B — Dummy LoRA files (keep gate passing)

The file-gate check (`_check_lora`) still runs at startup. Keep the placeholder files:
```
loras/kids_duo_max.safetensors   ← placeholder already there, fine to keep
loras/kids_duo_zoe.safetensors   ← same
```
Set `VIDEO_ME_RENDER_ALLOW_PLACEHOLDER_LORA=true` in your env — the Leonardo adapter
won't use them for anything, but the gate check will pass.

---

## Verify (both paths)

```bash
python -m scripts.check_track_b
```

Expected:
```
Track B: READY_FOR_CODE_TESTS   ← Path B (placeholder files + allow flag)
Track B: READY                  ← Path A (real trained weights)
```

---

## Trigger words

| Character | Trigger word |
|---|---|
| Max | `kids_duo_max` |
| Zoe | `kids_duo_zoe` |

Used in Leonardo training instance prompt and (Path A) injected as `<lora:kids_duo_max:0.9>`
by `DiffusionRenderAdapter._build_prompt()`.
