# Track B LoRA Training Guide — kids_duo (Leonardo.ai)

Two characters: **Max** (5-year-old boy) and **Zoe** (3-year-old girl).

---

## Important: three paths

Leonardo.ai's trained Custom Models are **platform-locked** — you can use them inside Leonardo
but can't download a `.safetensors` file. This means:

| | Path A — local sd-scripts LoRA | Path B — Replicate LoRA | Path C — Leonardo adapter |
|---|---|---|---|
| Generate images | Leonardo.ai | Leonardo.ai | Leonardo.ai |
| Train LoRA | Local A100 + sd-scripts | Replicate cloud kohya | Leonardo Custom Model |
| render_character | AUTOMATIC1111 (Track D) | AUTOMATIC1111 (Track D) | Leonardo API (no A1111 needed) |
| Code changes | None | None | Update diffusion_adapter.py |
| Local GPU needed | Yes | No | No |

**Current completed path:** Path A was used locally on 2026-06-24. Max and Zoe LoRAs
were trained for 1000 steps each, rank 32, against SD 1.5. Use Path B only if the
local weights must be retrained without a GPU, and Path C only if replacing the
AUTOMATIC1111 render adapter is worth the extra code.

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

## Path A — Train locally with sd-scripts, use with AUTOMATIC1111

This is the current completed path for the local workspace. Dataset configs live at:

```text
assets/kids_duo/training/dataset_max.toml
assets/kids_duo/training/dataset_zoe.toml
```

Venv workaround: use `/workspace/venv` for sd-scripts training. The project `.venv`
is still the normal env for pipeline commands, but it had a Torch/CUDA mismatch for
training on this GPU image. The trainer dependencies were installed with:

```bash
/workspace/venv/bin/python3 -m pip install -r /workspace/sd-scripts/requirements.txt
```

Max command:

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 /workspace/venv/bin/python3 /workspace/sd-scripts/train_network.py \
  --pretrained_model_name_or_path /workspace/stable-diffusion-webui/models/Stable-diffusion/v1-5-pruned-emaonly.safetensors \
  --dataset_config assets/kids_duo/training/dataset_max.toml \
  --output_dir loras --output_name kids_duo_max --save_model_as safetensors \
  --network_module networks.lora --network_dim 32 --network_alpha 16 \
  --train_batch_size 1 --max_train_steps 1000 \
  --learning_rate 1e-4 --unet_lr 1e-4 --text_encoder_lr 5e-5 \
  --lr_scheduler cosine_with_restarts --lr_warmup_steps 50 \
  --optimizer_type AdamW8bit --mixed_precision fp16 --save_precision fp16 \
  --clip_skip 2 --seed 42 --cache_latents --gradient_checkpointing --sdpa \
  --max_data_loader_n_workers 0 --save_every_n_steps 500
```

Zoe uses the same command with `dataset_zoe.toml`, `--output_name kids_duo_zoe`,
and `--seed 43`.

Expected outputs:

```text
loras/kids_duo_max.safetensors
loras/kids_duo_zoe.safetensors
```

Training logs are local-only under `training_logs/` and are ignored by git.

---

## Path B — Train on Replicate, use with AUTOMATIC1111

### Step 2B — Train on Replicate

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

### Step 3B — Drop files and verify

```
loras/kids_duo_max.safetensors   ← download from Replicate
loras/kids_duo_zoe.safetensors   ← download from Replicate
```

```bash
python -m scripts.check_track_b
```

Render still requires AUTOMATIC1111 running (Track D).

---

## Path C — Train on Leonardo, update the render adapter

### Step 2C — Train Custom Model on Leonardo.ai

1. Go to Leonardo.ai → **Train a Custom Model**
2. Upload your curated images for Max
3. Set:
   - **Model type**: Character
   - **Instance prompt** (trigger word): `kids_duo_max`
   - **Resolution**: 768
4. Train. Note the **Model ID** from the URL when done (e.g. `abc123-...`)
5. Repeat for Zoe, note its Model ID

### Step 3C — Update diffusion_adapter.py

The current adapter calls AUTOMATIC1111's `/sdapi/v1/txt2img`. It needs a Leonardo variant
that calls `POST https://cloud.leonardo.ai/api/rest/v1/generations` and polls for results.

Tell your engineer (or ask Claude Code):

> "Add a `LeonardoRenderAdapter` in `adapters/render_character/leonardo_adapter.py` that
> calls the Leonardo.ai REST API instead of AUTOMATIC1111. Model IDs are stored in config
> (kids_duo_max_model_id, kids_duo_zoe_model_id). Wire it into AppConfig and load_app_config()."

The existing `DiffusionRenderAdapter` and its tests stay untouched — the Leonardo adapter
is additive. Swap which adapter is wired in `core/workflow.py`.

### Step 4C — Keep local file gates satisfied

The current pipeline still verifies the cast asset paths before rendering. If using a
future Leonardo adapter, keep valid files at the expected paths or update the gate for
that adapter:

```text
loras/kids_duo_max.safetensors
loras/kids_duo_zoe.safetensors
```

In the current workspace these are real trained LoRA weights, not placeholders.

---

## Verify

```bash
python -m scripts.check_track_b
```

Expected:
```
Track B: INCOMPLETE            ← current local state until voice WAVs exist
Track B: READY                  ← LoRAs plus voice WAVs are present
```

---

## Trigger words

| Character | Trigger word |
|---|---|
| Max | `kids_duo_max` |
| Zoe | `kids_duo_zoe` |

Used in Leonardo training instance prompt and injected by the current AUTOMATIC1111
`DiffusionRenderAdapter._build_prompt()` as `<lora:kids_duo_max:0.9>` /
`<lora:kids_duo_zoe:0.9>`.
