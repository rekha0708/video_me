# LoRA Training Guide — kids_duo (Flux 2.0)

Train the Max and Zoe character LoRAs for the `render_character` stage. The default
render path is **musubi-tuner Flux 2.0**, so train Flux LoRAs (SD 1.5 weights won't work).

> **Status:** `loras/kids_duo_max.safetensors` is missing and `kids_duo_zoe.safetensors`
> is a TEST-ONLY placeholder — `check_track_b` reports INCOMPLETE until both are trained.

---

## 🎯 Overview

The intended workflow uses your own Flux 2.0 stack to produce training images, then
kohya `flux_train_network.py` to train the LoRA:

✅ **Flux 2.0 (via musubi-tuner)** generates the training images locally  
✅ **qwen3.6:35b** refines each prompt for the base model  
✅ **Human approval loop** — review each image before saving  
✅ **Auto-generated captions** — one `.txt` per image  

> ⚠️ **The `scripts/generate_training_images.py` helper described below is planned and
> not yet in the repo.** Until it lands, generate training images manually (musubi-tuner,
> or any external tool such as Leonardo.ai — see "Alternatives" at the end), place them
> under `assets/kids_duo/training/images/<char>/`, and skip to Step 3 (training).

---

## 📋 Prerequisites

Before starting, ensure:

- ✅ GPU setup complete (`bash scripts/setup_gpu.sh` finished)
- ✅ musubi-tuner + Flux 2.0 weights present (image generation engine)
- ✅ Ollama + qwen3.6:35b responding (`http://localhost:11434`) for prompt refinement

**Quick check:**
```bash
python -m scripts.check_runtime_readiness
```

---

## 🚀 Quick Start

### **Step 1: Generate Training Images (Human-in-Loop)**

```bash
# For Max character (20 images with approval UI)
python scripts/generate_training_images.py --character max

# For Zoe character
python scripts/generate_training_images.py --character zoe

# Auto-approve all (skip approval UI, for testing)
python scripts/generate_training_images.py --character max --auto-approve

# Resume from specific prompt (if interrupted)
python scripts/generate_training_images.py --character max --start-from 010
```

**What happens:**
1. Script reads prompts from `assets/kids_duo/training/max_prompts.txt`
2. For each prompt:
   - qwen3.6:35b refines prompt for Flux 2.0 compatibility
   - musubi-tuner + Flux 2.0 generates the image (base model, no LoRA)
   - Browser opens at `http://localhost:8765` showing the image
   - You approve ✅ or reject ❌
   - Approved images saved to `assets/kids_duo/training/images/max/`
   - Caption `.txt` file auto-generated with refined prompt

**Time estimate:** ~2-3 minutes per image (20 sec generation + review time)

---

### **Step 2: Review & Curate Dataset**

```bash
# Check generated images
ls -lh assets/kids_duo/training/images/max/

# Expected output:
# max_001.png + max_001.txt
# max_002.png + max_002.txt
# ...
# max_020.png + max_020.txt
```

**Quality criteria:**
- ✅ Clean white background
- ✅ Consistent character appearance (hair, skin tone, outfit)
- ✅ Correct pose/expression matching prompt
- ✅ No anatomy issues (extra fingers, weird limbs)
- ✅ Clear, sharp rendering

**If you rejected too many:**
- Tweak prompts in `assets/kids_duo/training/max_prompts.txt`
- Re-run from that prompt: `--start-from NNN`

**Recommended:** 15-20 high-quality images per character for good LoRA

---

### **Step 3: Train LoRA with Flux 2.0**

Now train the LoRA using your approved images:

```bash
# Install kohya_ss if not already done
cd /workspace
git clone https://github.com/kohya-ss/sd-scripts.git
cd sd-scripts
pip install -r requirements.txt

# Update config to point to your Flux 2.0 model
# (Already configured in assets/kids_duo/training/kohya_config.toml)

# Train Max LoRA
cd /workspace/video_me
accelerate launch /workspace/sd-scripts/flux_train_network.py \
  --config_file assets/kids_duo/training/kohya_config.toml

# Train Zoe LoRA (edit kohya_config.toml: change output_name to "kids_duo_zoe")
# Then re-run the accelerate command
```

**Training settings (already in kohya_config.toml):**
- Model: `flux.2-dev.safetensors` (32B params)
- Network: LoRA rank 32, alpha 16
- Epochs: 25 (~500-625 steps for 20 images)
- Batch size: 16
- Learning rate: 1e-4 (UNET), 5e-5 (text encoder)
- Mixed precision: BF16 (optimal for H200/G200)
- FlashAttention: enabled (fast on Hopper)

**Time estimate:** ~2-4 hours per character on H200/G200

**Output:**
```
loras/kids_duo_max.safetensors  (~50 MB, Flux 2.0 format)
loras/kids_duo_zoe.safetensors
```

---

### **Step 4: Test Your LoRA**

```bash
# Quick test via ComfyUI UI
# 1. Open http://localhost:8188
# 2. Load workflow: flux_lora_txt2img.json
# 3. Set LoRA: kids_duo/kids_duo_max.safetensors
# 4. Prompt: "full body front view, 5-year-old cartoon boy, blue striped shirt, cheerful"
# 5. Generate and verify it looks like Max

# Or test via pipeline (once LoRAs are trained)
python -m scripts.check_track_b
# Should now pass: ✅ LoRA files exist and are valid
```

---

## 🔧 How the Automated Script Works

### **Workflow Diagram**

```
[Load Prompts]
    ↓
[For each prompt]
    ↓
[LLM Refine] ← qwen3.6:35b adapts prompt for Flux 2.0
    ↓          (removes trigger token, adds quality boosters)
[Flux Generate] ← musubi-tuner + Flux 2.0 base model (no LoRA)
    ↓             ~20 seconds per image
[Show in Browser] ← http://localhost:8765 approval UI
    ↓
[Human Decision]
    ├─ ✅ Approve → Save PNG + TXT caption
    └─ ❌ Reject → Delete, log reason
```

### **Prompt Refinement (qwen3.6:35b)**

**Original prompt (from max_prompts.txt):**
```
[001] full body front view, kids_duo_max, 5-year-old cartoon boy, round friendly face,
light olive skin tone, blue striped t-shirt, navy shorts, white sneakers, happy expression
```

**LLM refinement (for Flux 2.0 base model):**
```
full body front view, 5-year-old cartoon boy, round friendly face, light olive skin tone,
blue striped t-shirt, navy shorts, white sneakers, happy expression, high quality digital
illustration, professional character design, clean white background
```

**Changes:**
- ❌ Removed `[001]` prefix
- ❌ Removed `kids_duo_max` trigger token (using base model, not LoRA)
- ✅ Kept all visual descriptors intact
- ✅ Added Flux quality boosters
- ✅ Emphasized clean white background

---

## 💡 Tips & Best Practices

### **Prompt Quality**
- ✅ Be specific about clothing, hair, skin tone, pose
- ✅ Include "clean white background" in every prompt
- ✅ Vary poses: full body, waist up, close-up faces
- ✅ Vary expressions: happy, surprised, concentrating, excited
- ❌ Avoid complex backgrounds or props
- ❌ Avoid multiple characters in one image

### **Dataset Curation**
- **Minimum:** 15 images per character
- **Optimal:** 20-25 images per character
- **Diversity:** Mix of poses, angles, expressions
- **Consistency:** Same outfit, hair, skin tone across all images
- **Quality over quantity:** 15 perfect images > 30 mediocre images

### **LoRA Training**
- **Rank 32** is good for characters (balance detail vs. flexibility)
- **Rank 64** for more complex styles or if overfitting
- **Rank 16** if underfitting (too generic)
- **Epochs:** Start with 25, adjust based on results
  - Underfitting: bland, generic, doesn't look like your character
  - Good fit: recognizable, consistent, flexible poses
  - Overfitting: rigid, can't vary pose, memorized training set

### **Testing LoRA Quality**
```bash
# Test with prompts NOT in training set
# Good sign: character recognizable in new poses/expressions
# Bad sign: only works with exact training prompts

# Example test prompts:
# - "kids_duo_max riding a bicycle, cartoon style"
# - "kids_duo_max reading a book, side view"
# - "kids_duo_max waving goodbye, cheerful expression"
```

---

## 🐛 Troubleshooting

### **Problem: ComfyUI not responding**

**Solution:**
```bash
bash scripts/start_services.sh
# Wait ~30s for ComfyUI to load models
curl http://localhost:8188/system_stats
```

### **Problem: qwen3.6:35b refining prompts poorly**

**Solution:** Manually edit refined prompts in the approval UI before training, or skip LLM refinement and use original prompts directly (edit script line 90-100).

### **Problem: Flux generating wrong style (too realistic, wrong art style)**

**Solution:** Add negative prompt to workflow template:
```json
"negative_prompt": "photorealistic, 3D render, realistic, photograph, hyperrealistic"
```

### **Problem: Images have backgrounds instead of white**

**Solution:** Emphasize in prompt: "solid white background, no objects, studio lighting, clean background"

### **Problem: Training takes too long**

**Solution:** Reduce `max_train_epochs` in `kohya_config.toml` from 25 to 15-20.

### **Problem: LoRA output doesn't look like character**

**Solution:**
1. Check dataset consistency (all images should look similar)
2. Increase `network_dim` from 32 to 64 in `kohya_config.toml`
3. Train longer (increase epochs)
4. Use more training images (aim for 20-25)

---

## 📊 Resource Usage

### **Image Generation (per image)**
- VRAM: ~20 GB (Flux 2.0)
- Time: ~20 seconds
- Disk: ~2 MB per PNG

### **LoRA Training (20 images, 25 epochs)**
- VRAM: ~30-40 GB (Flux 2.0 training)
- Time: ~2-4 hours on H200/G200
- Disk: ~50 MB output LoRA file

### **Total for Both Characters**
- Generation: ~80 images × 20s = 27 minutes active time (+ review time)
- Training: 2 characters × 3 hours = 6 hours
- **Total: ~1 day hands-off + occasional approval clicks**

---

## 🎯 Next Steps After Training

1. ✅ **Verify LoRAs exist:**
   ```bash
   ls -lh loras/
   # Should show:
   # kids_duo_max.safetensors  (~50 MB)
   # kids_duo_zoe.safetensors  (~50 MB)
   ```

2. ✅ **Test in pipeline:**
   ```bash
   python -m scripts.check_track_b
   # Should now pass LoRA checks
   ```

3. ✅ **Run end-to-end pipeline test** (see CLAUDE.md "Running the pipeline" —
   call `core.workflow.run_pipeline_job()` / `run_with_critique()`).

4. ✅ **Review output:**
   ```bash
   ls -lh review/
   # Check if Max and Zoe render consistently
   ```

---

## 📎 Reference

### Trigger words & assets

| Character | Trigger word | Prompts | Dataset config |
|---|---|---|---|
| Max | `kids_duo_max` | `assets/kids_duo/training/max_prompts.txt` | `dataset_max.toml` |
| Zoe | `kids_duo_zoe` | `assets/kids_duo/training/zoe_prompts.txt` | `dataset_zoe.toml` |

kohya config: `assets/kids_duo/training/kohya_config.toml`. Trained weights go in
`loras/` as `kids_duo_<name>.safetensors`. Training logs under `training_logs/` are
git-ignored.

### Alternatives (folded from the former Leonardo.ai guide)

**No local GPU for training?** Generate images with any tool (e.g. Leonardo.ai —
Leonardo Anime XL, Alchemy on, 768×768, with the shared negative prompt + positive
suffix from the prompt files), then train on **Replicate**
([ostris/flux-dev-lora-trainer](https://replicate.com/ostris/flux-dev-lora-trainer)):
upload a zip of `images/<char>/` (PNGs + caption `.txt`), set `trigger_word=kids_duo_<char>`,
`steps=1000`, `lora_rank=32`, `learning_rate=0.0004`; download the `.safetensors` into `loras/`.

**Leonardo Custom Model (platform-locked)** is also possible but its trained model can't
be exported as `.safetensors`; using it would require a new `LeonardoRenderAdapter`
calling the Leonardo REST API instead of the local render path.

### Verify

```bash
python -m scripts.check_track_b   # INCOMPLETE until both LoRAs + voice WAVs exist
```
