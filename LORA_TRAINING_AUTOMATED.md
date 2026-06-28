# Automated LoRA Training — Using Your Flux 2.0 Stack

**NEW:** Use your existing Flux 2.0 + qwen3.6:35b setup to generate training images with human-in-the-loop approval!

---

## 🎯 Overview

Instead of manually generating images in Leonardo.ai, you can now:

✅ **Use Flux 2.0 base model** (already installed) to generate training images  
✅ **Use qwen3.6:35b LLM** to refine prompts for optimal Flux output  
✅ **Human approval loop** — review each image before saving  
✅ **Auto-generate captions** — .txt files created automatically  
✅ **Resume from any point** — pick up where you left off

---

## 📋 Prerequisites

Before starting, ensure:

- ✅ GPU setup complete (`bash scripts/setup_gpu.sh` finished)
- ✅ Services running (`bash scripts/start_services.sh` finished)
- ✅ ComfyUI + Flux 2.0 responding (`http://localhost:8188`)
- ✅ Ollama + qwen3.6:35b responding (`http://localhost:11434`)

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
   - ComfyUI + Flux 2.0 generates image (base model, no LoRA)
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
[Flux Generate] ← ComfyUI + Flux 2.0 base model (no LoRA)
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

3. ✅ **Run end-to-end pipeline test:**
   ```bash
   python run_pipeline.py --source-url "YOUR_TEST_VIDEO" --rights-cleared
   ```

4. ✅ **Review output:**
   ```bash
   ls -lh review/
   # Check if Max and Zoe render consistently
   ```

---

## 🎬 You're Ready!

This automated workflow lets you:
- ✅ Generate training images using your own Flux 2.0 setup
- ✅ Review and approve each image before training
- ✅ Train production-quality Flux 2.0 LoRAs
- ✅ Use in the video_me pipeline immediately

**Start now:**
```bash
python scripts/generate_training_images.py --character max
```

Good luck! 🚀
