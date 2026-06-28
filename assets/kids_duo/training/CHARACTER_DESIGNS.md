# kids_duo Cast — Character Design Reference

Visual reference guide for all four characters in the kids_duo educational series.

---

## 👦 Max (Primary Character)

**Age:** 5 years old  
**Role:** Enthusiastic big-kid teacher, learner in cooking/art

### Visual Design
- **Face:** Round friendly face, light olive skin tone with cool undertone
- **Hair:** Short wavy brown hair
- **Eyes:** Big warm brown eyes
- **Outfit:** Blue and white striped t-shirt, navy blue shorts, white sneakers with blue laces
- **Build:** Child proportions, energetic upright posture

### Signature Expressions
- Wide-eyed "aha!" teaching face
- Proud big-kid grin when Zoe succeeds
- Puzzled but trying face when learning from Zoe
- Counting on fingers with tongue out in concentration

### Signature Moves
- Points at imaginary letters in the air with big arm sweep
- Claps hands in a big cheer when Zoe gets it right
- Stirs bowl very carefully and slowly while following Zoe's instructions
- Holds pencil proudly and draws a letter in the air

**Prompts:** `assets/kids_duo/training/max_prompts.txt` (20 prompts)

---

## 👧 Zoe (Primary Character)

**Age:** 3 years old  
**Role:** Confident little expert in cooking/art/creative play, learner in letters/words

### Visual Design
- **Face:** Round chubby toddler face, light olive skin tone with cool undertone
- **Hair:** Soft black loosely curled hair in two small puffs with pink bows
- **Eyes:** Big sparkly dark brown eyes
- **Outfit:** Pink polka-dot dress, white t-shirt underneath, pink shoes
- **Details:** Often has flour on hands or paint on fingers
- **Build:** Toddler proportions (bigger head-to-body ratio, chubbier cheeks, shorter than Max)

### Signature Expressions
- Hands-on-hips confident "watch me!" pose
- Giggly laugh when Max makes a mess
- Serious concentrating face when mixing or painting
- Delighted clap when something comes out perfectly

### Signature Moves
- Stirs mixing bowl with both hands very seriously
- Holds up a painting proudly at arm's length to show Max
- Dabs pretend makeup brush on cheek with a big grin
- Traces letters carefully with one finger while Max teaches

**Prompts:** `assets/kids_duo/training/zoe_prompts.txt` (20 prompts)

---

## 👩 Mom (Supporting Character)

**Age:** Mid-30s adult  
**Role:** Warm and nurturing parent, encourages both kids

### Visual Design
- **Face:** Gentle oval face, warm medium-brown skin tone, kind brown eyes with gentle crow's feet
- **Hair:** Long dark brown hair in a neat low ponytail with side-swept bangs
- **Outfit:** Soft lavender cardigan over white blouse, blue jeans, comfortable gray flats
- **Build:** Adult proportions (taller than children, mature facial features)

### Key Expressions
- Warm nurturing smile (default)
- Proud delighted expression when kids succeed
- Patient listening expression with slight head tilt
- Gentle concerned expression with caring smile
- Encouraging gesture with hands offering support

### Common Poses
- Kneeling down to child height to talk with kids
- Sitting cross-legged on ground for story time
- Hands clasped together in proud approval
- One hand on heart, touched emotional expression
- Arms slightly open in welcoming gesture

**Prompts:** `assets/kids_duo/training/mom_prompts.txt` (20 prompts)

---

## 👨 Dad (Supporting Character)

**Age:** Mid-30s adult  
**Role:** Friendly and playful parent, encourages both kids

### Visual Design
- **Face:** Square friendly face, light olive skin tone, warm hazel eyes with laugh lines
- **Hair:** Short neat dark brown hair with slight wave
- **Facial Hair:** Trimmed beard
- **Outfit:** Navy blue henley shirt, khaki pants, brown casual shoes
- **Build:** Adult proportions (taller and broader than children, mature facial features)

### Key Expressions
- Warm encouraging grin (default)
- Big proud smile with eyes crinkled with joy
- Playful silly expression (tongue out, one eye winking)
- Gentle patient expression, attentive listening
- Enthusiastic thumbs up with big smile

### Common Poses
- Kneeling on one knee to child height
- Hands on hips in playful superhero pose
- Sitting cross-legged on ground with kids
- Arms spread wide in playful airplane pose
- Both arms spread wide in big welcoming hug gesture
- One hand stroking beard in thoughtful expression

**Prompts:** `assets/kids_duo/training/dad_prompts.txt` (20 prompts)

---

## 🎨 Design Consistency Rules

### Color Palette
- **Max:** Blue tones (blue/white striped shirt, navy shorts, blue laces)
- **Zoe:** Pink tones (pink polka dots, pink bows, pink shoes)
- **Mom:** Warm earth/purple tones (lavender cardigan, blue jeans, brown hair)
- **Dad:** Navy/neutral tones (navy henley, khaki pants, brown shoes)

### Visual Hierarchy
1. **Primary characters** (Max, Zoe): Child-friendly cartoon proportions, highly expressive
2. **Supporting characters** (Mom, Dad): More realistic adult proportions, warm and approachable

### Distinguishing Features
- **Max vs Zoe:** Max is taller, less round face, blue palette; Zoe has toddler proportions, rounder face, pink palette
- **Mom vs Dad:** Mom has ponytail + cardigan; Dad has beard + henley shirt
- **Adults vs Kids:** Height difference, face maturity, body proportions

### Shared Style
- Soft cartoon illustration, 2D flat shading
- Clean white backgrounds for training images
- Bright cheerful colors
- Crisp clean line art
- Family-friendly, educational content aesthetic

---

## 📁 Training Files Structure

```
assets/kids_duo/training/
├── max_prompts.txt        # 20 prompts for Max
├── zoe_prompts.txt        # 20 prompts for Zoe
├── mom_prompts.txt        # 20 prompts for Mom
├── dad_prompts.txt        # 20 prompts for Dad
├── images/
│   ├── max/              # Max training images + captions
│   ├── zoe/              # Zoe training images + captions
│   ├── mom/              # Mom training images + captions
│   └── dad/              # Dad training images + captions
└── kohya_config.toml     # Flux 2.0 LoRA training config
```

---

## 🚀 Quick Start

Generate training images for all characters:

```bash
# Primary characters (essential)
python scripts/generate_training_images.py --character max
python scripts/generate_training_images.py --character zoe

# Supporting characters (optional, add as needed)
python scripts/generate_training_images.py --character mom
python scripts/generate_training_images.py --character dad
```

---

## 🎯 Training Priority

### Phase 1: Core Cast (Required)
1. **Max** — Primary educator character
2. **Zoe** — Primary student character

### Phase 2: Extended Cast (Optional)
3. **Mom** — Family context, encouragement
4. **Dad** — Family context, encouragement

**Recommendation:** Start with Max and Zoe. Add Mom and Dad later if you need family scenes or parental encouragement segments in videos.

---

## 📊 Expected Output

After training all characters:

```
loras/
├── kids_duo_max.safetensors   (~50 MB) ✅ Required
├── kids_duo_zoe.safetensors   (~50 MB) ✅ Required
├── kids_duo_mom.safetensors   (~50 MB) ⚠️  Optional
└── kids_duo_dad.safetensors   (~50 MB) ⚠️  Optional
```

**Pipeline compatibility:**
- Max + Zoe: Fully supported in current pipeline (2-character limit per scene)
- Mom + Dad: Can be added as cast members in `config/casts/kids_duo.yaml` when needed
