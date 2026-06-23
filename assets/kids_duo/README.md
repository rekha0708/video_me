# Kids Duo Track B Asset Plan

`kids_duo` is the final cast: Max and Zoe. This folder tracks the non-code work
needed before the pipeline can render and speak with the final cast.

## Source Of Truth

- Cast config: `config/casts/kids_duo.yaml`
- LoRA drop location: `loras/`
- Voice drop location: `voices/kids_duo/`
- Verification: `python -m scripts.check_track_b`
- Runtime/GPU readiness: `bash scripts/setup_gpu.sh` then `python -m scripts.check_runtime_readiness`

## Character Reference Sheets

Create one approved reference sheet per character before LoRA training.

First-pass review sheets:

- `assets/kids_duo/reference/max_reference_sheet_v1.png`
- `assets/kids_duo/reference/zoe_reference_sheet_v1.png`

Max:

- Soft cartoon 5-year-old boy.
- Round friendly face, light olive skin tone with cool undertone, short wavy brown hair, big warm brown eyes.
- Blue and white striped t-shirt, navy shorts, white sneakers with blue laces.
- Energetic posture; reads as older and taller than Zoe.
- Expressions: teaching face, proud grin, puzzled-but-trying face, finger-counting concentration.

Zoe:

- Soft cartoon 3-year-old girl.
- Round chubby toddler face, light olive skin tone with cool undertone, soft black loosely curled hair in two puffs with pink bows.
- Pink polka-dot dress over white t-shirt, pink shoes.
- Often has flour on hands or paint on fingers.
- Expressions: confident watch-me pose, giggly laugh, serious concentrating face, delighted clap.

Each sheet should include front view, 3/4 view, side view, 4-6 expressions, 3-4 signature poses,
and a plain background. Avoid any resemblance to existing kids-show characters.

Status: provisionally accepted for code testing. Final operator approval is still
recommended before generating a full training set or training production LoRAs.

## LoRA Training Inputs

Per character:

- 15-30 consistent images in the final visual style.
- Captions that include the trigger stem: `kids_duo_max` or `kids_duo_zoe`.
- Same outfit and core silhouette across the dataset.
- Some variation in pose, camera distance, and expression.

Expected outputs:

```text
loras/kids_duo_max.safetensors
loras/kids_duo_zoe.safetensors
```

The render adapter injects tags like `<lora:kids_duo_max:0.9>` automatically.
During temporary code-test smoke runs only, explicit `TEST-ONLY placeholder`
LoRAs can be accepted with `VIDEO_ME_RENDER_ALLOW_PLACEHOLDER_LORA=true`; the
adapter then omits fake LoRA tags from prompts. Strict real-run readiness still
requires trained weights.

## Voice References

Record or synthesize one clean reference per character:

```text
voices/kids_duo/max.wav
voices/kids_duo/zoe.wav
```

Use the scripts in `voice_scripts.md` as recording text. Keep each take natural,
clear, and free of music/noise.
