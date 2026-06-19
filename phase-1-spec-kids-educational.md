# Phase 1 Spec — Animated Multi-Character Kids' Educational Shorts

> **Reads with:** `orchestration-build-plan.md` (the master plan). This document specializes
> Phase 1 of that plan for the first concrete product instance.
> **Audience:** the AI coding agent building the system.
> **Deliverable of Phase 1:** a working end-to-end pipeline that turns a reference link into an
> original, captioned, age-appropriate animated short starring a fixed cast of 4 original
> child characters, for an educational-kids channel — with manual review before publishing.

This instance is deliberately *one configuration* of a general system. The genre (education-kids)
is a swappable `ChannelProfile`; the cast (4 pig kids) is a swappable `Cast`. Nothing about
"education" or "pigs" is hardcoded into the pipeline — see Section 8 (Extensibility).

---

## 1. What Phase 1 must produce

Given a reference educational kids' video link, the pipeline outputs a short (target ~30–60s,
9:16) in which the original 4-character cast performs an *original, transformed* script that
teaches the same concept — captioned, with the "made for kids" / AI-disclosure flags set, written
to a file for a human to review and publish. No critic loop yet (that is Phase 2), but the
guardrail checks in Section 6 are enforced now.

---

## 2. Concrete config for this instance

### 2a. ChannelProfile: `education_kids` (config/channels/education_kids.yaml)
```yaml
id: education_kids
genre_content: educational_kids
target_audience: { age_range: "3-6", reading: pre_reader }
tone: warm, simple, playful, encouraging
format: animated_character
aspect_ratio: "9:16"
target_length_sec: 45
language: en
made_for_kids: true            # platform designation (see guardrails)
disclosure_label_required: true
pedagogy:
  one_concept_per_video: true  # a single, concrete learning objective
  vocabulary_level: simple     # short sentences, concrete nouns
  repetition: true             # reinforce the key idea 2-3 times
  positive_framing: true
```

### 2b. Cast: `pig_kids_v1` (config/casts/pig_kids_v1.yaml)
A `Cast` is an ensemble of `CastMember`s. Species, count, and identities are all config.
```yaml
id: pig_kids_v1
species: pig                    # SWAPPABLE: dog, bear, robot, human-kid, etc.
is_original_synthetic: true     # REQUIRED true; design must be original, not an existing show
members:
  - id: c1
    name: <choose original name>
    gender: boy
    visual_descriptor: <original look — color, clothing, distinguishing feature>
    lora_ref: loras/pig_kids_v1/c1
    voice_profile_ref: voices/pig_kids_v1/c1   # designed synthetic child-like voice
    personality: curious, asks the questions
    signature_expressions: [wide-eyed wonder, big grin]
  - id: c2 { gender: boy,  name: ..., personality: playful/silly, ... }
  - id: c3 { gender: girl, name: ..., personality: knows-the-answer/explainer, ... }
  - id: c4 { gender: girl, name: ..., personality: shy/kind, ... }
design_constraints:
  - Original silhouette and color palette; do NOT mimic any existing kids' show character.
  - Distinct shape/color per member so young viewers tell them apart instantly.
  - Consistent across shots via per-member LoRA (built in Stage 4).
```

---

## 3. Data model additions/changes for this phase

Add to `core/models/` alongside the master-plan models.

**Cast** = `{ id, species, is_original_synthetic, members[CastMember], design_constraints }`
**CastMember** = `{ id, name, gender, visual_descriptor, lora_ref, voice_profile_ref, personality, signature_expressions[] }`

**LearningObjective** (the spine of an educational video) =
`{ concept, age_range, success_phrase, key_vocabulary[], reinforcement_count }`

**ContentMetadata** — extend with `learning_objective: LearningObjective` and keep
`content_genre`, `music_genre`, `topic`, `hook`, `structure[]`, `pacing`, `length_sec`, `call_to_action`.

**Script** — now multi-speaker. =
```
Script = {
  mode,                         # transformed (default for kids/educational sourcing)
  learning_objective,
  scenes: [ Scene ],
  caption_text,
  source_rights                 # required; see guardrails
}
Scene = { setting, characters_present[member_id], lines: [Line] }
Line  = { speaker member_id, text, expression?, action?, start?, end? }
```

**Storyboard** (output of the new shot-planner step, Section 4) =
```
Storyboard = { shots: [Shot] }
Shot = {
  shot_id, scene_ref,
  characters_on_screen[member_id],   # keep to 1-2 where possible
  setting, camera, action, dialogue_line_refs[],
  duration_sec
}
```

---

## 4. Key technical decision: shot-based generation

Do **not** attempt to generate a whole multi-character short in one pass. Multi-character
consistency degrades fast. Instead insert a **shot-planner** between script and video generation:

Pipeline for this instance becomes:
`ingest → understand → adapt_script → plan_shots → generate_assets → (per shot) generate_video + lip_sync → synthesize_voices → assemble → output`

- **`plan_shots`** (new capability `PlanShots`): decomposes the multi-speaker `Script` into a
  `Storyboard` of short shots. Rules: prefer 1–2 characters on screen per shot; one dialogue beat
  per shot; cut to the speaker; keep each shot 2–5s. This is what keeps generation tractable and
  keeps each character consistent.
- **Per-shot generation**: each shot is generated independently (image-to-video from the relevant
  character LoRA renders + the shot's action), then lip-synced to that shot's line.
- **Voices**: each `CastMember` has its own designed synthetic child-like voice; lines are
  synthesized per speaker and aligned to shots via timestamps.
- **Assemble**: concatenate shots in storyboard order, lay in dialogue + optional music + captions,
  format to 9:16.

Add `PlanShots` to the capability contracts in the master plan (`core/capabilities/`).

---

## 5. Stage-by-stage Phase 1 implementation (reference adapters)

1. **Ingest** — `yt-dlp` fetch + `ffmpeg` stream split. Record source + rights decision on the Job.
2. **Understand** — Whisper-family transcription (word timestamps); open LLM/VLM produces
   `ContentMetadata` *including* a `LearningObjective` extracted from the reference (what concept
   does the source teach?). Music-genre tagging optional this phase.
3. **Adapt script** — open LLM rewrites into a `transformed`, multi-speaker `Script` for the cast,
   following the `pedagogy` rules (one concept, simple vocab, repetition, positive framing) and
   assigning lines to members by personality (e.g. c1 asks, c3 explains).
4. **Plan shots** — open LLM produces the `Storyboard` per Section 4 rules.
5. **Generate assets** — per member: character renders via image-diffusion + that member's LoRA;
   designed synthetic voice prepared. (LoRA training is a one-time setup task per cast member;
   treat missing LoRA as a blocking setup step, not a per-job step.)
6. **Generate video (per shot)** — Wan 2.7 adapter, image-to-video from the on-screen members'
   renders + the shot action. Then lip-sync adapter aligns mouths to that shot's line.
7. **Synthesize voices** — per-speaker TTS; mix optional gentle background music.
8. **Assemble** — `ffmpeg` concatenates shots, lays audio + burned-in captions, exports 9:16.
9. **Output (manual publish)** — write the final file + a metadata sidecar (learning objective,
   rights record, disclosure + made-for-kids flags) to the review folder. No auto-publish in Phase 1.

Routing is still hardcoded in Phase 1 (one adapter per capability). The registry/router refactor
is Phase 3 — but write adapters behind the capability interfaces now so that refactor is clean.

---

## 6. Guardrails enforced in Phase 1 (children's-content specifics)

These extend the master-plan guardrails and are enforced as checks, not advice:

1. **Original characters.** Reject the cast if `is_original_synthetic != true`. Design review
   note in the job: characters must not resemble any existing kids'-show cast (silhouette, palette,
   signature features). Pigs-in-general are fine; a specific show's look is not.
2. **Transformative sourcing.** Default `script_mode = transformed`. Educational reference videos
   are usually copyrighted; the pipeline extracts the *concept/structure*, not the script/assets.
   `rights_cleared` must be true to pass the script stage.
3. **Made-for-kids designation.** When `made_for_kids: true`, the output metadata sets the
   platform "made for kids" flag; the publish step (even manual) surfaces it.
4. **COPPA / kids'-data posture.** No collection of personal data from child viewers; the learning
   loop (later phase) uses only aggregate platform metrics, never child-level data. Document this.
5. **Age-appropriateness check.** Even before the full Phase 2 critic, run a lightweight content
   check on the final script + video: no frightening imagery, no unsafe behavior shown without
   context, simple language, positive framing. A failure routes to human review, never to output.
6. **AI disclosure.** Disclosure label flag set on the output metadata (default true).

---

## 7. Phase 1 acceptance criteria

- A real educational-kids reference link produces a watchable ~30–60s 9:16 short starring the
  4-member original cast, teaching the same concept via an original transformed script.
- Each character is visually consistent across its shots and has a distinct synthetic voice.
- Dialogue is correctly attributed and lip-synced per shot; captions are present and readable.
- The output sidecar contains: learning objective, rights record, made-for-kids flag, disclosure flag.
- All Section 6 guardrails are enforced: a job with a non-original cast, missing rights, or a failed
  age-appropriateness check is blocked and routed to review — never written as final output.
- Swapping `species: pig` → another value in the cast config changes the characters with **no code
  change** (LoRAs aside). Swapping the channel profile changes genre with no code change.

---

## 8. Extensibility checklist (proving "changeable over time")

The agent must verify these without touching pipeline code:

- **New genre** → add a new `config/channels/<genre>.yaml` (e.g. `bedtime_stories`, `early_math`).
  The script-adaptation prompt template reads `genre_content` + `pedagogy`/profile fields; no genre
  logic is hardcoded.
- **New species / look** → change `species` and `visual_descriptor`s in the cast config; train new
  LoRAs for the new look. Pipeline unchanged.
- **Different cast size** → add/remove `members`; shot-planner and script-adapter read the member
  list dynamically (do not assume exactly 4).
- **Different language/length/aspect** → all from the channel profile.

If any of these requires editing pipeline code, the abstraction is wrong — fix the abstraction.

---

## 9. Golden-path integration test

Fixture: one short educational-kids reference link (concept e.g. "counting to five").
Assert: transcript produced → `LearningObjective` extracted → multi-speaker transformed `Script`
with lines assigned across all 4 members → `Storyboard` with ≤2 characters per shot → per-shot
clips generated and lip-synced → 4 distinct voices → assembled 9:16 captioned file → sidecar with
all required flags → all guardrail checks pass. Mock the GPU models in unit tests; run the real
models in the GPU integration suite on rented capacity.

---

## 10. Still need from the operator

1. **Workflow engine** for the build (Temporal vs Prefect/Dagster vs custom-for-MVP).
2. **Target platform(s)** for output (sets the publish adapter + the made-for-kids/disclosure mechanism).
3. **Cast identities** — names and original visual designs for the 4 members (fill the cast config).
4. **Learning reward** (for the later learning phase) — what "best" means for a kids' educational channel
   (watch-through? completion of the concept? returning viewers?).
5. **Source policy** — which reference channels/links are acceptable inputs, given platform ToS.

---

*End of Phase 1 spec. Build behind the capability interfaces, keep genre and cast in config,
use shot-based generation for the multi-character scenes, and enforce every kids'-content guardrail.*
