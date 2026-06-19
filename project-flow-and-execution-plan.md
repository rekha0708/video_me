# Project Flow & Execution Plan — Synthetic Kids' Educational Channel

> **Role of this document:** the top-level map. It sequences everything from zero to a running
> channel and ties the other two docs together:
> - `orchestration-build-plan.md` — the architecture and full phased roadmap (Phases 0–5).
> - `phase-1-spec-kids-educational.md` — the detailed spec for the first product instance.
>
> This document adds what those don't cover: prerequisites, parallel workstreams, decision gates,
> the content/editorial layer, risks, cost/effort framing, and the honest list of what is still open.

---

## 1. End-to-end project lifecycle

```
SETUP  →  BUILD (Phases 0–5)  →  CONTENT OPS  →  LIVE CHANNEL
```
- **Setup** — accounts, infrastructure, rented GPUs, the cast/voice decisions, the content strategy.
- **Build** — the software pipeline, in the phase order from the master plan.
- **Content Ops** — the non-software layer: what to teach, in what order, and how each video is
  reviewed before it goes out.
- **Live** — posting on a cadence within guardrails, with metrics feeding the learning loop.

The whole point of this sequencing is to reach the **"test the waters" milestone** (Section 6) on
*rented* compute, before any decision about buying hardware.

---

## 2. Workstreams (these run in parallel, not strictly in series)

The build is not one queue. Five tracks run alongside each other; only some gate each other.

- **Track A — Framework & pipeline build.** The orchestration code, Phases 0→5. This is the spine.
- **Track B — Cast & voice.** Original character design → per-character LoRA training → designed
  synthetic child voices. *Currently deferred by you (character look). This track can lag without
  blocking framework work* — Phase 1 code can be built and tested with placeholder renders/voices,
  and the real cast dropped in before the "test the waters" milestone.
- **Track C — Content / curriculum.** What concepts the channel teaches, in what order, and the
  source-link policy. Needed before producing real videos, not before writing code.
- **Track D — Infrastructure & accounts.** GPU rental, storage, DB, the social/platform account
  and its "made for kids" setup.
- **Track E — Compliance.** Rights/sourcing policy, disclosure, COPPA posture, age-appropriateness
  rubric. Mostly already specified as guardrails; needs operator sign-off, not new code.

Dependency summary: A is the critical path. B and C can develop in parallel and only need to
*converge with A* at the point of producing the first real video. D underpins everything and should
start immediately. E is sign-off, done early and revisited each phase.

---

## 3. Phase timeline with decision gates

| Phase | Track | Gated by (must be decided first) | Exit criteria |
|-------|-------|----------------------------------|---------------|
| Setup | D, E | GPU provider; target platform; source policy | accounts + rented GPU + storage/DB live; compliance signed off |
| 0 Skeleton | A | workflow engine choice | empty DAG runs a no-op job, recorded + logged |
| 1 MVP | A (+B/C converge) | cast identities *(or placeholders)*; one starter concept | a real link → watchable original short (Phase 1 acceptance) |
| 2 Critic | A | age-appropriateness rubric finalized | failing output auto-regenerated; only passes proceed |
| 3 Framework + healing | A | — | hot-swap an adapter via config; fallback on failure |
| 4 Learning | A | learning-reward definition | choices shift toward higher-reward variants |
| 5 Automation/scale | A, C, D | posting cadence; content calendar | unattended cadence within guardrails + kill switch |

**"Test the waters" milestone = end of Phase 2 with the real cast in place.** That is the earliest
point you can honestly judge output quality and decide on hardware. Phases 3–5 are
productionization, not proof.

---

## 4. Prerequisites & setup checklist (do before/at Phase 0)

- [ ] Rented GPU account (per the hardware discussion — start on cloud, not owned hardware).
- [ ] S3-compatible object storage + PostgreSQL provisioned.
- [ ] Repo, containerization, CI baseline.
- [ ] Social/platform account created and set to "made for kids".
- [ ] Royalty-free / generated music source decided (for kids' background audio).
- [ ] Asset & version store for LoRAs, voices, and outputs (characters *will* change over time, so
      version them from day one).
- [ ] Operator sign-off on the compliance posture (Track E).

---

## 5. Open decisions log (this is the "anything else?" answer)

Beyond the character look you're deferring, these are genuinely still open. Each shows when it is
needed and a safe default if you don't decide in time.

| # | Decision | Needed by | Safe default if undecided |
|---|----------|-----------|---------------------------|
| 1 | Workflow engine (Temporal / Prefect / Dagster / custom) | Phase 0 | Prefect for MVP, revisit at Phase 3 |
| 2 | Target platform(s) | Setup | one platform first; design publish adapter generically |
| 3 | Cast identities (names, looks) | converge by Phase 1 | placeholders; real cast before "test the waters" |
| 4 | Synthetic child voices design | with cast | neutral designed voices, refine later |
| 5 | Curriculum / what to teach + order | before first real video | start with one concept, expand into a calendar |
| 6 | Source-link policy (which references are OK) | Setup | own/licensed/public-domain only until policy set |
| 7 | Learning reward (what "best" means) | Phase 4 | watch-through + concept completion |
| 8 | Music strategy (licensed vs generated) | Phase 1 | royalty-free library to start |
| 9 | Human review workflow for kids' content | Phase 1 | mandatory manual approval before every publish |
| 10 | Build budget ceiling on rented GPU | Setup | cap monthly spend; iterate in batches |

---

## 6. Content operations (the non-software layer most plans forget)

A working pipeline is not a channel. You also need:

- **Curriculum plan.** A list of concepts appropriate for the 3–6 age range, sequenced (e.g. counting,
  colors, shapes, simple words), one concept per video. This drives what links you source and what
  the channel actually teaches.
- **Content calendar.** Which concept posts when, mapped to the posting cadence.
- **Human review gate.** For children's content, every video gets human approval before publishing —
  even after the Phase 2 critic. This is non-negotiable for a kids' channel and is wired in as the
  manual publish step that stays manual longer than for other genres.
- **Iteration policy.** How many regenerations/variants per video before a human steps in.

---

## 7. Risk register

| Risk | Type | Mitigation (already in the plan) |
|------|------|----------------------------------|
| Characters resemble an existing kids' show | Legal/IP | Original-design hard requirement; design review in guardrails |
| Re-performing copyrighted source too closely | Legal/IP | Default `transformed` mode; rights checkpoint before script stage |
| Inappropriate content reaching kids | Safety | Age-appropriateness gate + mandatory human review |
| Platform "made for kids" / COPPA non-compliance | Regulatory | Made-for-kids flag + aggregate-only metrics, no child data |
| Missing AI disclosure | Policy | Disclosure label enforced at publish |
| Multi-character inconsistency | Technical | Shot-based generation (Phase 1 spec, Section 4) |
| Model becomes obsolete | Technical | Adapter/registry pattern — swap via config |
| Runaway cloud GPU spend | Cost | Batch iteration; monthly cap; rent-not-buy until milestone |
| Over-reliance on one external platform | Strategic | Generic publish adapter; platform is config |

---

## 8. Build cost & effort framing (rough, not a quote)

- **Compute during build is intermittent and modest** — generation runs in bursts, so rented-GPU
  cost is driven by *iteration volume*, not uptime. Cap a monthly ceiling and work in batches.
- **The expensive resource is iteration, not infrastructure.** Most spend goes to regenerating until
  quality is acceptable; the critic loop (Phase 2) is what brings that cost down.
- **One-time setup costs:** LoRA training per character (short GPU jobs), voice design, account setup.
- **Don't buy hardware until the "test the waters" milestone proves the output is good enough.** That
  decision is deferred by design.

(Use the earlier hardware comparison for per-hour rental figures; treat any total here as a planning
placeholder, not a fixed number.)

---

## 9. What is NOT yet specced (honest gaps + suggested order to close)

In rough priority:
1. **Detailed specs for Phases 2–5** (only Phase 1 is detailed so far). Write each as its own doc
   when you reach it, in the same agent-followable style.
2. **Curriculum / content plan** (Track C) — the editorial backbone; needed before real videos.
3. **Cast & voice design + LoRA training guide** (Track B) — deferred by you; the LoRA setup guide
   can be written now so it's ready when you finalize the look.
4. **Infra/accounts setup runbook** (Track D) — concrete provisioning steps.
5. **Review-workflow definition** (Section 6) — exactly how a human approves each kids' video.

Everything else (architecture, contracts, Phase 1, guardrails, risk posture, sequencing) is in place.

---

## 10. Immediate next actions

1. Start Track D (accounts + rented GPU + storage/DB) — unblocks everything, no decisions pending.
2. Decide #1 (workflow engine) and #2 (platform) — small, unblocks Phase 0.
3. Begin Track A Phase 0 against the master plan.
4. In parallel, draft the curriculum (Track C) and — when ready — the cast design (Track B).
5. Build to the **"test the waters" milestone** (end of Phase 2, real cast), then judge quality and
   revisit the hardware decision.

---

*End of execution plan. Critical path is the framework (Track A); the cast can lag behind it; reach
the test-the-waters milestone on rented compute before committing to hardware.*
