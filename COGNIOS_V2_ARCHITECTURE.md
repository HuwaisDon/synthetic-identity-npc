# CogniOS — Architecture & Roadmap Specification (v2.1)

**Status:** Frozen design — source of truth for implementation
**Supersedes:** the original `synthetic-identity-npc` repo's implicit Morgan-only architecture
**Scope:** this document governs *what gets built, in what order, and how we know each phase is done*. It intentionally does not contain implementation code — each phase should be implemented against this spec, and checked against its Definition of Done before the next phase starts.
**Changelog v2.1:** added CogniRuntime facade (§7), Intention/Expression split into the core data flow (§3, §6), Simulation Mode scoped to scripted-transcript only (§8 Phase 2, autonomous-player mode moved to §9).

---

## 1. Vision

Existing LLM-driven characters use the model as the mind:

```
Input (text/vision/audio) → LLM → Output
```

CogniOS inverts this. The LLM is the **voice**, not the **mind**:

```
Input → Perception → Cognitive Events → Memory → Emotion → Suppression
      → Attention → Goals → Identity → Prediction → Intention → Expression → LLM → Output
```

A CogniOS character has a persistent internal psychological state — memory, emotional residue, goals, identity, suppression — that exists and evolves independently of any single LLM call. The LLM's only job is to render the *current state* of that mind into natural language (or, later, speech).

**Design principles, stated once and enforced everywhere:**
- *Perception never talks directly to the LLM.* Every external stimulus (text, image, audio, OCR) becomes a `CognitiveEvent` and enters the pipeline at the same point a player message does. There is no shortcut from raw input to prompt.
- *Cognition produces an Intention before it produces language.* The pipeline's terminal cognitive output is not "a prompt" — it's a structured `Intention` (what the character wants to do, and how strongly). Today the only supported expression of an Intention is dialogue; the object itself does not assume that.
- *External systems talk to CogniOS through one interface.* Games, chat surfaces, and robotics integrations call `CogniRuntime`, never the engines directly.

---

## 2. Where we are today (baseline)

The current repo (`synthetic-identity-npc`) implements Layer 2 (Cognitive Runtime) correctly, but only for one hardcoded character, "Morgan":

- `core/cognition_pipeline.py` — `CognitionPipeline.process_turn()` runs the full 13-step turn loop.
- `schemas/cognitive_schemas.py` — the typed contract every engine communicates through. `PredictiveState.chosen_strategy` + `GoalState` together already function as a proto-Intention (see §6).
- `memory/` — ChromaDB semantic retrieval (`memory_store.py`) → emotional activation (`activation.py`) → NetworkX spreading (`spreading.py`).
- `engines/` — suppression, attention, goals, emotional persistence, self-concept defense, predictive simulation, cognitive summarization. `engines/regulation_engine.py` exists as an empty file; `RegulationState` is defined in schema but never populated.
- `llm/` — `GeminiClient` (active), `OpenRouterClient` (unused alternative), `BehavioralPromptBuilder`, `ResponseValidator` (log-only).
- `persistence/npc_state_store.py` — JSON file per NPC, `STATE_VERSION` defined but never checked on load.
- `api/routes.py` — thin FastAPI layer, in-process pipeline cache (`_pipeline_registry`, `get_pipeline()`) — this is the informal precursor to `CogniRuntime` (§7).
- `core/event_bus.py` — `CognitiveEventBus` fully implemented (event types, injection logic, three example event factories) but **never invoked from `api/routes.py`** — built, not wired.

Everything Morgan-specific (`build_morgan_goal_profile()`, `MORGAN_IDENTITY_CLAIMS`, `THREAT_MAP`, `MORGAN_CHARACTER_BRIEF`, `MORGAN_MEMORIES`) is Python constants, not data — and critically, `THREAT_MAP`'s matching logic (`if pattern in node_id or node_id in pattern`) is *procedural psychology*, not just character data. This distinction drives Phase 1 below.

---

## 3. Target Architecture

```
                              CogniOS
              (Cognitive Runtime for AI Characters)

  ┌──────────────────────┐        ┌──────────────────────┐
  │   Character DSL (YAML)│       │  Character Creator    │
  │   human-authored       │──┐   │  (LLM, creative)       │
  └──────────────────────┘  │   └───────────┬────────────┘
                             │               │
                             └───────┬───────┘
                                     ▼
                        Psychological Profile
                     (identity, values, fears, trauma,
                      secrets, relationships, threat/defense rules —
                      one schema, two possible producers)
                                     │
                                     ▼
                        ┌────────────────────────┐
                        │   Character Compiler     │
                        │   (deterministic, NO LLM)│
                        │   validates referential   │
                        │   integrity, derives       │
                        │   MemoryNodes/Goals/       │
                        │   MemoryEdges              │
                        └────────────┬─────────────┘
                                     │
                        Character Runtime Objects
                        (MemoryNode[], Goal[], MemoryEdge[],
                         ThreatRule[], CharacterSchema)
                                     │
                                     ▼
                ┌────────────────────────────────────────┐
                │         CogniRuntime (facade)            │
                │  load_character · perceive · think        │
                │  get_state · save · reset · shutdown       │
                │  — the ONLY interface external callers use │
                └───────────────────┬──────────────────────┘
                                     ▼
                     ┌───────────────────────────────┐
                     │       Cognitive Runtime         │
                     │        (private engines)         │
                     │  Memory · Activation · Spreading │
                     │  Suppression · Attention          │
                     │  Goals · Identity (generic)       │
                     │  Prediction · Regulation          │
                     └───────────────┬─────────────────┘
                                     │
                                Intention
                       (what the character wants to do,
                        and how strongly — modality-agnostic)
                                     │
                                     ▼
                             Expression Layer
                        (Cognitive Summary — today's only
                         supported modality: dialogue)
                                     │
                                     ▼
                                   LLM
                                     │
                                     ▼
                          Natural Language / Speech

  ┌─────────────────────────────────────────────────────┐
  │                    Perception Layer                   │
  │   Vision · Speech-to-Text · OCR                        │
  │   → always emits CognitiveEvent, NEVER calls the LLM   │
  │   → always enters through CogniRuntime.perceive()       │
  └───────────────────────────┬───────────────────────────┘
                              │
                              ▼
                     core/event_bus.py
              (CognitiveEventBus — already implemented,
               needs to be wired into CogniRuntime.perceive())
```

**Invariants, non-negotiable:**
1. No arrow skips from Perception directly to LLM.
2. No external caller reaches an engine except through `CogniRuntime`.
3. Cognition always produces an `Intention` object before any LLM call — even though today the only consumer of that object is the dialogue Expression Layer.

---

## 4. Design Principles

1. **Extend, don't replace.** `schemas/cognitive_schemas.py` and the engine boundary contract stay intact. New capability is added by generalizing inputs (character data) and completing gaps (regulation, decay), not by rewriting engines.
2. **Don't invent new fields when an existing one already does the job.** E.g. generic threat-rule matching should use `MemoryNode.associated_concepts`/`sensory_tags` (already defined in `schemas/memory_schema.py`), not a new embedding-based matcher. Same principle applied to Intention: don't invent it from scratch, formalize `PredictiveState` + `GoalState`, which already carry most of what it needs.
3. **Generator is creative, Compiler is deterministic.** The Character Compiler never calls an LLM. It only validates structure and derives runtime objects. Semantic plausibility checking happens inside the Generator stage, never inside the Compiler.
4. **Perception never talks to the LLM; external callers never talk to engines.** Both are enforced by routing everything through `CogniRuntime`.
5. **Definition of done is falsifiable.** Every phase below ends in a runnable check, not a description.
6. **Sparse by default.** Character generation produces load-bearing memories by default, not a full simulated biography.
7. **Formalize an abstraction when it's nearly free, not when it's needed.** The Intention object is added during Phase 1 (while schemas are already being touched) rather than retrofitted later once Perception/Action code depends on the old shape.

---

## 5. Folder Structure (target)

```
cognios/
├── main.py
├── runtime/
│   └── cogni_runtime.py            # NEW — CogniRuntime facade
├── api/
│   └── routes.py                    # thin — delegates to CogniRuntime
├── core/
│   ├── cognition_pipeline.py         # unchanged control flow, generic inputs
│   └── event_bus.py                  # invoked via CogniRuntime.perceive()
├── schemas/
│   ├── cognitive_schemas.py
│   ├── memory_schema.py
│   ├── character_schema.py           # NEW — CharacterSchema, ThreatRule, PsychProfile
│   └── intention_schema.py           # NEW — Intention, ExpressionModality
├── memory/
│   ├── memory_store.py
│   ├── activation.py
│   ├── spreading.py                  # graph built from character data, not hardcoded
│   └── seed_loader.py                 # replaces morgan_memories.py's role, generic
├── engines/
│   ├── suppression_engine.py
│   ├── attention_engine.py
│   ├── goal_engine.py                 # generic goal profile loader
│   ├── emotional_persistence.py       # + decay/reinforcement/reconsolidation
│   ├── self_concept_defense.py        # generic ThreatRule evaluation
│   ├── predictive_simulation.py
│   ├── cognitive_summarizer.py         # consumes Intention, not raw engine state
│   └── regulation_engine.py           # IMPLEMENTED (currently empty)
├── character/
│   ├── loader.py                       # CharacterLoader
│   ├── repository.py                   # CharacterRepository
│   ├── generator.py                    # LLM-driven Psychological Profile generation
│   ├── dsl_parser.py                   # YAML Character DSL → Psychological Profile
│   └── compiler.py                     # deterministic validation + derivation
├── perception/
│   ├── vision.py
│   ├── speech.py
│   ├── ocr.py
│   └── event_builder.py                # perception output → CognitiveEvent
├── simulation/
│   └── scripted_runner.py              # NEW — scripted-transcript simulate()
├── llm/
│   ├── gemini_client.py
│   ├── openrouter_client.py
│   ├── prompt_builder.py                # consumes Expression Layer output
│   └── response_validator.py
├── persistence/
│   └── npc_state_store.py               # + STATE_VERSION migration check
├── characters/
│   └── morgan.yaml                      # Morgan migrated to Character DSL
├── evaluation/
└── testing/
```

---

## 6. Data Contracts (new/changed schemas)

### `schemas/character_schema.py` (new)

```
CharacterSchema
  character_id: str
  name: str
  archetype: str | None
  identity:
    core_beliefs: list[str]
    values: list[str]
    identity_claims: list[str]        # generalizes MORGAN_IDENTITY_CLAIMS
  goals: list[GoalTemplate]           # generalizes build_morgan_goal_profile()
  threat_rules: list[ThreatRule]      # generalizes THREAT_MAP
  reframe_library: dict[str, list[str]]  # generalizes REFRAME_LIBRARY
  memories: list[MemoryNode]          # generalizes MORGAN_MEMORIES
  memory_edges: list[MemoryEdge]
  character_brief: str                # generalizes MORGAN_CHARACTER_BRIEF

ThreatRule                            # generalizes THREAT_MAP tuples
  trigger_concepts: list[str]         # matched against MemoryNode.associated_concepts
  threatened_claim_idx: int
  threat_type: str
  defense_mechanism: str

PsychologicalProfile                  # intermediate — output of Generator/DSL,
                                       # input to Compiler. NOT a runtime object.
  identity: {...}
  values: list[str]
  fears: list[str]
  trauma: list[str]                   # free text, compiled into MemoryNodes
  secrets: list[str]
  relationships: list[str]
  goals: list[str]
```

### `schemas/intention_schema.py` (new)

```
Intention                              # the pipeline's terminal COGNITIVE output —
                                        # produced before any language is generated
  dominant_drive: GoalType             # what the character most wants right now
  disposition: str                     # "disclose" | "partial" | "deflect" | "lie" | "silence"
                                        # (today: sourced directly from PredictiveState.chosen_strategy)
  intensity: float                     # 0.0-1.0, how strongly this intention is held
  target_topics: list[str]             # memory nodes this intention concerns
  supported_modalities: list[str]      # e.g. ["dialogue"] today; future: ["gesture","silence","action"]

ExpressionRequest                       # Expression Layer's input — today, CognitiveSummary IS this
  intention: Intention
  modality: str                        # which supported_modality to render into
  # for modality="dialogue": renders via existing CognitiveSummarizer/BehavioralPromptBuilder path
```

**Rule:** `Intention` is populated from existing engine outputs (`PredictiveState.chosen_strategy` → `disposition`, `GoalState.dominant_goal` → `dominant_drive`, `goal_state.strategic_silence`/`disclosure_pressure_topics` → `target_topics`) — this is a restructuring of data that already exists, not new engine logic. `supported_modalities` defaults to `["dialogue"]` for every character today; this is the single field a future Action Layer would extend.

### Referential integrity rules the Compiler must enforce
- Every `Goal.blocking_topics` / `linked_memory_nodes` entry must correspond to an existing `MemoryNode.memory_id`.
- Every `MemoryEdge.source_id` / `target_id` must correspond to an existing `MemoryNode.memory_id`.
- Every `ThreatRule.threatened_claim_idx` must be a valid index into `identity.identity_claims`.
- Every `ThreatRule.trigger_concepts` entry should intersect at least one `MemoryNode.associated_concepts` (warn, don't hard-fail).
- A `CharacterSchema` that fails hard checks is rejected before it reaches `CharacterLoader`.

---

## 7. CogniRuntime — the external interface

**Why it exists:** every external caller (Unity/Unreal/Godot, Discord, a web frontend, robotics) must interact with CogniOS through one stable surface, never by importing engines directly. This also gives `api/routes.py` a thin, testable dependency instead of owning pipeline lifecycle itself.

```
CogniRuntime
  load_character(character_id: str) -> None      # via CharacterLoader
  perceive(event: CognitiveEvent) -> None          # routes to CognitiveEventBus.emit()
  think(player_input: str, **turn_context) -> TurnOutput   # wraps CognitionPipeline.process_turn()
  get_state(npc_id: str) -> StateResponse          # read-only introspection
  save() -> None                                   # wraps NPCStateStore.save()
  reset() -> None                                  # wraps CognitionPipeline.reset_conversation()
  shutdown() -> None                               # flush + persist on teardown
```

`api/routes.py` becomes a translation layer only: HTTP request → `CogniRuntime` call → HTTP response. It should contain no pipeline-lifecycle logic of its own after this phase (today it has `_pipeline_registry`, `get_pipeline()`, `_build_pipeline()` — these move into `CogniRuntime`).

---

## 8. Module Responsibilities (new modules only — existing modules keep current responsibilities)

| Module | Responsibility | Calls LLM? |
|---|---|---|
| `runtime/cogni_runtime.py` — `CogniRuntime` | single external-facing interface (§7) | No |
| `character/loader.py` — `CharacterLoader` | loads a `CharacterSchema` from JSON/YAML | No |
| `character/repository.py` — `CharacterRepository` | CRUD over stored character definitions | No |
| `character/generator.py` | prompts an LLM to produce a `PsychologicalProfile` from a natural-language description | Yes |
| `character/dsl_parser.py` | parses a Character DSL YAML file into a `PsychologicalProfile` | No |
| `character/compiler.py` | validates a `PsychologicalProfile`, derives runtime objects, enforces referential integrity from §6 | No |
| `perception/vision.py`, `speech.py`, `ocr.py` | wrap external models, return structured detections | model call, not LLM-as-brain |
| `perception/event_builder.py` | converts perception output into `CognitiveEvent` objects | No |
| `simulation/scripted_runner.py` | drives `CogniRuntime.think()` across a pre-written transcript of player turns, for eval/debug/benchmark | No (LLM calls happen inside `think()` as normal — this module adds no new LLM usage) |

---

## 9. Roadmap

### Phase 0 — Finish the existing runtime (prerequisite for everything else)

**Why first:** every later phase either generates data for these engines or generates *characters* that will silently have broken fatigue/decay behavior if this isn't done first.

- Implement `engines/regulation_engine.py` against the existing `RegulationState` schema. Wire a call site into `CognitionPipeline.process_turn()`, feed `regulation_fatigue_level` into `CognitiveSummary`.
- Implement memory decay/reinforcement: mutate `MemoryNode.current_strength`/`activation_count`/`decay_rate` on retrieval and over time.
- Add `NPCStateStore` version check: reject or migrate on `STATE_VERSION` mismatch instead of silently loading.

**Definition of Done:**
- Script: run 5+ turns with repeated high-suppression content on the *existing* Morgan character; assert `suppression_fatigue` increases monotonically and `CognitiveSummary.regulation_fatigue_level` is nonzero by turn 5.
- Script: retrieve the same memory 3x across sessions; assert `current_strength` changed and `activation_count` incremented.
- Script: load a state file with a mismatched `STATE_VERSION`; assert a clear migration path or explicit rejection.

### Phase 1 — Generalize the character system + formalize Intention + CogniRuntime

**Why second:** highest-risk phase — generalizing *logic* (threat matching), not just data. Bundled with Intention and CogniRuntime because both are cheap right now (schemas are already being touched) and expensive later (retrofitting after Perception/Action code depends on the old shapes).

- Define `schemas/character_schema.py` and `schemas/intention_schema.py` per §6.
- Rewrite `engines/self_concept_defense.py`'s threat detection to evaluate generic `ThreatRule.trigger_concepts` against `MemoryNode.associated_concepts`.
- Rewrite `engines/goal_engine.py`'s goal construction to accept a `list[GoalTemplate]` from `CharacterSchema`.
- Insert `Intention` construction as the pipeline's terminal cognitive step, before `CognitiveSummarizer` — `CognitiveSummarizer`/`BehavioralPromptBuilder` now consume `Intention` (modality="dialogue") instead of raw engine outputs directly.
- Build `character/loader.py`, `character/repository.py`.
- Build `runtime/cogni_runtime.py`; migrate `api/routes.py`'s pipeline-lifecycle logic into it.
- Migrate Morgan to `characters/morgan.yaml`, loaded via `CharacterLoader`.

**Definition of Done:**
- Every existing `evaluation/test_*.py` scenario produces behaviorally equivalent output running Morgan-from-YAML vs. Morgan-from-Python-constants (deterministic numbers matched within tolerance).
- A second, trivially different character (zero `threat_rules`) loads and runs a full turn without error.
- `api/routes.py` contains no direct `CognitionPipeline` instantiation — every route calls `CogniRuntime` only.
- `Intention.disposition` for a given turn matches `PredictiveState.chosen_strategy` for that same turn (proves the restructuring preserved existing behavior).

### Phase 2 — Character Generator (Creator + Compiler) + Scripted Simulation Mode

- `character/generator.py`: LLM prompt producing a `PsychologicalProfile` from a one-line description. Default: sparse mode (load-bearing memories only). Full-biography mode: explicit opt-in.
- `character/dsl_parser.py`: YAML Character DSL → same `PsychologicalProfile` type.
- `character/compiler.py`: validates + derives `CharacterSchema` per §6 referential-integrity rules. Never calls an LLM.
- `simulation/scripted_runner.py`: `CogniRuntime.simulate(character_id, transcript: list[str]) -> list[TurnOutput]` — drives a pre-written sequence of player turns through `think()` and returns the full per-turn cognitive trace. **Scope note:** this is scripted-transcript simulation only. An autonomous adversarial-player mode (a second LLM playing the player) is a distinct, larger feature — see §10, not built in this phase.

**Definition of Done:**
- Generate 3 distinct characters from 3 different one-line prompts; all pass Compiler validation with zero dangling-ID errors.
- Deliberately construct one `PsychologicalProfile` with a goal referencing a nonexistent memory; Compiler rejects it with a specific error.
- One character authored via YAML DSL and one via the LLM generator both produce valid `CharacterSchema` objects through the same Compiler path.
- `simulate()` run against a 10-turn scripted transcript (e.g. escalating accusation, mirroring `evaluation/test_suppression_breakdown.py`) returns a full per-turn trace of `Intention`/`suppression_fatigue`/`coherence` without requiring a human to drive turns one at a time.

### Phase 3 — Perception Layer (demo-priority — cheapest phase relative to payoff)

**Why here, not later:** `core/event_bus.py` already implements the relevant event types and handlers. This phase is mostly "replace the hand-authored event-factory trigger with a model-detected trigger."

- `perception/vision.py`: wraps a vision model, returns detected concepts.
- `perception/event_builder.py`: maps detected concepts → `MemoryNode.associated_concepts` matches → constructs `CognitiveEvent` → `CogniRuntime.perceive()`.
- `CogniRuntime.perceive()` wires into `CognitiveEventBus.apply_to_pipeline()` (currently built but never called).
- New API endpoint: `POST /npc/{id}/perceive/image`, delegating to `CogniRuntime.perceive()`.

**Definition of Done:**
- Upload an image containing a concept tagged on an existing character memory; confirm the next `/turn` call reflects elevated fear/dread residue attributable to that event, verifiable via `GET /npc/{id}/state`.
- Confirm no code path in `perception/` imports or calls anything in `llm/`.
- Confirm no code path outside `runtime/cogni_runtime.py` calls `CognitiveEventBus` directly.

### Phase 4 — Speech (Voice)

- `perception/speech.py`: STT wrapper, feeds into `CogniRuntime.think()` or `.perceive()` depending on whether it's dialogue or ambient sound.
- TTS on the output side: extend `Intention`/`CognitiveSummary` fields into delivery parameters (pace, pause, pitch) for a TTS call after `npc_response` is generated.

**Definition of Done:**
- End-to-end voice round trip with at least one audibly different delivery on a high-suppression turn vs. a neutral turn, attributable to fields already computed.

---

## 10. Explicitly Out of Scope (future work, not this roadmap)

- **Relationship Engine** — multi-dimensional per-party relationship objects. Requires generalizing `GoalEngine._apply_trust_context()`.
- **World State** — entities/locations/objects/time/weather queryable by cognition. Depends on Perception (Phase 3).
- **Knowledge Graph Generator** — full people/places/objects/events graph beyond `memory/spreading.py`'s association graph.
- **Action Layer** — consuming `Intention.supported_modalities` beyond `"dialogue"` (gesture/movement/tool-use). The schema groundwork is laid in Phase 1; the actual multi-modality Expression Layer and action-selection logic is deferred. This is now a much smaller lift than originally scoped, because `Intention` already exists by the time this is picked up.
- **Autonomous-player Simulation Mode** — a second LLM playing the adversarial player role across a `simulate()` run, rather than a scripted transcript. Real feature (prompt design, termination conditions, 2x LLM cost); not in Phase 2.

---

## 11. Hackathon Narrative (for reference, not implementation)

> Existing AI agents use the LLM as their brain. CogniOS builds a mind first — memory, emotion, suppression, identity, goals — and uses the LLM only as the voice that expresses it. Perception (vision, speech) never talks to the LLM directly; every sensory input becomes a cognitive event that flows through the same pipeline a line of dialogue does. Before the model ever writes a word, the runtime has already decided what the character wants to do — that decision is a real object, not just a paragraph baked into a prompt.

Demo sequence: (1) show `/npc/{id}/state` mid-conversation as raw numbers; (2) upload an image, show the resulting cognitive event and changed emotional state before any dialogue happens; (3) generate a brand-new character from a one-line prompt live, run it through the same runtime Morgan uses, with zero code changes; (4) run `simulate()` against a 15-turn scripted interrogation and show the suppression-fatigue curve breaking down over time, without a human typing 15 messages live.
