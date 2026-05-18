"""
schemas/cognitive_schemas.py

Typed data contracts for the entire Synthetic Identity NPC cognitive pipeline.

Every system communicates through these schemas.
This is intentional: it enforces the separation between cognition layers
and prevents any one engine from bleeding into another's domain.

Psychological rationale:
  These schemas model the *outputs* of psychological processes, not the processes
  themselves. A GoalState is what the goal system exposes to the pipeline —
  not the full internal scoring machinery.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import time


# ─────────────────────────────────────────────
# ENUMERATIONS
# ─────────────────────────────────────────────

class EmotionType(Enum):
    FEAR       = "fear"
    GRIEF      = "grief"
    SHAME      = "shame"
    GUILT      = "guilt"
    ANGER      = "anger"
    LONGING    = "longing"
    RELIEF     = "relief"
    JOY        = "joy"
    CONTEMPT   = "contempt"
    DREAD      = "dread"
    NUMBNESS   = "numbness"

class GoalType(Enum):
    SURVIVAL        = "survival"       # physical safety
    TRUST_GAIN      = "trust_gain"     # build rapport
    TRUST_PROTECT   = "trust_protect"  # do not lose existing trust
    INFORMATION     = "information"    # extract/hoard knowledge
    RESOURCE        = "resource"       # coin, objects, leverage
    FREEDOM         = "freedom"        # autonomy, escape
    REPUTATION      = "reputation"     # social standing
    REVENGE         = "revenge"        # delayed harm
    IDENTITY        = "identity"       # self-concept preservation
    CONNECTION      = "connection"     # intimacy, belonging
    CONCEALMENT     = "concealment"    # hide something specific

class AttentionalState(Enum):
    OPEN      = "open"       # broad, curious, exploratory
    FOCUSED   = "focused"    # narrowed to salient topic
    STRESSED  = "stressed"   # stress-narrowed, threat-biased
    FLOODED   = "flooded"    # overwhelmed, incoherent
    DISSOCIATED = "dissociated"  # detached, flat


# ─────────────────────────────────────────────
# EMOTIONAL STATE
# ─────────────────────────────────────────────

@dataclass
class EmotionalReading:
    """
    A single emotion with intensity and source.
    Intensity is 0.0–1.0. Source links to a memory node ID or event.
    """
    emotion: EmotionType
    intensity: float           # 0.0 – 1.0
    source_node: Optional[str] = None   # memory graph node ID
    timestamp: float = field(default_factory=time.time)
    is_suppressed: bool = False

@dataclass
class EmotionalState:
    """
    The full emotional profile at a given moment.
    Dominant emotion drives attentional and behavioral biasing.
    Residue carries lingering activations from prior turns.
    """
    readings: list[EmotionalReading] = field(default_factory=list)
    dominant: Optional[EmotionalReading] = None
    valence: float = 0.0          # -1.0 (negative) to +1.0 (positive)
    arousal: float = 0.0          # 0.0 (calm) to 1.0 (agitated)
    residue: list[EmotionalReading] = field(default_factory=list)  # lingering from past
    turn_delta: float = 0.0       # net emotional shift this turn


# ─────────────────────────────────────────────
# GOAL ENGINE
# ─────────────────────────────────────────────

@dataclass
class Goal:
    """
    A single motivational objective.
    
    Psychological rationale:
      Goals are not instructions. They are internal drives that compete
      for behavioral priority. Morgan doesn't 'want to hide the murder'
      as a rule — she has identity-preservation goals that make disclosure
      feel dangerous. These are different things.
    """
    goal_id: str
    goal_type: GoalType
    description: str
    utility: float              # current expected value: -1.0 to +1.0
    urgency: float              # 0.0–1.0; how time-sensitive
    active: bool = True
    blocking_topics: list[str] = field(default_factory=list)   # memory node IDs to suppress
    linked_memory_nodes: list[str] = field(default_factory=list)

@dataclass
class GoalState:
    """
    Output of GoalEngine for a given turn.
    Arbitrated result of competing motivations.
    """
    active_goals: list[Goal] = field(default_factory=list)
    dominant_goal: Optional[Goal] = None
    strategic_silence: list[str] = field(default_factory=list)  # node IDs to avoid
    disclosure_pressure: float = 0.0   # pull toward revealing something
    concealment_pressure: float = 0.0  # pull toward hiding something
    manipulation_intent: float = 0.0   # 0.0 = honest; 1.0 = full manipulation
    goal_conflict_score: float = 0.0   # how much goals are fighting each other


# ─────────────────────────────────────────────
# ATTENTION ENGINE
# ─────────────────────────────────────────────

@dataclass
class AttentionState:
    """
    Attentional bandwidth and focus.
    
    Psychological rationale:
      The mind cannot hold everything at once. Fear narrows attention
      to threat-relevant content. Dissociation flattens it.
      Attention determines which activated memories become 'visible'
      to cognition vs. remaining subconscious pressure.
    """
    state: AttentionalState = AttentionalState.OPEN
    bandwidth: float = 1.0          # 1.0 = full; reduced under stress/load
    focal_nodes: list[str] = field(default_factory=list)     # currently dominant memory nodes
    suppressed_from_attention: list[str] = field(default_factory=list)  # activated but not attended
    salience_bias: Optional[EmotionType] = None   # which emotion is steering attention
    cognitive_load: float = 0.0     # 0.0–1.0; how taxed is working cognition


# ─────────────────────────────────────────────
# TEMPORAL EMOTIONAL PERSISTENCE
# ─────────────────────────────────────────────

@dataclass
class EmotionalResidueEntry:
    """
    A lingering emotional activation from a past interaction.
    Decays over turns.
    """
    emotion: EmotionType
    intensity: float
    source_description: str       # human-readable (for debugging / summarization)
    source_node: Optional[str]
    turns_remaining: int          # how many turns until fully decayed
    decay_rate: float = 0.15      # per-turn decay

@dataclass
class EmotionalPersistenceState:
    """
    Tracks mood carryover and rumination across turns.
    """
    residue: list[EmotionalResidueEntry] = field(default_factory=list)
    mood_baseline_shift: float = 0.0   # long-term drift from repeated emotional events
    rumination_node: Optional[str] = None   # node being obsessively re-activated
    rumination_intensity: float = 0.0


# ─────────────────────────────────────────────
# SELF-CONCEPT DEFENSE
# ─────────────────────────────────────────────

@dataclass
class SelfConceptThreat:
    """
    A detected threat to the NPC's self-concept.
    
    Psychological rationale:
      Humans don't process threats to self-image neutrally.
      They activate defensive maneuvers: reframing, blame externalization,
      mythologization. This schema captures what triggered defense
      and what mechanism was deployed.
    """
    threatening_node: str           # memory node that threatens self-concept
    threat_type: str                # "shame", "helplessness", "moral_failure", "betrayal"
    threat_intensity: float
    defense_mechanism: str          # "reframe", "externalize_blame", "minimize", "mythologize", "deny"
    reframe_narrative: Optional[str] = None   # the altered self-narrative

@dataclass
class SelfConceptState:
    """
    Current integrity of the NPC's self-concept and active defenses.
    """
    coherence: float = 1.0            # 1.0 = intact; 0.0 = fragmenting
    active_threats: list[SelfConceptThreat] = field(default_factory=list)
    active_defenses: list[str] = field(default_factory=list)   # mechanism labels
    core_identity_claims: list[str] = field(default_factory=list)  # "I am loyal", "I am a survivor"
    threatened_claims: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# PREDICTIVE SIMULATION
# ─────────────────────────────────────────────

@dataclass
class SimulatedOutcome:
    """
    A single simulated consequence of a potential disclosure.
    """
    disclosure_content: str         # what would be said
    predicted_trust_delta: float    # -1.0 to +1.0
    predicted_emotional_cost: float # 0.0 to 1.0 (how much it hurts to say it)
    predicted_danger: float         # 0.0 to 1.0 (risk of harm to NPC)
    predicted_goal_alignment: float # -1.0 to +1.0 (helps or hurts active goals)
    net_utility: float = 0.0        # computed by simulation engine

@dataclass
class PredictiveState:
    """
    Internal simulation results for the current turn.
    """
    simulated_outcomes: list[SimulatedOutcome] = field(default_factory=list)
    chosen_strategy: str = "neutral"   # "disclose", "deflect", "lie", "silence", "partial"
    anticipated_emotional_state: Optional[EmotionalState] = None
    silence_justified: bool = False
    preemptive_deflection_topics: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# COGNITIVE SUMMARY (LLM INPUT LAYER)
# ─────────────────────────────────────────────

@dataclass
class CognitiveSummary:
    """
    The compressed cognitive state passed to the LLM.
    
    This is NOT a dump of raw internal state.
    It is the behavioral *pressure* that should shape language.
    
    Design principle:
      The LLM receives signals, not explanations.
      It should feel the weight of suppression without being
      told 'there is suppression.'
    """
    # Surface emotional signal
    surface_emotion: Optional[str] = None          # what the NPC is 'showing'
    undercurrent_emotion: Optional[str] = None     # what's bleeding through
    emotional_stability: float = 1.0               # 1.0 = composed; 0.0 = fracturing

    # Behavioral pressures
    avoidance_topics: list[str] = field(default_factory=list)   # topics to steer from
    disclosure_pressure_topics: list[str] = field(default_factory=list)  # pulling to reveal
    strategic_intent: str = "neutral"              # "deflect", "probe", "trust_build", etc.

    # Memory access
    accessible_memories: list[str] = field(default_factory=list)   # node IDs available
    suppressed_pressure_nodes: list[str] = field(default_factory=list)  # activated but blocked

    # Identity signals
    self_concept_under_threat: bool = False
    active_defense_mechanism: Optional[str] = None
    identity_claim_being_defended: Optional[str] = None

    # Attentional state
    attentional_state: str = "open"
    cognitive_load_descriptor: str = "clear"   # "clear", "strained", "overwhelmed"

    # Behavioral output guidance
    response_style: str = "direct"             # "guarded", "deflecting", "fragmented", "flat"
    leakage_signals: list[str] = field(default_factory=list)   # behavioral tells


# ─────────────────────────────────────────────
# FULL NPC COGNITIVE STATE (MASTER CONTAINER)
# ─────────────────────────────────────────────

@dataclass
class NPCCognitiveState:
    """
    Complete cognitive snapshot for a single NPC at a point in time.
    This is what gets persisted and restored across sessions.
    """
    npc_id: str
    turn: int = 0
    timestamp: float = field(default_factory=time.time)

    emotional_state: EmotionalState = field(default_factory=EmotionalState)
    goal_state: GoalState = field(default_factory=GoalState)
    attention_state: AttentionState = field(default_factory=AttentionState)
    persistence_state: EmotionalPersistenceState = field(default_factory=EmotionalPersistenceState)
    self_concept_state: SelfConceptState = field(default_factory=SelfConceptState)
    predictive_state: PredictiveState = field(default_factory=PredictiveState)
    cognitive_summary: CognitiveSummary = field(default_factory=CognitiveSummary)
