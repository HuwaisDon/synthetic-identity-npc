
"""
Memory node and edge schemas.

Designed for extensibility — fields marked FUTURE are unused now
but define the shape of the narrative identity layer we're building toward.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class EdgeType(str, Enum):
    """
    Typed edges are what separates a memory graph from a lore database.
    Each type carries different spreading activation weights.
    """
    REMINDS_OF         = "reminds_of"          # semantic/thematic link
    TRIGGERS_FEAR      = "triggers_fear"        # activates fear response
    UNRESOLVED_CONFLICT = "unresolved_conflict" # psychologically open wound
    ASSOCIATED_SENSORY = "associated_sensory"   # smell/sound/touch link
    CAUSED_BY          = "caused_by"            # causal chain
    LEADS_TO           = "leads_to"             # consequence link
    CONTRADICTS        = "contradicts"          # FUTURE: narrative conflict
    REFRAMES           = "reframes"             # FUTURE: later event changes meaning of earlier
    IDENTITY_ANCHOR    = "identity_anchor"      # FUTURE: load-bearing self-narrative piece


class EventType(str, Enum):
    TRAUMA      = "trauma"
    FORMATIVE   = "formative"       # shaped personality without being traumatic
    RELATIONAL  = "relational"      # about a person
    ACHIEVEMENT = "achievement"
    LOSS        = "loss"
    BETRAYAL    = "betrayal"
    DISCOVERY   = "discovery"
    SHAME       = "shame"
    PRIDE       = "pride"
    MUNDANE     = "mundane"         # low weight, background texture


@dataclass
class PsychologicalEffect:
    trait: str          # e.g. "fear_of_deep_water"
    magnitude: float    # 0.0 - 1.0


@dataclass
class MemoryNode:
    """
    A single autobiographical memory.

    The gap between `objective_description` and `self_narrative_description`
    is where character identity lives. Initially they match. Over time,
    distortion, suppression, and reframing pull them apart.
    """
    # --- Core identity ---
    memory_id: str
    npc_id: str
    event_type: EventType

    # --- Temporal anchoring ---
    age_at_event: int
    in_world_year: Optional[str] = None
    season: Optional[str] = None

    # --- The event itself ---
    objective_description: str = ""     # what actually happened
    self_narrative_description: str = ""  # what the NPC believes/tells themselves
    # NOTE: these start identical; diverge via distortion + reframing over time

    # --- Emotional structure ---
    emotional_weight: float = 5.0       # 0.0 (trivial) - 10.0 (defining)
    valence: float = 0.0                # -1.0 (negative) to +1.0 (positive)
    suppression_level: float = 0.0      # 0.0 (open) to 1.0 (deeply repressed)

    # --- Sensory anchors (drive unexpected activation) ---
    sensory_tags: list[str] = field(default_factory=list)
    # e.g. ["cold_water", "screaming", "salt", "rope", "darkness"]

    # --- Social context ---
    people_involved: list[str] = field(default_factory=list)
    location: str = ""

    # --- Epistemics ---
    confidence_score: float = 1.0       # how clearly Morgan remembers this
    distortion_score: float = 0.0       # how much recall has drifted from truth
    contradiction_flag: bool = False    # FUTURE: conflicts with another memory

    # --- Decay mechanics ---
    decay_rate: float = 0.01            # per-session weakening (trauma = low rate)
    current_strength: float = 1.0      # degrades over time, restored by activation
    activation_count: int = 0          # how many times retrieved
    last_activated_turn: Optional[int] = None

    # --- Behavioral outputs ---
    psychological_effects: list[PsychologicalEffect] = field(default_factory=list)

    # --- Concept associations (for spreading activation) ---
    associated_concepts: list[str] = field(default_factory=list)

    # --- FUTURE: Narrative identity fields ---
    # repression_depth: float = 0.0     # how deeply buried (affects retrieval ceiling)
    # narrative_role: str = ""          # "origin_wound" | "pride_source" | "cautionary_tale"
    # identity_load: float = 0.0        # how much self-concept depends on this memory
    # last_reframed_turn: int = None    # when NPC last updated their interpretation

    # --- Embedding (populated at load time) ---
    embedding: Optional[list[float]] = None


@dataclass
class MemoryEdge:
    """
    A typed, weighted link between two memories.
    Edge weight controls how much activation spreads across it.
    """
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 0.5     # 0.0 (weak association) to 1.0 (strong link)
    bidirectional: bool = False
    notes: str = ""
