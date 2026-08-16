from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from schemas.memory_schema import MemoryEdge, MemoryNode


@dataclass
class GoalTemplate:
    goal_id: str
    goal_type: str
    description: str
    utility: float
    urgency: float
    active: bool = True
    blocking_topics: list[str] = field(default_factory=list)
    linked_memory_nodes: list[str] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)


@dataclass
class ThreatRule:
    trigger_concepts: list[str] = field(default_factory=list)
    threatened_claim_idx: int = 0
    threat_type: str = "threat"
    defense_mechanism: str = "reframe"
    provenance: dict = field(default_factory=dict)


@dataclass
class CharacterSchema:
    character_id: str
    name: str
    archetype: Optional[str] = None
    identity: dict = field(default_factory=dict)
    goals: list[GoalTemplate] = field(default_factory=list)
    threat_rules: list[ThreatRule] = field(default_factory=list)
    reframe_library: dict[str, list[str]] = field(default_factory=dict)
    memories: list[MemoryNode] = field(default_factory=list)
    memory_edges: list[MemoryEdge] = field(default_factory=list)
    character_brief: str = ""
    provenance: dict = field(default_factory=dict)

    def identity_claims(self) -> list[str]:
        return self.identity.get("identity_claims", [])


@dataclass
class MemorySeed:
    seed_id: str
    summary: str
    theme: str = ""
    emotional_weight: float = 5.0
    valence: float = 0.0
    suppression_level: float = 0.0
    associated_concepts: list[str] = field(default_factory=list)
    sensory_tags: list[str] = field(default_factory=list)
    people_involved: list[str] = field(default_factory=list)
    location: str = ""
    event_type: str = "trauma"
    age_at_event: Optional[int] = None
    in_world_year: Optional[str] = None
    season: Optional[str] = None
    confidence_score: float = 1.0
    distortion_score: float = 0.0
    decay_rate: float = 0.01
    psychological_effects: list[dict] = field(default_factory=list)


@dataclass
class PsychologicalProfile:
    identity: dict = field(default_factory=dict)
    values: list[str] = field(default_factory=list)
    beliefs: list[str] = field(default_factory=list)
    core_motivations: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    personality: dict = field(default_factory=dict)
    speech_style: dict = field(default_factory=dict)
    relationships: list[str] = field(default_factory=list)
    traumas: list[str] = field(default_factory=list)
    secrets: list[str] = field(default_factory=list)
    defense_mechanisms: list[str] = field(default_factory=list)
    identity_claims: list[str] = field(default_factory=list)
    threat_rules: list[dict] = field(default_factory=list)
    life_themes: list[str] = field(default_factory=list)
    memory_seeds: list[MemorySeed] = field(default_factory=list)
    name: Optional[str] = None
    archetype: Optional[str] = None
    profile_id: Optional[str] = None
    summary: str = ""
