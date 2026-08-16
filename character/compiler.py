from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from schemas.character_schema import (
    CharacterSchema,
    GoalTemplate,
    MemorySeed,
    PsychologicalProfile,
    ThreatRule,
)
from schemas.memory_schema import EdgeType, EventType, MemoryEdge, MemoryNode, PsychologicalEffect


class MemoryBuilder:
    def build(self, profile: PsychologicalProfile, character_id: str, provenance: dict) -> list[MemoryNode]:
        memories: list[MemoryNode] = []
        for seed in profile.memory_seeds:
            memory = MemoryNode(
                memory_id=seed.seed_id,
                npc_id=character_id,
                event_type=EventType(seed.event_type) if isinstance(seed.event_type, str) else EventType.TRAUMA,
                age_at_event=seed.age_at_event if seed.age_at_event is not None else 0,
                objective_description=seed.summary,
                self_narrative_description=seed.summary,
                emotional_weight=seed.emotional_weight,
                valence=seed.valence,
                suppression_level=seed.suppression_level,
                sensory_tags=list(seed.sensory_tags),
                people_involved=list(seed.people_involved),
                location=seed.location or "",
                psychological_effects=[PsychologicalEffect(trait=item, magnitude=0.5) for item in seed.associated_concepts[:3]],
                associated_concepts=list(seed.associated_concepts),
                embedding=None,
            )
            memory.provenance = {
                "source_seed": seed.seed_id,
                "source_profile_id": profile.profile_id or character_id,
                "compiler_version": "phase2-1",
                **provenance,
            }
            memories.append(memory)
        return memories


class GoalBuilder:
    def build(self, profile: PsychologicalProfile, character_id: str, provenance: dict) -> list[GoalTemplate]:
        goals: list[GoalTemplate] = []
        for index, goal_text in enumerate(profile.goals or []):
            goals.append(
                GoalTemplate(
                    goal_id=f"g_{index}",
                    goal_type="identity",
                    description=goal_text,
                    utility=0.7,
                    urgency=0.5,
                    active=True,
                    blocking_topics=[],
                    linked_memory_nodes=[],
                )
            )
        if not goals:
            goals.append(
                GoalTemplate(
                    goal_id="g_0",
                    goal_type="identity",
                    description="Preserve coherence",
                    utility=0.7,
                    urgency=0.5,
                    active=True,
                    blocking_topics=[],
                    linked_memory_nodes=[],
                )
            )
        return goals


class ThreatBuilder:
    def build(self, profile: PsychologicalProfile, provenance: dict) -> list[ThreatRule]:
        threats: list[ThreatRule] = []
        for rule in profile.threat_rules or []:
            threats.append(
                ThreatRule(
                    trigger_concepts=list(rule.get("trigger_concepts", [])),
                    threatened_claim_idx=int(rule.get("threatened_claim_idx", 0)),
                    threat_type=str(rule.get("threat_type", "threat")),
                    defense_mechanism=str(rule.get("defense_mechanism", "reframe")),
                )
            )
        return threats


class RelationshipBuilder:
    def build(self, profile: PsychologicalProfile, provenance: dict) -> list[MemoryEdge]:
        return []


class CharacterCompiler:
    def __init__(self) -> None:
        self.memory_builder = MemoryBuilder()
        self.goal_builder = GoalBuilder()
        self.threat_builder = ThreatBuilder()
        self.relationship_builder = RelationshipBuilder()

    def compile(self, profile: PsychologicalProfile, character_id: Optional[str] = None) -> CharacterSchema:
        character_id = character_id or profile.profile_id or "psychological-profile"
        provenance = {
            "source_profile_id": profile.profile_id or character_id,
            "compiler_version": "phase2-1",
        }
        memories = self.memory_builder.build(profile, character_id, provenance)
        goals = self.goal_builder.build(profile, character_id, provenance)
        threats = self.threat_builder.build(profile, provenance)
        edges = self.relationship_builder.build(profile, provenance)

        character = CharacterSchema(
            character_id=character_id,
            name=profile.name or character_id,
            archetype=profile.archetype or profile.personality.get("archetype") if isinstance(profile.personality, dict) else None,
            identity={
                "core_beliefs": list(profile.identity.get("core_beliefs", [])),
                "values": list(profile.identity.get("values", [])),
                "identity_claims": list(profile.identity_claims),
            },
            goals=goals,
            threat_rules=threats,
            reframe_library={},
            memories=memories,
            memory_edges=edges,
            character_brief=profile.summary or "",
            provenance=provenance,
        )
        return character
