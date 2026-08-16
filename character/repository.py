from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from character.loader import CharacterLoader
from schemas.character_schema import CharacterSchema


class CharacterRepository:
    """Resolve character definitions from disk and cache compiled characters."""

    def __init__(self, source_dir: Optional[str | Path] = None):
        self.loader = CharacterLoader(source_dir=source_dir)
        self._cache: dict[str, CharacterSchema] = {}

    def get(self, character_id: str) -> CharacterSchema:
        if character_id not in self._cache:
            self._cache[character_id] = self.loader.load(character_id)
        return self._cache[character_id]

    def load_yaml(self, path: str | Path) -> CharacterSchema:
        character_id = Path(path).stem
        return self.loader.load(character_id)

    def compile_yaml(self, path: str | Path) -> CharacterSchema:
        return self.load_yaml(path)

    def save(self, character: CharacterSchema, path: Optional[str | Path] = None) -> Path:
        target = Path(path) if path is not None else self.loader.source_dir / f"{character.character_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "character_id": character.character_id,
            "name": character.name,
            "archetype": character.archetype,
            "identity": character.identity,
            "goals": [
                {
                    "goal_id": goal.goal_id,
                    "goal_type": goal.goal_type,
                    "description": goal.description,
                    "utility": goal.utility,
                    "urgency": goal.urgency,
                    "active": goal.active,
                    "blocking_topics": goal.blocking_topics,
                    "linked_memory_nodes": goal.linked_memory_nodes,
                    "provenance": goal.provenance,
                }
                for goal in character.goals
            ],
            "threat_rules": [
                {
                    "trigger_concepts": rule.trigger_concepts,
                    "threatened_claim_idx": rule.threatened_claim_idx,
                    "threat_type": rule.threat_type,
                    "defense_mechanism": rule.defense_mechanism,
                    "provenance": rule.provenance,
                }
                for rule in character.threat_rules
            ],
            "reframe_library": character.reframe_library,
            "memories": [
                {
                    "memory_id": memory.memory_id,
                    "npc_id": memory.npc_id,
                    "event_type": memory.event_type.value if hasattr(memory.event_type, "value") else memory.event_type,
                    "age_at_event": memory.age_at_event,
                    "objective_description": memory.objective_description,
                    "self_narrative_description": memory.self_narrative_description,
                    "emotional_weight": memory.emotional_weight,
                    "valence": memory.valence,
                    "suppression_level": memory.suppression_level,
                    "sensory_tags": memory.sensory_tags,
                    "people_involved": memory.people_involved,
                    "location": memory.location,
                    "psychological_effects": [
                        {"trait": effect.trait, "magnitude": effect.magnitude}
                        for effect in memory.psychological_effects
                    ],
                    "associated_concepts": memory.associated_concepts,
                    "provenance": memory.provenance,
                }
                for memory in character.memories
            ],
            "memory_edges": [],
            "character_brief": character.character_brief,
            "provenance": character.provenance,
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    def list_ids(self) -> list[str]:
        ids = set()
        for path in self.loader.source_dir.glob("*.json"):
            ids.add(path.stem)
        for path in self.loader.source_dir.glob("*.yaml"):
            ids.add(path.stem)
        for path in self.loader.source_dir.glob("*.yml"):
            ids.add(path.stem)
        return sorted(ids)

    def reload(self, character_id: str) -> CharacterSchema:
        self._cache.pop(character_id, None)
        return self.get(character_id)
