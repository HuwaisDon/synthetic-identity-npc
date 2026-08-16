from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from character.compiler import CharacterCompiler
from character.dsl_parser import CharacterDSLParser
from character.validator import CharacterValidator
from schemas.character_schema import CharacterSchema, GoalTemplate, ThreatRule
from schemas.memory_schema import EdgeType, EventType, MemoryEdge, MemoryNode, PsychologicalEffect


class CharacterLoader:
    """Load a character schema from JSON, YAML, or a compiled profile."""

    def __init__(self, source_dir: Optional[str | Path] = None):
        self.source_dir = Path(source_dir) if source_dir is not None else Path(__file__).resolve().parents[1] / "characters"
        self.parser = CharacterDSLParser()
        self.validator = CharacterValidator()
        self.compiler = CharacterCompiler()

    def load(self, character_id: str) -> CharacterSchema:
        path = self._resolve_path(character_id)
        if path is None:
            raise FileNotFoundError(f"Character definition not found for: {character_id}")

        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            memories = [self._coerce_memory(item) for item in payload.get("memories", [])]
            memory_edges = [self._coerce_memory_edge(item) for item in payload.get("memory_edges", [])]
            goals = [
                GoalTemplate(**item) if isinstance(item, dict) else item
                for item in payload.get("goals", [])
            ]
            threat_rules = [
                ThreatRule(**item) if isinstance(item, dict) else item
                for item in payload.get("threat_rules", [])
            ]
            return CharacterSchema(
                character_id=payload.get("character_id", character_id),
                name=payload.get("name", character_id),
                archetype=payload.get("archetype"),
                identity=payload.get("identity", {}),
                goals=goals,
                threat_rules=threat_rules,
                reframe_library=payload.get("reframe_library", {}),
                memories=memories,
                memory_edges=memory_edges,
                character_brief=payload.get("character_brief", ""),
                provenance=payload.get("provenance", {}),
            )

        profile = self.parser.parse(path)
        errors = self.validator.validate(profile)
        if errors:
            raise ValueError(f"Invalid character profile for {character_id}: {'; '.join(errors)}")
        return self.compiler.compile(profile, character_id=character_id)

    def _resolve_path(self, character_id: str) -> Optional[Path]:
        if Path(character_id).exists():
            return Path(character_id)

        for suffix in (".json", ".yaml", ".yml"):
            candidate = self.source_dir / f"{character_id}{suffix}"
            if candidate.exists():
                return candidate
        return None

    def _coerce_memory(self, item: dict) -> MemoryNode:
        payload = dict(item)
        event_type = payload.pop("event_type", None)
        if isinstance(event_type, str):
            payload["event_type"] = EventType(event_type)

        effects = payload.pop("psychological_effects", [])
        if effects:
            payload["psychological_effects"] = [
                PsychologicalEffect(**effect) if isinstance(effect, dict) else effect
                for effect in effects
            ]
        return MemoryNode(**payload)

    def _coerce_memory_edge(self, item: dict) -> MemoryEdge:
        payload = dict(item)
        edge_type = payload.pop("edge_type", None)
        if isinstance(edge_type, str):
            payload["edge_type"] = EdgeType(edge_type)
        return MemoryEdge(**payload)
