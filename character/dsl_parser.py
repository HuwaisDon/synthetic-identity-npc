from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import yaml

from schemas.character_schema import MemorySeed, PsychologicalProfile


class CharacterDSLParser:
    """Parse a YAML character definition into a PsychologicalProfile."""

    def parse(self, source: str | Path) -> PsychologicalProfile:
        path = Path(source)
        if path.exists():
            payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        else:
            payload = yaml.safe_load(source) or {}

        return self._deserialize(payload)

    def _deserialize(self, payload: dict[str, Any]) -> PsychologicalProfile:
        identity = payload.get("identity", {}) or {}
        memory_seeds = [self._coerce_memory_seed(item) for item in payload.get("memory_seeds", [])]
        threat_rules = payload.get("threat_rules", []) or []
        return PsychologicalProfile(
            identity=identity,
            values=payload.get("values") or [],
            beliefs=payload.get("beliefs") or [],
            core_motivations=payload.get("core_motivations") or [],
            goals=payload.get("goals") or [],
            personality=payload.get("personality") or {},
            speech_style=payload.get("speech_style") or {},
            relationships=payload.get("relationships") or [],
            traumas=payload.get("traumas") or [],
            secrets=payload.get("secrets") or [],
            defense_mechanisms=payload.get("defense_mechanisms") or [],
            identity_claims=identity.get("identity_claims") or [],
            threat_rules=[dict(item) for item in threat_rules],
            life_themes=payload.get("life_themes") or [],
            memory_seeds=memory_seeds,
            name=payload.get("name"),
            archetype=payload.get("personality", {}).get("archetype") if isinstance(payload.get("personality"), dict) else None,
        )

    def _coerce_memory_seed(self, item: dict[str, Any]) -> MemorySeed:
        payload = dict(item)
        event_type = payload.get("event_type")
        if isinstance(event_type, str):
            payload["event_type"] = event_type
        return MemorySeed(**payload)
