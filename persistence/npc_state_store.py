"""
persistence/npc_state_store.py

NPC State Persistence: Serialization and Restoration Across Sessions

Architecture note:
  The cognitive state of an NPC is NOT ephemeral.
  A conversation that caused shame should be felt in the next conversation.
  Residue accumulates. Goals shift. Coherence fractures.
  
  This store handles:
    - Serialization of full cognitive state to JSON
    - Restoration of engines from persisted state
    - Session-keyed storage (per-NPC, per-world, optionally per-player)
    - State versioning (for schema evolution)
  
  Storage backend is intentionally pluggable.
  Default: JSON files. Production: Redis or Postgres JSONB.
"""

from __future__ import annotations
import json
import os
import time
import logging
from pathlib import Path
from typing import Optional
from dataclasses import asdict

logger = logging.getLogger(__name__)

STATE_VERSION = "1.0.0"
DEFAULT_STATE_DIR = Path("./npc_states")


class NPCStateStore:
    """
    Saves and loads full NPC cognitive engine state.
    JSON-backed, filesystem by default.
    """

    def __init__(self, state_dir: Optional[Path] = None):
        self.state_dir = state_dir or DEFAULT_STATE_DIR
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        npc_id: str,
        goal_engine,
        persistence_engine,
        self_concept_system,
        player_trust: float,
        turn_count: int,
        extra_context: Optional[dict] = None,
    ) -> Path:
        """
        Serialize all engine states to JSON.
        """
        payload = {
            "version": STATE_VERSION,
            "npc_id": npc_id,
            "saved_at": time.time(),
            "turn_count": turn_count,
            "player_trust": player_trust,
            "goal_engine": goal_engine.serialize(),
            "persistence_engine": persistence_engine.serialize(),
            "self_concept": self_concept_system.serialize(),
            "extra": extra_context or {},
        }

        path = self._path(npc_id)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        logger.info(f"[StateStore] Saved {npc_id} → {path}")
        return path

    def load(self, npc_id: str) -> Optional[dict]:
        """
        Load raw state payload. Caller reconstructs engines from it.
        Returns None if no saved state exists.
        """
        path = self._path(npc_id)
        if not path.exists():
            logger.info(f"[StateStore] No saved state for {npc_id}")
            return None

        with open(path, "r") as f:
            data = json.load(f)

        logger.info(
            f"[StateStore] Loaded {npc_id} | "
            f"turn={data.get('turn_count')}, "
            f"trust={data.get('player_trust'):.2f}"
        )
        return data

    def restore_engines(self, npc_id: str):
        """
        Restore all cognitive engines from saved state.
        Returns tuple (goal_engine, persistence_engine, self_concept_system, meta)
        or (None, None, None, {}) if no state found.
        """
        from engines.goal_engine import GoalEngine
        from engines.emotional_persistence import EmotionalPersistenceEngine
        from engines.self_concept_defense import SelfConceptDefenseSystem

        data = self.load(npc_id)
        if not data:
            return None, None, None, {}

        goal_engine = GoalEngine.deserialize(data["goal_engine"])
        persistence_engine = EmotionalPersistenceEngine.deserialize(data["persistence_engine"])
        self_concept = SelfConceptDefenseSystem.deserialize(data["self_concept"])
        meta = {
            "turn_count": data.get("turn_count", 0),
            "player_trust": data.get("player_trust", 0.5),
            "extra": data.get("extra", {}),
        }

        return goal_engine, persistence_engine, self_concept, meta

    def delete(self, npc_id: str) -> bool:
        path = self._path(npc_id)
        if path.exists():
            path.unlink()
            logger.info(f"[StateStore] Deleted state for {npc_id}")
            return True
        return False

    def list_npcs(self) -> list[str]:
        return [p.stem for p in self.state_dir.glob("*.json")]

    def _path(self, npc_id: str) -> Path:
        safe_id = npc_id.replace("/", "_").replace("\\", "_")
        return self.state_dir / f"{safe_id}.json"
