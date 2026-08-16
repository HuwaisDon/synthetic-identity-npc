from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from api.routes import TurnRequest
from character.loader import CharacterLoader
from core.cognition_pipeline import CognitionPipeline, TurnInput
from persistence.npc_state_store import NPCStateStore
from schemas.character_schema import CharacterSchema


@dataclass
class RuntimeState:
    character: Optional[CharacterSchema] = None
    pipeline: Optional[CognitionPipeline] = None


class CogniRuntime:
    """Facade for external callers, replacing direct engine access."""

    def __init__(self, state_store: Optional[NPCStateStore] = None, loader: Optional[CharacterLoader] = None):
        self.state_store = state_store or NPCStateStore()
        self.loader = loader or CharacterLoader()
        self._runtime_state: dict[str, RuntimeState] = {}

    def load_character(self, character_id: str) -> CharacterSchema:
        character = self.loader.load(character_id)
        runtime_state = self._runtime_state.setdefault(character_id, RuntimeState(character=character))
        runtime_state.character = character
        if runtime_state.pipeline is None:
            runtime_state.pipeline = CognitionPipeline(
                npc_id=character.character_id,
                npc_name=character.name,
                character_brief=character.character_brief,
                state_store=self.state_store,
            )
        return character

    def process_turn(self, npc_id: str, turn_request: TurnRequest) -> object:
        runtime_state = self._runtime_state.setdefault(npc_id, RuntimeState())
        if runtime_state.pipeline is None:
            character = self.load_character(npc_id)
            runtime_state.character = character
            runtime_state.pipeline = CognitionPipeline(
                npc_id=character.character_id,
                npc_name=character.name,
                character_brief=character.character_brief,
                state_store=self.state_store,
            )
        turn_input = TurnInput(
            npc_id=npc_id,
            player_message=turn_request.player_message,
            player_trust_level=turn_request.player_trust_level,
            threat_level=turn_request.threat_level,
            node_emotion_tags=turn_request.node_emotion_tags,
            node_descriptions=turn_request.node_descriptions,
        )
        return runtime_state.pipeline.process_turn(turn_input)

    def get_state(self, npc_id: str) -> dict:
        runtime_state = self._runtime_state.get(npc_id)
        if runtime_state is None or runtime_state.pipeline is None:
            return {}
        return {
            "npc_id": npc_id,
            "turn": runtime_state.pipeline._turn_count,
            "player_trust": runtime_state.pipeline._player_trust,
        }

    def save(self, npc_id: str) -> None:
        runtime_state = self._runtime_state.get(npc_id)
        if runtime_state is None or runtime_state.pipeline is None:
            return
        self.state_store.save(
            npc_id,
            runtime_state.pipeline.goal_engine,
            runtime_state.pipeline.persistence_engine,
            runtime_state.pipeline.self_concept,
            runtime_state.pipeline._player_trust,
            runtime_state.pipeline._turn_count,
        )

    def reset(self, npc_id: str) -> None:
        runtime_state = self._runtime_state.get(npc_id)
        if runtime_state is None or runtime_state.pipeline is None:
            return
        runtime_state.pipeline.reset_conversation()

    def shutdown(self, npc_id: str) -> None:
        self.save(npc_id)
