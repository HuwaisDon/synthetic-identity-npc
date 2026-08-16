"""
api/routes.py

FastAPI Integration: REST API for the Synthetic Identity NPC System

Endpoints:
  POST /npc/{npc_id}/turn     - Process a conversation turn
  GET  /npc/{npc_id}/state    - Get current cognitive state summary
  POST /npc/{npc_id}/reset    - Reset conversation (not cognitive state)
  DELETE /npc/{npc_id}/state  - Delete persisted state (restart NPC)
  GET  /npc/{npc_id}/debug    - Full internal state (dev only)

Architecture note:
  The API layer is thin. All logic lives in CognitionPipeline.
  The API only handles request/response translation and pipeline lifecycle.
  
  Pipelines are cached in-process (one per NPC). For multi-worker deployments,
  use Redis for state and reconstruct pipeline per request.
"""

from __future__ import annotations
import logging
import os
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from character.loader import CharacterLoader
from core.cognition_pipeline import CognitionPipeline, TurnInput
from persistence.npc_state_store import NPCStateStore

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Synthetic Identity NPC System",
    description="Autobiographical cognitive architecture for synthetic NPCs",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# PIPELINE REGISTRY
# In-process cache: npc_id -> CognitionPipeline
# ─────────────────────────────────────────────

_pipeline_registry: dict[str, CognitionPipeline] = {}
_state_store = NPCStateStore()


def get_pipeline(npc_id: str) -> CognitionPipeline:
    """Get or create pipeline for this NPC."""
    if npc_id not in _pipeline_registry:
        pipeline = _build_pipeline(npc_id)
        _pipeline_registry[npc_id] = pipeline
    return _pipeline_registry[npc_id]


def _build_pipeline(npc_id: str) -> CognitionPipeline:
    """
    Build a CognitionPipeline for the given NPC.
    In production: load character config from DB.
    """
    loader = CharacterLoader()
    character = None
    try:
        character = loader.load(npc_id)
    except FileNotFoundError:
        character = None

    return CognitionPipeline(
        npc_id=npc_id,
        npc_name=character.name if character else _npc_name_from_id(npc_id),
        character_brief=character.character_brief if character else "",
        state_store=_state_store,
        llm_client=_CallableLLMClient(_build_llm_caller()),
        character=character,
    )


def _build_llm_caller():
    """
    Build the LLM caller. Swap in your actual provider here.
    Supports: Anthropic, OpenAI, local Ollama, etc.
    """
    # Example: Anthropic integration
    # Uncomment and configure for production use
    #
    # import anthropic
    # client = anthropic.Anthropic()
    # def anthropic_caller(system_prompt: str, conversation_history: list) -> str:
    #     response = client.messages.create(
    #         model="claude-opus-4-5",
    #         max_tokens=1024,
    #         system=system_prompt,
    #         messages=conversation_history,
    #     )
    #     return response.content[0].text
    # return anthropic_caller

    def stub_caller(system_prompt: str, conversation_history: list) -> str:
        return f"[Configure LLM caller in api/routes.py | system_length={len(system_prompt)}]"

    return stub_caller


def _npc_name_from_id(npc_id: str) -> str:
    return npc_id.split("_")[0].capitalize()


class _CallableLLMClient:
    """Adapts legacy callable LLM hooks to the pipeline client interface."""

    def __init__(self, caller):
        self._caller = caller

    def generate_response(
        self,
        prompt: str | None = None,
        acting_note: str | None = None,
        system_instruction: str | None = None,
        history: list | None = None,
    ) -> str:
        system_prompt = "\n\n".join(
            part for part in (system_instruction, prompt or acting_note) if part
        )
        return self._caller(system_prompt, history or [])


# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────

class TurnRequest(BaseModel):
    player_message: str = Field(..., description="What the player said")
    player_trust_level: float = Field(0.5, ge=0.0, le=1.0)
    threat_level: float = Field(0.0, ge=0.0, le=1.0)
    node_emotion_tags: dict = Field(default_factory=dict)
    node_descriptions: dict = Field(default_factory=dict)


class TurnResponse(BaseModel):
    npc_response: str
    turn: int
    processing_time_ms: float
    # Cognitive state summary (non-sensitive; for UI display)
    surface_emotion: Optional[str] = None
    attentional_state: str = "open"
    stability: float = 1.0
    strategic_intent: str = "neutral"
    # Debug fields (omitted in production)
    debug_prompt_block: Optional[str] = None
    debug_activated_nodes: Optional[dict] = None


class StateResponse(BaseModel):
    npc_id: str
    turn: int
    player_trust: float
    goal_summary: list[dict]
    persistence_summary: dict
    self_concept_coherence: float


class ResetResponse(BaseModel):
    message: str
    npc_id: str


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.post("/npc/{npc_id}/turn", response_model=TurnResponse)
async def process_turn(npc_id: str, request: TurnRequest):
    """Process a single conversation turn for this NPC."""
    pipeline = get_pipeline(npc_id)

    turn_input = TurnInput(
        npc_id=npc_id,
        player_message=request.player_message,
        player_trust_level=request.player_trust_level,
        threat_level=request.threat_level,
        node_emotion_tags=request.node_emotion_tags,
        node_descriptions=request.node_descriptions,
    )

    try:
        output = pipeline.process_turn(turn_input)
    except Exception as e:
        logger.exception(f"Pipeline error for {npc_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    is_debug = os.environ.get("NPC_DEBUG_MODE", "false").lower() == "true"

    summary = output.cognitive_summary

    return TurnResponse(
        npc_response=output.npc_response,
        turn=output.cognitive_state.turn,
        processing_time_ms=output.processing_time_ms,
        surface_emotion=summary.surface_emotion,
        attentional_state=summary.attentional_state,
        stability=summary.emotional_stability,
        strategic_intent=summary.strategic_intent,
        debug_prompt_block=output.prompt_block if is_debug else None,
        debug_activated_nodes=output.activated_nodes if is_debug else None,
    )


@app.get("/npc/{npc_id}/state", response_model=StateResponse)
async def get_state(npc_id: str):
    """Get current cognitive state summary for this NPC."""
    pipeline = get_pipeline(npc_id)

    goal_summary = [
        {
            "id": g.goal_id,
            "type": g.goal_type.value,
            "utility": round(g.utility, 3),
            "urgency": round(g.urgency, 3),
            "active": g.active,
        }
        for g in pipeline.goal_engine.goals
    ]

    persistence = pipeline.persistence_engine.state
    persistence_summary = {
        "residue_count": len(persistence.residue),
        "mood_baseline_shift": round(persistence.mood_baseline_shift, 3),
        "rumination_node": persistence.rumination_node,
        "rumination_intensity": round(persistence.rumination_intensity, 3),
    }

    return StateResponse(
        npc_id=npc_id,
        turn=pipeline._turn_count,
        player_trust=pipeline._player_trust,
        goal_summary=goal_summary,
        persistence_summary=persistence_summary,
        self_concept_coherence=pipeline.self_concept.state.coherence,
    )


@app.post("/npc/{npc_id}/reset", response_model=ResetResponse)
async def reset_conversation(npc_id: str):
    """
    Reset conversation history without clearing cognitive state.
    The NPC's emotional residue, goals, and identity persist.
    """
    if npc_id in _pipeline_registry:
        _pipeline_registry[npc_id].reset_conversation()
    return ResetResponse(
        message="Conversation history cleared. Cognitive state preserved.",
        npc_id=npc_id,
    )


@app.delete("/npc/{npc_id}/state", response_model=ResetResponse)
async def delete_state(npc_id: str):
    """
    Delete all persisted state for this NPC and reset pipeline.
    Use with caution — NPC forgets everything.
    """
    _state_store.delete(npc_id)
    if npc_id in _pipeline_registry:
        del _pipeline_registry[npc_id]
    return ResetResponse(
        message="NPC state deleted. Pipeline will reinitialize on next turn.",
        npc_id=npc_id,
    )


@app.get("/npc/{npc_id}/debug")
async def debug_state(npc_id: str):
    """Full internal state dump. Dev/debug only. Disable in production."""
    if os.environ.get("NPC_DEBUG_MODE", "false").lower() != "true":
        raise HTTPException(status_code=403, detail="Debug mode not enabled")

    pipeline = get_pipeline(npc_id)

    return {
        "npc_id": npc_id,
        "turn": pipeline._turn_count,
        "goals": pipeline.goal_engine.serialize(),
        "persistence": pipeline.persistence_engine.serialize(),
        "self_concept": pipeline.self_concept.serialize(),
        "conversation_turns": len(pipeline._conversation_history),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "active_npcs": list(_pipeline_registry.keys())}

