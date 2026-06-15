"""
core/cognition_pipeline.py

CognitionPipeline: Runtime Orchestration of the Full Cognitive Loop

This is the central execution engine. Every turn, this pipeline:

1. Receives player input and current NPC state
2. Runs memory retrieval + spreading activation (existing system)
3. Applies emotional persistence (residue from prior turns)
4. Runs attention filtering
5. Runs goal arbitration
6. Evaluates self-concept defense
7. Runs predictive simulation
8. Compresses to cognitive summary
9. Generates LLM prompt block
10. Calls LLM for language realization
11. Updates all engine states (post-turn)
12. Persists state

The pipeline is the ONLY entry point for a conversation turn.
No engine should be called directly from the API layer.

Architecture note:
  This file intentionally integrates the existing ChromaDB/NetworkX
  activation systems as abstract interfaces (retrieve_memories, spread_activation).
  Replace the stub implementations with your actual systems.
"""

from __future__ import annotations
import logging
import time
from typing import Optional
from dataclasses import dataclass, field

from schemas.cognitive_schemas import (
    EmotionalState, EmotionalReading, EmotionType,
    NPCCognitiveState,CognitiveSummary,
)
from engines.goal_engine import GoalEngine, build_morgan_goal_profile
from engines.attention_engine import AttentionEngine
from engines.emotional_persistence import EmotionalPersistenceEngine
from engines.self_concept_defense import SelfConceptDefenseSystem
from engines.predictive_simulation import PredictiveSimulationEngine
from engines.cognitive_summarizer import CognitiveSummarizer
from engines.suppression_engine import SuppressionEngine
from llm.gemini_client import GeminiClient
# from llm.openrouter_client import OpenRouterClient
from llm.prompt_builder import BehavioralPromptBuilder
from llm.response_validator import ResponseValidator
from persistence.npc_state_store import NPCStateStore


from memory.memory_store import MemoryStore
from memory.activation import ActivationEngine
from memory.spreading import SpreadingActivationEngine
from memory.morgan_memories import MORGAN_MEMORIES

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# TURN INPUT / OUTPUT STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class TurnInput:
    """Everything the pipeline needs to process a conversation turn."""
    npc_id: str
    player_message: str
    player_trust_level: float = 0.5          # accumulated trust
    threat_level: float = 0.0                # perceived danger from this message
    node_emotion_tags: dict = field(default_factory=dict)  # node_id -> [emotion_labels]
    node_descriptions: dict = field(default_factory=dict)  # node_id -> brief text


@dataclass
class TurnOutput:
    npc_response: str
    cognitive_state: NPCCognitiveState
    prompt_block: str
    activated_nodes: dict

    cognitive_summary: CognitiveSummary | None = None
    cognitive_summary_text: str = ""

    retrieved_memories: list = field(default_factory=list)
    activated_memories: list = field(default_factory=list)
    spread_results: list = field(default_factory=list)
    suppression_results: list = field(default_factory=list)
    attention_activation_scores: dict = field(default_factory=dict)
    activation_emotional_state: dict = field(default_factory=dict)
    processing_time_ms: float = 0.0


# ─────────────────────────────────────────────
# COGNITION PIPELINE
# ─────────────────────────────────────────────

class CognitionPipeline:
    """
    Orchestrates the full per-turn cognitive loop for a single NPC.
    
    One pipeline instance per NPC.
    Maintains engine state across turns.
    """

    def __init__(
        self,
        npc_id: str,
        npc_name: str = "Morgan",
        character_brief: str = "",
        state_store: Optional[NPCStateStore] = None,
        llm_client = None,
        memory_retriever = None,   # callable: (str, str) -> dict[str, float]
        activation_spreader = None,  # callable: (list[str], dict) -> dict[str, float]
    ):
        self.npc_id = npc_id
        self.npc_name = npc_name
        self.character_brief = character_brief
        self.state_store = state_store or NPCStateStore()
        self.llm_client = llm_client or GeminiClient()
        # self.llm_client = OpenRouterClient()
        self.memory_retriever = memory_retriever
        self.activation_spreader = activation_spreader
        self.memory_store = MemoryStore(npc_id)
        self.activation_engine = ActivationEngine()
        self.spreading_engine = SpreadingActivationEngine()
        self.suppression_engine = SuppressionEngine()
        self._load_seed_memories()
        self._last_emotional_state = EmotionalState()
        

        # Conversation history for LLM context
        self._conversation_history: list[dict] = []
        self._turn_count: int = 0
        self._focal_history: list[list[str]] = []
        self._player_trust: float = 0.5

        self.prompt_builder = BehavioralPromptBuilder()
        self.validator = ResponseValidator()

        # Initialize or restore engines
        self._init_engines()

    def _init_engines(self) -> None:
        """Initialize engines, restoring from persistence if available."""
        goal_engine, persistence_engine, self_concept, meta = \
            self.state_store.restore_engines(self.npc_id)

        if goal_engine:
            self.goal_engine = goal_engine
            self.persistence_engine = persistence_engine
            self.self_concept = self_concept
            self._turn_count = meta["turn_count"]
            self._player_trust = meta["player_trust"]
            logger.info(
                f"[Pipeline] Restored {self.npc_id}: "
                f"turn={self._turn_count}, trust={self._player_trust:.2f}"
            )
        else:
            # Fresh NPC
            self.goal_engine = GoalEngine(build_morgan_goal_profile())
            self.persistence_engine = EmotionalPersistenceEngine()
            self.self_concept = SelfConceptDefenseSystem()
            logger.info(f"[Pipeline] New NPC initialized: {self.npc_id}")

        self.attention_engine = AttentionEngine()
        self.predictive_engine = PredictiveSimulationEngine(self._player_trust)
        self.summarizer = CognitiveSummarizer()

    def _load_seed_memories(self) -> None:
        """Load local seed memories for known demo NPCs."""
        if "morgan" in self.npc_id.lower():
            self.memory_store.load_all(MORGAN_MEMORIES)

    # ── MAIN TURN PROCESSING ──────────────────

    def process_turn(self, turn_input: TurnInput) -> TurnOutput:
        """
        Process a single conversation turn through the full cognitive pipeline.
        """
        start_time = time.time()
        previous_player_trust = self._player_trust
        self._turn_count += 1
        self._player_trust = turn_input.player_trust_level

        logger.info(
            f"[Pipeline] Turn {self._turn_count} | "
            f"NPC={self.npc_id} | "
            f"trust={self._player_trust:.2f} | "
            f"threat={turn_input.threat_level:.2f}"
        )

        # ── STEP 1: Memory Retrieval + Activation + Spreading ──
        prior_emotional_state = self.persistence_engine.get_modified_baseline(
            self._last_emotional_state
        )
        activation_emotional_state = emotional_state_to_activation_dict(
            prior_emotional_state
        )

        if self.memory_retriever:
            memory_state = self.memory_retriever(
                npc_id=self.npc_id,
                query=turn_input.player_message,
                emotional_state=activation_emotional_state,
            )
        else:
            memory_state = self.retrieve_and_activate_memories(
                query=turn_input.player_message,
                emotional_state=activation_emotional_state,
            )

        retrieved = memory_state["retrieved"]
        activated = memory_state["activated"]
        spread = memory_state["spread"]

        logger.debug(
            f"[Pipeline] Retrieved memories: "
            f"{[m[0] for m in retrieved]}"
        )

        # Convert activated memories into provenance-rich node map.
        activated_nodes = {}
        for memory in activated:
            activated_nodes[memory.memory_id] = {
                "activation": memory.activation_score,
                "source": "direct",
                "parent": None,
            }

        # Add spread activation influence
        for spread_result in spread:

            target = spread_result.target_memory

            spread_strength = spread_result.spread_strength

            existing = activated_nodes.get(target)
            if existing is None or spread_strength > existing["activation"]:
                activated_nodes[target] = {
                    "activation": spread_strength,
                    "source": "spread",
                    "parent": spread_result.source_memory,
                }

        activated_node_scores = activation_scores_from_provenance(
            activated_nodes
        )

        suppression_state = self.suppression_engine.evaluate(
            activated_memories=activated,
            activated_nodes=activated_node_scores,
        )
        attention_activation_scores = (
            self.suppression_engine.apply_attention_interception(
                activated_nodes=activated_node_scores,
                suppression_state=suppression_state,
            )
        )

        # ── STEP 3: Build Base Emotional State ─
        base_emotional_state = self._build_base_emotional_state(
            activated_nodes=activated_node_scores,
            node_emotion_tags=turn_input.node_emotion_tags,
        )

        # ── STEP 4: Apply Emotional Persistence ─
        # Residue from past turns modifies baseline
        modified_emotional_state = self.persistence_engine.get_modified_baseline(
            base_emotional_state
        )
        modified_emotional_state.readings.extend(
            self.suppression_engine.build_leakage_readings(
                suppression_state=suppression_state,
                node_emotion_tags=turn_input.node_emotion_tags,
            )
        )
        modified_emotional_state = recompute_emotional_state(
            modified_emotional_state
        )

        # ── STEP 5: Goal Arbitration ──────────
        goal_state = self.goal_engine.arbitrate(
            emotional_state=modified_emotional_state,
            activated_node_ids=list(activated_node_scores.keys()),
            player_trust_level=self._player_trust,
            threat_level=turn_input.threat_level,
        )
        goal_state.strategic_silence = list(
            set(goal_state.strategic_silence + suppression_state.intercepted_nodes)
        )
        goal_state.concealment_pressure = max(
            goal_state.concealment_pressure,
            suppression_state.total_disclosure_inhibition,
        )
        goal_state.manipulation_intent = max(
            goal_state.manipulation_intent,
            suppression_state.total_deflection_pressure * 0.7,
        )
        if suppression_state.total_deflection_pressure > 0.45:
            goal_state.disclosure_pressure = max(
                0.0,
                goal_state.disclosure_pressure
                - suppression_state.total_deflection_pressure * 0.35,
            )

        # ── STEP 6: Attention Filtering ───────
        attention_state = self.attention_engine.process(
            activated_nodes=attention_activation_scores,
            emotional_state=modified_emotional_state,
            goal_state=goal_state,
            node_emotion_tags=turn_input.node_emotion_tags,
        )

        # ── STEP 7: Self-Concept Defense ──────
        self_concept_state = self.self_concept.evaluate(
            activated_nodes=activated_node_scores,
            emotional_state=modified_emotional_state,
            goal_state=goal_state,
        )

        # ── STEP 8: Predictive Simulation ─────
        self.predictive_engine.update_trust(self._player_trust)
        predictive_state = self.predictive_engine.simulate(
            focal_nodes=attention_state.focal_nodes,
            node_descriptions=turn_input.node_descriptions,
            emotional_state=modified_emotional_state,
            goal_state=goal_state,
            attention_state=attention_state,
            self_concept_state=self_concept_state,
        )

        # ── STEP 9: Cognitive Summarization ───
        cognitive_summary = self.summarizer.summarize(
            emotional_state=modified_emotional_state,
            goal_state=goal_state,
            attention_state=attention_state,
            self_concept_state=self_concept_state,
            predictive_state=predictive_state,
            persistence_state=self.persistence_engine.state,
            node_descriptions=turn_input.node_descriptions,
        )

        # ── STEP 10: Build LLM Prompt ─────────

        prompt_block = self.summarizer.to_prompt_block(
            cognitive_summary,
            self.npc_name
        )

        acting_note = self.prompt_builder.build_acting_note(
            cognitive_summary,
            self.npc_name
        )

        system_instruction = (
            f"{self.character_brief}\n\n"
            f"{acting_note}"
        )

        # ── STEP 11: LLM Call ─────────────────
        self._conversation_history.append({
            "role": "user",
            "content": turn_input.player_message
        })
        npc_response = self.llm_client.generate_response(
            prompt=prompt_block,
            system_instruction=system_instruction,
            history=self._conversation_history,
        )

        # ── STEP 11.5: Validate ───────────────
        self.validator.validate(npc_response, cognitive_summary)

        self._conversation_history.append({
            "role": "assistant",
            "content": npc_response
        })

        # ── STEP 12: Post-Turn State Updates ──
        self._post_turn_updates(
            emotional_readings=modified_emotional_state.readings,
            focal_nodes=attention_state.focal_nodes,
            chosen_strategy=predictive_state.chosen_strategy,
            goal_state=goal_state,
            trust_level=self._player_trust,
            previous_trust_level=previous_player_trust,
            threat_level=turn_input.threat_level,
        )
        self._focal_history.append(list(attention_state.focal_nodes))

        # ── STEP 13: Persist State ────────────
        self.state_store.save(
            npc_id=self.npc_id,
            goal_engine=self.goal_engine,
            persistence_engine=self.persistence_engine,
            self_concept_system=self.self_concept,
            player_trust=self._player_trust,
            turn_count=self._turn_count,
        )

        self._last_emotional_state = modified_emotional_state

        # ── ASSEMBLE OUTPUT ───────────────────
        processing_ms = (time.time() - start_time) * 1000

        cognitive_state = NPCCognitiveState(
            npc_id=self.npc_id,
            turn=self._turn_count,
            emotional_state=modified_emotional_state,
            goal_state=goal_state,
            attention_state=attention_state,
            self_concept_state=self_concept_state,
            predictive_state=predictive_state,
            cognitive_summary=cognitive_summary,
        )

        return TurnOutput(
            npc_response=npc_response,
            cognitive_state=cognitive_state,
            prompt_block=prompt_block,
            activated_nodes=activated_nodes,
            cognitive_summary_text=prompt_block,
            cognitive_summary=cognitive_summary,
            retrieved_memories=retrieved,
            activated_memories=activated,
            spread_results=spread,
            suppression_results=suppression_state.results,
            attention_activation_scores=attention_activation_scores,
            activation_emotional_state=activation_emotional_state,
            processing_time_ms=processing_ms,
        )

    def retrieve_and_activate_memories(
        self,
        query: str,
        emotional_state: dict,
    ) -> dict:
        """
        Run semantic retrieval, emotional activation, and spreading activation
        using the pipeline-owned runtime engines.
        """
        retrieved = self.memory_store.semantic_search(query)

        activated = self.activation_engine.compute_activation(
            retrieved,
            emotional_state,
        )

        spread_results = self.spreading_engine.spread(activated)

        return {
            "retrieved": retrieved,
            "activated": activated,
            "spread": spread_results,
        }

    # ── POST-TURN UPDATES ─────────────────────

    def _post_turn_updates(
        self,
        emotional_readings,
        focal_nodes: list[str],
        chosen_strategy: str,
        goal_state,
        trust_level: float,
        previous_trust_level: float,
        threat_level: float,
    ) -> None:
        """
        Update all engines based on what happened this turn.
        """
        # Record to emotional persistence
        # Nodes in strategic silence were 'expressed' — reduce their residue
        expressed = focal_nodes if chosen_strategy == "disclose" else []
        self.persistence_engine.record_turn(
            emotional_readings=emotional_readings,
            focal_nodes=focal_nodes,
            resolved_nodes=expressed,
        )

        # Goal engine runtime updates
        if trust_level > previous_trust_level:
            self.goal_engine.on_trust_gained(trust_level - previous_trust_level)
        if threat_level > 0.3:
            self.goal_engine.on_threat_detected(threat_level)

    def _build_base_emotional_state(
        self,
        activated_nodes: dict[str, float],
        node_emotion_tags: dict[str, list[str]],
    ) -> EmotionalState:
        """
        Derive emotional state from activated node emotion tags.
        This is where your existing emotional-edge-biased activation feeds in.
        
        Replace this stub with your actual emotion computation.
        """
        readings = []
        emotion_totals: dict[str, float] = {}

        # Habituation: Nodes that have been focal recently lose some emotional impact
        # This prevents "trauma fixation" loops.
        recent_focal_counts = {}
        for turn_focal in self._focal_history[-5:]:
            for node_id in turn_focal:
                recent_focal_counts[node_id] = recent_focal_counts.get(node_id, 0) + 1

        for node_id, activation in activated_nodes.items():
            if activation < 0.1:
                continue
            
            habituation = max(0.4, 1.0 - (recent_focal_counts.get(node_id, 0) * 0.15))
            tags = node_emotion_tags.get(node_id, [])
            effective_activation = activation * habituation
            
            for tag in tags:
                emotion_totals[tag] = emotion_totals.get(tag, 0.0) + effective_activation

        for emotion_label, total in emotion_totals.items():
            try:
                emotion_type = EmotionType(emotion_label)
                intensity = min(1.0, total / 3.0)
                if intensity > 0.1:
                    readings.append(EmotionalReading(
                        emotion=emotion_type,
                        intensity=intensity,
                    ))
            except ValueError:
                pass

        dominant = max(readings, key=lambda r: r.intensity) if readings else None

        # Compute arousal and valence
        negative_emotions = {
            EmotionType.FEAR, EmotionType.GRIEF, EmotionType.SHAME,
            EmotionType.GUILT, EmotionType.ANGER, EmotionType.DREAD,
            EmotionType.NUMBNESS
        }
        total_intensity = sum(r.intensity for r in readings)
        if readings:
            neg_intensity = sum(r.intensity for r in readings if r.emotion in negative_emotions)
            valence = -1.0 * (neg_intensity / total_intensity) + \
                      1.0 * ((total_intensity - neg_intensity) / total_intensity)
            arousal = min(1.0, total_intensity / len(readings))
        else:
            valence = 0.0
            arousal = 0.0

        return EmotionalState(
            readings=readings,
            dominant=dominant,
            valence=valence,
            arousal=arousal,
        )

    def _build_system_prompt(self, cognitive_block: str) -> str:
        """Assemble the full system prompt from character brief + cognitive state block."""
        return f"""{self.character_brief}

{cognitive_block}

Respond as {self.npc_name}. Do not explain, describe, or narrate internal state. 
Speak. Let the internal state shape the voice, the deflections, the silences.
"""

    def reset_conversation(self) -> None:
        """Clear conversation history without resetting cognitive state."""
        self._conversation_history = []


# ─────────────────────────────────────────────
# RUNTIME ADAPTERS
# ─────────────────────────────────────────────
def emotional_state_to_activation_dict(
    emotional_state: EmotionalState,
) -> dict[str, float]:
    """
    Convert pipeline EmotionalState into ActivationEngine's compact affect dict.
    """
    affect = {
        "sadness": 0.0,
        "fear": 0.0,
        "anger": 0.0,
    }

    for reading in emotional_state.readings + emotional_state.residue:
        intensity = max(0.0, min(1.0, reading.intensity))

        if reading.emotion in (
            EmotionType.GRIEF,
            EmotionType.LONGING,
            EmotionType.NUMBNESS,
        ):
            affect["sadness"] = max(affect["sadness"], intensity)
        elif reading.emotion in (EmotionType.FEAR, EmotionType.DREAD):
            affect["fear"] = max(affect["fear"], intensity)
        elif reading.emotion in (EmotionType.ANGER, EmotionType.CONTEMPT):
            affect["anger"] = max(affect["anger"], intensity)
        elif reading.emotion in (EmotionType.SHAME, EmotionType.GUILT):
            affect["sadness"] = max(affect["sadness"], intensity * 0.6)
            affect["fear"] = max(affect["fear"], intensity * 0.4)

    if emotional_state.valence < -0.2:
        affect["sadness"] = max(
            affect["sadness"],
            min(1.0, abs(emotional_state.valence) * 0.4),
        )

    if emotional_state.arousal > 0.4:
        affect["fear"] = max(
            affect["fear"],
            min(1.0, emotional_state.arousal * 0.5),
        )

    return affect


def activation_scores_from_provenance(
    activated_nodes: dict[str, dict],
) -> dict[str, float]:
    """Extract node scores for engines that still consume flat activations."""
    return {
        node_id: data["activation"]
        for node_id, data in activated_nodes.items()
    }


def recompute_emotional_state(
    emotional_state: EmotionalState,
) -> EmotionalState:
    """Refresh dominant emotion, valence, and arousal after adding readings."""
    if not emotional_state.readings:
        emotional_state.dominant = None
        emotional_state.valence = 0.0
        emotional_state.arousal = 0.0
        return emotional_state

    negative_emotions = {
        EmotionType.FEAR,
        EmotionType.GRIEF,
        EmotionType.SHAME,
        EmotionType.GUILT,
        EmotionType.ANGER,
        EmotionType.DREAD,
        EmotionType.NUMBNESS,
    }
    emotional_state.dominant = max(
        emotional_state.readings,
        key=lambda reading: reading.intensity,
    )
    total_intensity = sum(reading.intensity for reading in emotional_state.readings)
    if total_intensity <= 0:
        emotional_state.valence = 0.0
        emotional_state.arousal = 0.0
        return emotional_state

    negative_intensity = sum(
        reading.intensity
        for reading in emotional_state.readings
        if reading.emotion in negative_emotions
    )
    positive_intensity = total_intensity - negative_intensity
    emotional_state.valence = (
        positive_intensity - negative_intensity
    ) / total_intensity
    emotional_state.arousal = min(
        1.0,
        total_intensity / len(emotional_state.readings),
    )
    return emotional_state


def _default_llm_caller(system_prompt: str, conversation_history: list[dict]) -> str:
    """
    Default LLM caller stub. Replace with your actual LLM integration.
    """
    logger.warning("[Pipeline] Using stub LLM caller — replace with actual LLM")
    return "[LLM response would appear here]"
