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
    NPCCognitiveState
)
from engines.goal_engine import GoalEngine, build_morgan_goal_profile
from engines.attention_engine import AttentionEngine
from engines.emotional_persistence import EmotionalPersistenceEngine
from engines.self_concept_defense import SelfConceptDefenseSystem
from engines.predictive_simulation import PredictiveSimulationEngine
from engines.cognitive_summarizer import CognitiveSummarizer
from persistence.npc_state_store import NPCStateStore

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
    """Full output of a pipeline turn, including NPC response and internal state."""
    npc_response: str
    cognitive_state: NPCCognitiveState
    prompt_block: str                        # the LLM prompt that was used
    activated_nodes: dict                    # node_id -> activation_score
    cognitive_summary_text: str             # rendered summary (for debugging)
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
        llm_caller = None,   # callable: (str) -> str
        memory_retriever = None,   # callable: (str, str) -> dict[str, float]
        activation_spreader = None,  # callable: (list[str], dict) -> dict[str, float]
    ):
        self.npc_id = npc_id
        self.npc_name = npc_name
        self.character_brief = character_brief
        self.state_store = state_store or NPCStateStore()
        self.llm_caller = llm_caller or _default_llm_caller
        self.memory_retriever = memory_retriever or _stub_memory_retriever
        self.activation_spreader = activation_spreader or _stub_activation_spreader

        # Conversation history for LLM context
        self._conversation_history: list[dict] = []
        self._turn_count: int = 0
        self._player_trust: float = 0.5

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

    # ── MAIN TURN PROCESSING ──────────────────

    def process_turn(self, turn_input: TurnInput) -> TurnOutput:
        """
        Process a single conversation turn through the full cognitive pipeline.
        """
        start_time = time.time()
        self._turn_count += 1
        self._player_trust = turn_input.player_trust_level

        logger.info(
            f"[Pipeline] Turn {self._turn_count} | "
            f"NPC={self.npc_id} | "
            f"trust={self._player_trust:.2f} | "
            f"threat={turn_input.threat_level:.2f}"
        )

        # ── STEP 1: Memory Retrieval ──────────
        # Your existing ChromaDB system plugs in here
        seed_nodes = self.memory_retriever(
            npc_id=self.npc_id,
            query=turn_input.player_message,
        )
        logger.debug(f"[Pipeline] Seed nodes: {list(seed_nodes.keys())}")

        # ── STEP 2: Spreading Activation ─────
        # Your existing NetworkX spreading activation plugs in here
        activated_nodes = self.activation_spreader(
            seed_nodes=list(seed_nodes.keys()),
            graph=None,  # pass your graph here
        )
        # Merge seed activation scores into full activated map
        for node_id, score in seed_nodes.items():
            activated_nodes[node_id] = max(activated_nodes.get(node_id, 0.0), score)

        # ── STEP 3: Build Base Emotional State ─
        base_emotional_state = self._build_base_emotional_state(
            activated_nodes=activated_nodes,
            node_emotion_tags=turn_input.node_emotion_tags,
        )

        # ── STEP 4: Apply Emotional Persistence ─
        # Residue from past turns modifies baseline
        modified_emotional_state = self.persistence_engine.get_modified_baseline(
            base_emotional_state
        )

        # ── STEP 5: Goal Arbitration ──────────
        goal_state = self.goal_engine.arbitrate(
            emotional_state=modified_emotional_state,
            activated_node_ids=list(activated_nodes.keys()),
            player_trust_level=self._player_trust,
            threat_level=turn_input.threat_level,
        )

        # ── STEP 6: Attention Filtering ───────
        attention_state = self.attention_engine.process(
            activated_nodes=activated_nodes,
            emotional_state=modified_emotional_state,
            goal_state=goal_state,
            node_emotion_tags=turn_input.node_emotion_tags,
        )

        # ── STEP 7: Self-Concept Defense ──────
        self_concept_state = self.self_concept.evaluate(
            activated_nodes=activated_nodes,
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
        prompt_block = self.summarizer.to_prompt_block(cognitive_summary, self.npc_name)
        system_prompt = self._build_system_prompt(prompt_block)

        # ── STEP 11: LLM Call ─────────────────
        self._conversation_history.append({
            "role": "user",
            "content": turn_input.player_message
        })
        npc_response = self.llm_caller(
            system_prompt=system_prompt,
            conversation_history=self._conversation_history,
        )
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
            threat_level=turn_input.threat_level,
        )

        # ── STEP 13: Persist State ────────────
        self.state_store.save(
            npc_id=self.npc_id,
            goal_engine=self.goal_engine,
            persistence_engine=self.persistence_engine,
            self_concept_system=self.self_concept,
            player_trust=self._player_trust,
            turn_count=self._turn_count,
        )

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
            processing_time_ms=processing_ms,
        )

    # ── POST-TURN UPDATES ─────────────────────

    def _post_turn_updates(
        self,
        emotional_readings,
        focal_nodes: list[str],
        chosen_strategy: str,
        goal_state,
        trust_level: float,
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
        if trust_level > self._player_trust:
            self.goal_engine.on_trust_gained(trust_level - self._player_trust)
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

        for node_id, activation in activated_nodes.items():
            if activation < 0.1:
                continue
            tags = node_emotion_tags.get(node_id, [])
            for tag in tags:
                emotion_totals[tag] = emotion_totals.get(tag, 0.0) + activation

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
# STUB IMPLEMENTATIONS
# Replace these with your actual ChromaDB / NetworkX systems
# ─────────────────────────────────────────────

def _stub_memory_retriever(npc_id: str, query: str) -> dict[str, float]:
    """
    Stub: Replace with ChromaDB semantic retrieval.
    Returns: node_id -> relevance_score
    """
    logger.warning("[Pipeline] Using stub memory retriever — replace with ChromaDB")
    # Example stub returns for Morgan
    if "kara" in query.lower():
        return {
            "childhood_with_kara": 0.85,
            "last_good_night": 0.7,
            "karas_body": 0.6,
        }
    elif "night" in query.lower() or "happened" in query.lower():
        return {
            "the_night_it_happened": 0.9,
            "river_location": 0.7,
        }
    return {}


def _stub_activation_spreader(seed_nodes: list[str], graph) -> dict[str, float]:
    """
    Stub: Replace with NetworkX spreading activation.
    Returns: node_id -> activation_score
    """
    logger.warning("[Pipeline] Using stub activation spreader — replace with NetworkX")
    activated = {}
    # Simple stub: seed nodes activate neighbors (hardcoded for demo)
    neighbor_map = {
        "childhood_with_kara": ["last_good_night", "self_as_protector", "connection_memory"],
        "karas_body":          ["river_location", "the_night_it_happened", "helplessness_memory"],
        "the_night_it_happened": ["karas_body", "the_knife", "helplessness_memory", "crying_alone"],
        "river_location":      ["karas_body", "the_night_it_happened"],
        "last_good_night":     ["childhood_with_kara", "grief_for_kara"],
    }
    for node in seed_nodes:
        activated[node] = 1.0
        for neighbor in neighbor_map.get(node, []):
            activated[neighbor] = max(activated.get(neighbor, 0.0), 0.55)
    return activated


def _default_llm_caller(system_prompt: str, conversation_history: list[dict]) -> str:
    """
    Default LLM caller stub. Replace with your actual LLM integration.
    """
    logger.warning("[Pipeline] Using stub LLM caller — replace with actual LLM")
    return "[LLM response would appear here]"
