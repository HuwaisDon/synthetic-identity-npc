"""
engines/attention_engine.py

AttentionEngine: Attentional Bandwidth, Salience Competition, Cognitive Narrowing

Psychological rationale:
  The mind cannot process everything simultaneously. Under stress, attention
  narrows to threat-relevant content — a form of cognitive triage.
  Under dissociation, attention flattens: everything feels equidistant.
  
  Attention determines which activated memories become 'dominant' —
  present in working cognition and potentially expressed —
  versus which remain subconscious, expressing only as behavioral pressure.

  This is not censorship. It is the bottleneck between activation and awareness.

Architecture note:
  AttentionEngine receives the output of spreading activation (node activations)
  and the current emotional state, and returns a filtered set of 'focal nodes'
  plus the attentional state descriptor.
  
  It does NOT modify the memory graph. It only determines what reaches awareness.
"""

from __future__ import annotations
import logging
import math
from typing import Optional

from schemas.cognitive_schemas import (
    AttentionState, AttentionalState, EmotionalState, EmotionType, GoalState
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# NODE SALIENCE RECORD
# ─────────────────────────────────────────────

class NodeSalience:
    """
    Tracks the salience score for a single memory node.
    Salience = how much cognitive attention this node is capturing.
    """
    def __init__(self, node_id: str, activation: float):
        self.node_id = node_id
        self.base_activation = activation
        self.salience_score = activation
        self.notes: list[str] = []

    def boost(self, amount: float, reason: str) -> None:
        self.salience_score = min(1.0, self.salience_score + amount)
        self.notes.append(f"+{amount:.2f} [{reason}]")

    def dampen(self, amount: float, reason: str) -> None:
        self.salience_score = max(0.0, self.salience_score - amount)
        self.notes.append(f"-{amount:.2f} [{reason}]")

    def __repr__(self):
        return f"NodeSalience({self.node_id}, score={self.salience_score:.3f})"


# ─────────────────────────────────────────────
# ATTENTION ENGINE
# ─────────────────────────────────────────────

class AttentionEngine:
    """
    Filters activated memory nodes through attentional bandwidth.
    Returns AttentionState describing what reaches working cognition.
    """

    # How many nodes can be in focal attention simultaneously
    # Psychological basis: working memory holds ~4 ± 1 items (Cowan, 2001)
    DEFAULT_FOCAL_CAPACITY = 4

    # Maximum nodes in attentional field (peripheral awareness)
    DEFAULT_FIELD_CAPACITY = 9

    def __init__(
        self,
        focal_capacity: int = DEFAULT_FOCAL_CAPACITY,
        field_capacity: int = DEFAULT_FIELD_CAPACITY,
    ):
        self.focal_capacity = focal_capacity
        self.field_capacity = field_capacity

    def process(
        self,
        activated_nodes: dict[str, float],  # node_id -> activation_score
        emotional_state: EmotionalState,
        goal_state: GoalState,
        node_emotion_tags: Optional[dict[str, list[str]]] = None,  # node_id -> [emotion_labels]
        prior_attention_state: Optional[AttentionState] = None,
    ) -> AttentionState:
        """
        Process activated nodes through attentional bandwidth.
        
        Args:
            activated_nodes: all currently activated memory nodes with scores
            emotional_state: current emotional profile (affects bandwidth + salience bias)
            goal_state: current goals (dominant goal boosts goal-relevant nodes)
            node_emotion_tags: emotion labels per node (for emotion-biased salience)
            prior_attention_state: previous turn's attention (attention has momentum)
        
        Returns:
            AttentionState with focal_nodes, suppressed_from_attention, state descriptor
        """
        if not activated_nodes:
            return AttentionState(state=AttentionalState.OPEN, bandwidth=1.0)

        # Step 1: Determine attentional state and bandwidth
        att_state, bandwidth = self._determine_bandwidth(emotional_state)

        # Step 2: Compute salience scores for all activated nodes
        saliences = self._compute_saliences(
            activated_nodes, emotional_state, goal_state, node_emotion_tags
        )

        # Step 3: Apply attention momentum from prior state
        if prior_attention_state:
            saliences = self._apply_momentum(saliences, prior_attention_state)

        # Step 4: Apply bandwidth constraint — shrink field under stress
        available_focal = max(1, int(self.focal_capacity * bandwidth))
        available_field = max(2, int(self.field_capacity * bandwidth))

        # Step 5: Rank and select focal vs. peripheral vs. suppressed-from-attention
        ranked = sorted(saliences, key=lambda s: s.salience_score, reverse=True)

        focal_nodes = [s.node_id for s in ranked[:available_focal]]
        peripheral_nodes = [s.node_id for s in ranked[available_focal:available_field]]
        suppressed_from_attention = [s.node_id for s in ranked[available_field:]]

        # Step 6: Cognitive load
        cognitive_load = self._compute_cognitive_load(emotional_state, len(activated_nodes))

        # Determine salience bias
        salience_bias = self._determine_salience_bias(emotional_state)

        result = AttentionState(
            state=att_state,
            bandwidth=bandwidth,
            focal_nodes=focal_nodes,
            suppressed_from_attention=suppressed_from_attention,
            salience_bias=salience_bias,
            cognitive_load=cognitive_load,
        )

        logger.debug(
            f"[AttentionEngine] state={att_state.value}, bandwidth={bandwidth:.2f}, "
            f"focal={focal_nodes}, load={cognitive_load:.2f}"
        )

        return result

    # ── BANDWIDTH ────────────────────────────

    def _determine_bandwidth(
        self, emotional_state: EmotionalState
    ) -> tuple[AttentionalState, float]:
        """
        Emotional state determines attentional bandwidth.
        
        Fear: narrows but sharpens — high salience discrimination, lower breadth
        Grief: diffuses attention — wandering, easily lost
        Flooding: overwhelmed — almost no working cognition available
        Dissociation: flat — bandwidth present but indiscriminate
        """
        dominant = emotional_state.dominant
        arousal = emotional_state.arousal

        if arousal > 0.85:
            return AttentionalState.FLOODED, 0.25

        if not dominant:
            return AttentionalState.OPEN, 1.0

        em = dominant.emotion
        intensity = dominant.intensity

        if em == EmotionType.FEAR:
            bandwidth = 1.0 - (0.5 * intensity)   # narrows significantly
            state = AttentionalState.STRESSED if intensity > 0.4 else AttentionalState.FOCUSED
            return state, max(0.3, bandwidth)

        elif em == EmotionType.GRIEF:
            bandwidth = 1.0 - (0.2 * intensity)   # mild narrowing, more diffuse
            return AttentionalState.OPEN, max(0.6, bandwidth)

        elif em == EmotionType.SHAME:
            bandwidth = 1.0 - (0.4 * intensity)   # self-focused narrowing
            return AttentionalState.FOCUSED, max(0.4, bandwidth)

        elif em == EmotionType.NUMBNESS:
            return AttentionalState.DISSOCIATED, 0.5

        elif em == EmotionType.DREAD:
            bandwidth = 1.0 - (0.45 * intensity)
            return AttentionalState.STRESSED, max(0.35, bandwidth)

        elif em == EmotionType.ANGER:
            bandwidth = 1.0 - (0.3 * intensity)
            return AttentionalState.FOCUSED, max(0.5, bandwidth)

        return AttentionalState.OPEN, 1.0

    # ── SALIENCE COMPUTATION ─────────────────

    def _compute_saliences(
        self,
        activated_nodes: dict[str, float],
        emotional_state: EmotionalState,
        goal_state: GoalState,
        node_emotion_tags: Optional[dict[str, list[str]]],
    ) -> list[NodeSalience]:
        """
        Compute per-node salience scores.
        
        Salience = base activation
                 + emotional congruence boost (fear-tagged nodes salient under fear)
                 + goal relevance boost (goal-linked nodes become focal)
                 - goal blocking dampen (concealment-blocked nodes lose salience)
        """
        saliences = []
        dominant_emotion = emotional_state.dominant

        for node_id, activation in activated_nodes.items():
            ns = NodeSalience(node_id, activation)

            # Emotional congruence: nodes matching dominant emotion become more salient
            if dominant_emotion and node_emotion_tags:
                tags = node_emotion_tags.get(node_id, [])
                if dominant_emotion.emotion.value in tags:
                    boost = 0.2 * dominant_emotion.intensity
                    ns.boost(boost, f"emotion_congruence:{dominant_emotion.emotion.value}")

            # Goal salience: nodes linked to active high-utility goals
            if goal_state.active_goals:
                for g in goal_state.active_goals:
                    if g.active and g.utility > 0.5 and node_id in g.linked_memory_nodes:
                        ns.boost(0.15 * g.utility, f"goal_link:{g.goal_id}")
                    # Strategic silence nodes lose attentional pull
                    # (they still activate subconsciously but don't surface)
                    if node_id in goal_state.strategic_silence:
                        ns.dampen(0.25, "strategic_silence")

            saliences.append(ns)

        return saliences

    def _apply_momentum(
        self,
        saliences: list[NodeSalience],
        prior_state: AttentionState
    ) -> list[NodeSalience]:
        """
        Attention has inertia. Nodes that were focal last turn get a small
        persistence boost — mirroring how humans return to recent thoughts.
        """
        prior_focal_set = set(prior_state.focal_nodes)
        for ns in saliences:
            if ns.node_id in prior_focal_set:
                ns.boost(0.08, "attention_momentum")
        return saliences

    # ── SALIENCE BIAS ────────────────────────

    def _determine_salience_bias(
        self, emotional_state: EmotionalState
    ) -> Optional[EmotionType]:
        """Which emotion is currently steering attentional salience?"""
        if emotional_state.dominant and emotional_state.dominant.intensity > 0.3:
            return emotional_state.dominant.emotion
        return None

    # ── COGNITIVE LOAD ───────────────────────

    def _compute_cognitive_load(
        self, emotional_state: EmotionalState, n_activated: int
    ) -> float:
        """
        Cognitive load is a function of:
        - Arousal level
        - Number of competing activated nodes
        - Conflict between emotional valence and arousal
        
        High load = overwhelmed, incoherent, fragmented speech
        """
        arousal_load = emotional_state.arousal * 0.5
        volume_load = min(0.5, n_activated / 20.0)
        # High arousal + negative valence = extra load
        valence_penalty = max(0.0, -emotional_state.valence) * emotional_state.arousal * 0.2
        return min(1.0, arousal_load + volume_load + valence_penalty)
