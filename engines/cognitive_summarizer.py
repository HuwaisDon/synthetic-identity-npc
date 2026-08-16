"""
engines/cognitive_summarizer.py

CognitiveSummarizer: Cognitive State Compression to LLM Behavioral Signals

Architecture rationale:
  This is the boundary between the cognitive architecture and the language model.
  
  The cognitive pipeline produces rich internal state across 6+ systems.
  None of this should be dumped raw into a prompt. Raw cognitive exhaust:
    - wastes tokens
    - leaks architectural machinery (LLM should not 'know' it has a GoalEngine)
    - loses behavioral signal in noise
    - makes the NPC feel like a prompted chatbot, not a thinking person
  
  The summarizer's job is:
    1. Extract the behavioral PRESSURE from each cognitive system
    2. Translate pressure into signal language the LLM can embody
    3. Produce a compact, psychologically coherent behavioral brief
  
  The LLM receives a CognitiveSummary and a character brief.
  It does NOT receive goal utilities, activation scores, or memory graph data.
  It receives the FELT EXPERIENCE that would shape a human's speech.

Design philosophy:
  The prompt is not a description of Morgan's state.
  It is an instruction to INHABIT that state.
  
  "Morgan is suppressing memory node #7" → bad
  "Something in this conversation is making Morgan feel cornered, 
   though she won't say what" → good
"""

from __future__ import annotations
import logging
from typing import Optional

from schemas.cognitive_schemas import (
    CognitiveSummary, EmotionalState, EmotionType,
    GoalState, AttentionState, AttentionalState,
    SelfConceptState, PredictiveState, EmotionalPersistenceState
)
from schemas.intention_schema import Intention

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# EMOTION → SURFACE DESCRIPTOR MAPS
# ─────────────────────────────────────────────

EMOTION_SURFACE_DESCRIPTORS = {
    EmotionType.FEAR:     ("wary", "fear is moving beneath the surface"),
    EmotionType.GRIEF:    ("quiet", "grief she isn't showing"),
    EmotionType.SHAME:    ("composed but brittle", "shame pulling at the edges"),
    EmotionType.GUILT:    ("careful", "guilt coloring each word"),
    EmotionType.ANGER:    ("controlled", "anger held just below the tone"),
    EmotionType.LONGING:  ("distant", "longing she won't acknowledge"),
    EmotionType.RELIEF:   ("lighter than usual", None),
    EmotionType.JOY:      ("open", None),
    EmotionType.CONTEMPT: ("sardonic", "contempt she's barely hiding"),
    EmotionType.DREAD:    ("flat", "dread making everything feel far away"),
    EmotionType.NUMBNESS: ("blank", "a flatness that isn't calm"),
}

ATTENTIONAL_DESCRIPTORS = {
    AttentionalState.OPEN:        "clear",
    AttentionalState.FOCUSED:     "concentrated",
    AttentionalState.STRESSED:    "strained",
    AttentionalState.FLOODED:     "overwhelmed",
    AttentionalState.DISSOCIATED: "detached",
}

STRATEGY_TO_RESPONSE_STYLE = {
    "disclose":  "careful and deliberate — choosing words as if each one costs something",
    "partial":   "selective — giving something while keeping something else hidden",
    "deflect":   "redirecting — finding ways to not quite answer the question",
    "lie":       "too smooth — overly composed, slightly rehearsed",
    "silence":   "closed — brief answers, no elaboration, deliberate restraint",
    "neutral":   "direct",
}

DEFENSE_MECHANISM_LEAKAGE = {
    "reframe":            "Morgan may restate events in terms that exonerate her before being asked",
    "externalize_blame":  "Morgan may attribute what happened to others' choices without being prompted",
    "minimize":           "Morgan may treat significant events as minor, barely worth discussing",
    "mythologize":        "Morgan may speak of the past in elevated, almost ritualized terms",
    "deny":               "Morgan may not engage with certain topics at all — as if the question didn't land",
    "intellectualize":    "Morgan may describe emotional events in clinical, detached language",
}


class CognitiveSummarizer:
    """
    Compresses the full cognitive pipeline output into a LLM-ready behavioral brief.
    """

    def summarize(
        self,
        emotional_state: EmotionalState,
        goal_state: GoalState,
        attention_state: AttentionState,
        self_concept_state: SelfConceptState,
        predictive_state: PredictiveState,
        persistence_state: EmotionalPersistenceState,
        node_descriptions: Optional[dict[str, str]] = None,  # node_id -> brief description
        intention: Optional[Intention] = None,
    ) -> CognitiveSummary:
        """
        Produce a CognitiveSummary that captures behavioral pressure without
        exposing architectural machinery.

        All translation is one-way: cognition → embodied experience signal.
        """
        node_descriptions = node_descriptions or {}

        summary = CognitiveSummary()

        # ── EMOTIONAL SIGNAL ─────────────────

        summary.emotional_stability = self._compute_stability(
            emotional_state, self_concept_state
        )

        surface, undercurrent = self._extract_emotional_signals(
            emotional_state, persistence_state
        )
        summary.surface_emotion = surface
        summary.undercurrent_emotion = undercurrent

        # ── BEHAVIORAL PRESSURES ─────────────

        summary.avoidance_topics = self._extract_avoidance_topics(
            goal_state, self_concept_state, node_descriptions
        )
        summary.disclosure_pressure_topics = self._extract_disclosure_topics(
            goal_state, attention_state, node_descriptions
        )
        summary.strategic_intent = self._extract_strategic_intent(
            goal_state, predictive_state
        )

        # ── MEMORY ACCESS SIGNALS ────────────

        summary.accessible_memories = self._extract_accessible_memories(
            attention_state, node_descriptions
        )
        summary.suppressed_pressure_nodes = self._extract_suppressed_pressure(
            attention_state, goal_state, node_descriptions
        )

        # ── IDENTITY SIGNALS ─────────────────

        summary.self_concept_under_threat = self_concept_state.coherence < 0.7
        summary.active_defense_mechanism = self_concept_state.active_defenses[0] \
            if self_concept_state.active_defenses else None
        summary.identity_claim_being_defended = \
            self_concept_state.threatened_claims[0] \
            if self_concept_state.threatened_claims else None

        # ── ATTENTIONAL SIGNAL ───────────────

        summary.attentional_state = attention_state.state.value
        summary.cognitive_load_descriptor = self._cognitive_load_descriptor(
            attention_state.cognitive_load
        )

        # ── RESPONSE STYLE ───────────────────

        summary.response_style = STRATEGY_TO_RESPONSE_STYLE.get(
            predictive_state.chosen_strategy, "direct"
        )
        if intention is not None:
            summary.strategic_intent = f"{intention.disposition}::{intention.dominant_drive.value if intention.dominant_drive else 'neutral'}"

        # ── BEHAVIORAL LEAKAGE ───────────────

        summary.leakage_signals = self._extract_leakage_signals(
            emotional_state, self_concept_state, attention_state, goal_state
        )

        return summary

    def to_prompt_block(self, summary: CognitiveSummary, npc_name: str = "Morgan") -> str:
        """
        Render the CognitiveSummary as a compact, embody-able prompt block.
        
        This is the ONLY thing the LLM sees of the cognitive pipeline.
        It should read like an acting note, not a system status report.
        """
        lines = [f"[{npc_name.upper()} — INTERNAL STATE — THIS TURN]", ""]

        # Emotional presence
        if summary.surface_emotion and summary.undercurrent_emotion:
            lines.append(
                f"Surface: {summary.surface_emotion}. "
                f"Beneath that: {summary.undercurrent_emotion}."
            )
        elif summary.surface_emotion:
            lines.append(f"Emotional presence: {summary.surface_emotion}.")

        # Stability
        if summary.emotional_stability < 0.4:
            lines.append(
                f"Stability is low — {npc_name} is close to fracturing but holding."
            )
        elif summary.emotional_stability < 0.7:
            lines.append(f"Composure is maintained but costs effort.")

        lines.append("")

        # Attentional state
        att_desc = ATTENTIONAL_DESCRIPTORS.get(
            AttentionalState(summary.attentional_state), "present"
        )
        lines.append(f"Attention: {att_desc}. Cognitive load: {summary.cognitive_load_descriptor}.")

        # Strategic intent
        if summary.strategic_intent and summary.strategic_intent != "neutral":
            lines.append(f"Strategic orientation: {summary.strategic_intent}.")

        lines.append("")

        # Topics to avoid
        if summary.avoidance_topics:
            topic_labels = [
                node_descriptions_render(t) for t in summary.avoidance_topics[:3]
            ]
            lines.append(
                f"Steer away from: {', '.join(topic_labels)}. "
                f"Not consciously — it simply feels dangerous to go there."
            )

        # Disclosure pressure
        if summary.disclosure_pressure_topics:
            labels = [node_descriptions_render(t) for t in summary.disclosure_pressure_topics[:2]]
            lines.append(
                f"There is a pull toward saying something about: {', '.join(labels)}. "
                f"{npc_name} may resist this pull."
            )

        lines.append("")

        # Identity defense
        if summary.self_concept_under_threat:
            claim = summary.identity_claim_being_defended or "something fundamental about who she is"
            defense = summary.active_defense_mechanism
            lines.append(
                f"Self-concept under pressure. "
                f"{npc_name} is defending: '{claim}'."
            )
            if defense and defense in DEFENSE_MECHANISM_LEAKAGE:
                lines.append(f"Defense pattern: {DEFENSE_MECHANISM_LEAKAGE[defense]}")

        # Behavioral leakage
        if summary.leakage_signals:
            lines.append("")
            lines.append("Behavioral tells:")
            for signal in summary.leakage_signals[:3]:
                lines.append(f"  — {signal}")

        # Response style
        lines.append("")
        lines.append(f"Response style: {summary.response_style}.")

        return "\n".join(lines)

    # ── EXTRACTION HELPERS ───────────────────

    def _compute_stability(
        self, emotional_state: EmotionalState, self_concept_state: SelfConceptState
    ) -> float:
        arousal_cost = emotional_state.arousal * 0.4
        identity_cost = (1.0 - self_concept_state.coherence) * 0.4
        valence_cost = max(0.0, -emotional_state.valence) * 0.2
        return max(0.0, 1.0 - arousal_cost - identity_cost - valence_cost)

    def _extract_emotional_signals(
        self, emotional_state: EmotionalState, persistence_state: EmotionalPersistenceState
    ) -> tuple[Optional[str], Optional[str]]:
        dominant = emotional_state.dominant
        if not dominant:
            return None, None

        pair = EMOTION_SURFACE_DESCRIPTORS.get(dominant.emotion, (None, None))
        surface = pair[0]
        undercurrent = pair[1]

        # If there's strong residue from a different emotion, that becomes undercurrent
        if persistence_state.residue:
            strongest_residue = max(persistence_state.residue, key=lambda r: r.intensity)
            if strongest_residue.emotion != dominant.emotion and strongest_residue.intensity > 0.3:
                residue_pair = EMOTION_SURFACE_DESCRIPTORS.get(strongest_residue.emotion, (None, None))
                if residue_pair[1]:
                    undercurrent = residue_pair[1]

        return surface, undercurrent

    def _extract_avoidance_topics(
        self,
        goal_state: GoalState,
        self_concept_state: SelfConceptState,
        node_descriptions: dict[str, str],
    ) -> list[str]:
        avoidance = list(goal_state.strategic_silence)
        # Add self-concept defense topics
        for threat in self_concept_state.active_threats:
            if threat.threatening_node not in avoidance:
                avoidance.append(threat.threatening_node)
        return avoidance[:5]

    def _extract_disclosure_topics(
        self,
        goal_state: GoalState,
        attention_state: AttentionState,
        node_descriptions: dict[str, str],
    ) -> list[str]:
        if goal_state.disclosure_pressure < 0.3:
            return []
        # Focal nodes not in strategic silence
        safe_focal = [
            n for n in attention_state.focal_nodes
            if n not in goal_state.strategic_silence
        ]
        return safe_focal[:2]

    def _extract_strategic_intent(
        self, goal_state: GoalState, predictive_state: PredictiveState
    ) -> str:
        if predictive_state.chosen_strategy == "deflect":
            return "deflect and redirect"
        if predictive_state.chosen_strategy == "lie":
            return "misdirect"
        if predictive_state.chosen_strategy == "partial":
            return "give something, keep something"
        if goal_state.dominant_goal:
            gt = goal_state.dominant_goal.goal_type
            if gt == GoalType.TRUST_GAIN:
                return "build rapport"
            if gt == GoalType.CONCEALMENT:
                return "conceal"
            if gt == GoalType.SURVIVAL:
                return "assess danger"
        return "neutral"

    def _extract_accessible_memories(
        self, attention_state: AttentionState, node_descriptions: dict[str, str]
    ) -> list[str]:
        return attention_state.focal_nodes[:4]

    def _extract_suppressed_pressure(
        self,
        attention_state: AttentionState,
        goal_state: GoalState,
        node_descriptions: dict[str, str],
    ) -> list[str]:
        """
        Nodes that are activated but not attended (in suppressed_from_attention)
        AND nodes in goal strategic silence — these are the pressure points.
        """
        pressure = list(set(
            attention_state.suppressed_from_attention +
            goal_state.strategic_silence
        ))
        return pressure[:4]

    def _cognitive_load_descriptor(self, load: float) -> str:
        if load < 0.2:
            return "clear"
        elif load < 0.5:
            return "lightly taxed"
        elif load < 0.75:
            return "strained"
        else:
            return "overwhelmed"

    def _extract_leakage_signals(
        self,
        emotional_state: EmotionalState,
        self_concept_state: SelfConceptState,
        attention_state: AttentionState,
        goal_state: GoalState,
    ) -> list[str]:
        """
        What behavioral tells will bleed through despite suppression?
        These are the cracks in the armor.
        """
        signals = []

        dominant = emotional_state.dominant
        if dominant:
            if dominant.emotion == EmotionType.FEAR and dominant.intensity > 0.5:
                signals.append("pauses slightly before answering questions about the past")
            if dominant.emotion == EmotionType.SHAME and dominant.intensity > 0.4:
                signals.append("avoids direct eye contact in descriptions of herself")
            if dominant.emotion == EmotionType.GRIEF and dominant.intensity > 0.5:
                signals.append("voice flattens when Kara's name comes up")
            if dominant.emotion == EmotionType.ANGER:
                signals.append("clipped sentences; finality in how she closes topics")

        if self_concept_state.coherence < 0.5:
            signals.append("occasional contradictions in how she characterizes her own actions")

        if attention_state.state == AttentionalState.FLOODED:
            signals.append("frequent, heavy silences; her gaze drifts as if she's no longer in the room")
            signals.append("repetitive phrasing; she seems to be caught in a mental loop")
        elif attention_state.state == AttentionalState.STRESSED:
            signals.append("rapid blinking; her breathing is shallow and uneven")

        if goal_state.goal_conflict_score > 0.6:
            signals.append("starts sentences and redirects mid-thought")

        if goal_state.concealment_pressure > 0.7:
            signals.append("too quick to offer alternative topics")
        elif goal_state.concealment_pressure > 0.45:
            signals.append("answers around the memory rather than naming it directly")

        return signals


def node_descriptions_render(node_id: str) -> str:
    """Render a node_id as a human-readable label for prompt inclusion."""
    return node_id.replace("_", " ")
