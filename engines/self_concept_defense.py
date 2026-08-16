"""
engines/self_concept_defense.py

SelfConceptDefenseSystem: Identity Coherence Protection

Psychological rationale:
  The self-concept is not just a collection of facts about oneself.
  It is an organizing narrative structure that must remain coherent
  for the person to function. Threats to the self-concept activate
  automatic psychological defenses — not as conscious choices,
  but as cognitive reflexes.

  Morgan does not 'decide' to blame Kara for what happened.
  Her identity system detects that 'I chose to do nothing' threatens
  her self-narrative as a survivor who did everything possible,
  and the defense system reframes the memory before it reaches
  conscious articulation.

  Core defenses (after Anna Freud / contemporary emotion regulation research):
  - Reframing: alter interpretation without changing facts
  - Externalization: attribute cause/responsibility to external factors
  - Minimization: reduce the significance of the threatening event
  - Mythologization: elevate the event to tragic necessity
  - Denial: block the threatening interpretation entirely
  - Intellectualization: detach emotional content from factual description

Architecture note:
  This system operates on ACTIVATED MEMORY NODES.
  It evaluates whether activated content threatens core identity claims.
  If so, it produces defense instructions that modify:
    - Narrative drift (what the NPC believes about the event)
    - Suppression instructions (what specifically not to say)
    - Reframe narratives (alternative interpretations to offer)
"""

from __future__ import annotations
import logging
from typing import Any, Optional

from schemas.cognitive_schemas import (
    SelfConceptState, SelfConceptThreat, EmotionalState, EmotionType, GoalState
)
from schemas.character_schema import ThreatRule

logger = logging.getLogger(__name__)


class SelfConceptDefenseSystem:
    """
    Evaluates activated memory nodes for identity threats.
    Produces defense instructions that feed into suppression and narrative drift.
    """

    def __init__(
        self,
        identity_claims: Optional[list[str]] = None,
        threat_rules: Optional[list[ThreatRule]] = None,
    ):
        self.identity_claims = identity_claims or []
        self.threat_rules = threat_rules or []
        self.state = SelfConceptState(
            coherence=1.0,
            core_identity_claims=list(self.identity_claims),
        )
        self._turn_count = 0

    # ── MAIN EVALUATION ──────────────────────

    def evaluate(
        self,
        activated_nodes: dict[str, float],
        emotional_state: EmotionalState,
        goal_state: GoalState,
    ) -> SelfConceptState:
        """
        Evaluate activated memory nodes for self-concept threats.
        Returns updated SelfConceptState with active defenses.
        
        Args:
            activated_nodes: node_id -> activation_score
            emotional_state: current emotional state
            goal_state: current goals (identity goal urgency matters)
        
        Returns:
            SelfConceptState with detected threats and defense instructions
        """
        self._turn_count += 1

        active_threats: list[SelfConceptThreat] = []
        active_defenses: list[str] = []
        threatened_claims: list[str] = []

        for node_id, activation_source in activated_nodes.items():
            activation = self._coerce_activation(activation_source)
            if activation < 0.2:
                continue  # sub-threshold activations don't threaten self-concept

            threats = self._identify_threats(node_id, activation, emotional_state, activation_source)
            for threat in threats:
                active_threats.append(threat)
                if threat.defense_mechanism not in active_defenses:
                    active_defenses.append(threat.defense_mechanism)
                claim = self._identity_claim_for_threat(threat)
                if claim and claim not in threatened_claims:
                    threatened_claims.append(claim)

        # Coherence: degrades with high-intensity threats
        if active_threats:
            threat_load = sum(t.threat_intensity for t in active_threats)
            self.state.coherence = max(0.0, 1.0 - (threat_load * 0.15))
        else:
            # Coherence recovers slowly when not threatened
            self.state.coherence = min(1.0, self.state.coherence + 0.05)

        # Shame amplifies identity defense urgency
        dominant = emotional_state.dominant
        if dominant and dominant.emotion in (EmotionType.SHAME, EmotionType.GUILT):
            self.state.coherence = max(0.0, self.state.coherence - 0.1 * dominant.intensity)

        self.state.active_threats = active_threats
        self.state.active_defenses = active_defenses
        self.state.threatened_claims = threatened_claims

        if active_threats:
            logger.debug(
                f"[SelfConcept] coherence={self.state.coherence:.2f}, "
                f"threats={len(active_threats)}, defenses={active_defenses}"
            )

        return self.state

    # ── THREAT IDENTIFICATION ─────────────────

    def _identify_threats(
        self,
        node_id: str,
        activation: float,
        emotional_state: EmotionalState,
        activation_context: Any | None = None,
    ) -> list[SelfConceptThreat]:
        """
        Check if this node pattern matches any identity threat.
        """
        threats = []
        for rule in self.threat_rules:
            match = self._matches_rule(activation_context or node_id, rule)
            if not match:
                continue

            amplifier = 1.0
            dominant = emotional_state.dominant
            if dominant:
                if dominant.emotion in (EmotionType.SHAME, EmotionType.GUILT):
                    amplifier = 1.0 + dominant.intensity * 0.5
                elif dominant.emotion == EmotionType.FEAR:
                    amplifier = 1.0 + dominant.intensity * 0.3

            threat_intensity = min(1.0, activation * amplifier)
            reframe = self._select_reframe(rule.threat_type)

            threat = SelfConceptThreat(
                threatening_node=node_id,
                claim_idx=rule.threatened_claim_idx,
                threat_type=rule.threat_type,
                threat_intensity=threat_intensity,
                defense_mechanism=rule.defense_mechanism,
                reframe_narrative=reframe,
            )
            threats.append(threat)

        return threats

    def _matches_rule(self, node_or_context: Any, rule: ThreatRule) -> bool:
        if not rule.trigger_concepts:
            return False

        normalized_terms = {
            term.lower()
            for term in self._extract_match_terms(node_or_context)
            if term
        }
        normalized_triggers = {
            concept.lower()
            for concept in rule.trigger_concepts
            if concept
        }
        return bool(normalized_terms & normalized_triggers)

    def _extract_match_terms(self, node_or_context: Any) -> list[str]:
        if isinstance(node_or_context, str):
            return [node_or_context]

        if isinstance(node_or_context, dict):
            payload = node_or_context
        else:
            payload = getattr(node_or_context, "metadata", {}) or {}

        terms: list[str] = []
        for key in ("associated_concepts", "concepts", "concept"):
            value = payload.get(key, []) if isinstance(payload, dict) else None
            if isinstance(value, str):
                terms.extend([part.strip() for part in value.split(",") if part.strip()])
            elif isinstance(value, (list, tuple, set)):
                terms.extend(str(item) for item in value)

        if isinstance(payload, dict):
            for key in ("sensory_tags", "tags"):
                value = payload.get(key, [])
                if isinstance(value, str):
                    terms.extend([part.strip() for part in value.split(",") if part.strip()])
                elif isinstance(value, (list, tuple, set)):
                    terms.extend(str(item) for item in value)

        if not terms and hasattr(node_or_context, "associated_concepts"):
            terms.extend(str(item) for item in node_or_context.associated_concepts)
        if not terms and hasattr(node_or_context, "sensory_tags"):
            terms.extend(str(item) for item in node_or_context.sensory_tags)
        return terms

    def _coerce_activation(self, activation_source: Any) -> float:
        if isinstance(activation_source, dict):
            return float(activation_source.get("activation", 0.0))
        return float(activation_source)

    def _identity_claim_for_threat(
        self,
        threat: SelfConceptThreat,
    ) -> Optional[str]:
        """Return the authored identity claim for a detected threat."""
        if 0 <= threat.claim_idx < len(self.identity_claims):
            return self.identity_claims[threat.claim_idx]

        logger.warning(
            "[SelfConcept] threat %s has invalid claim_idx=%s",
            threat.threatening_node,
            threat.claim_idx,
        )
        return None

    def _select_reframe(self, threat_type: str) -> Optional[str]:
        """Select the most appropriate reframe narrative for this threat type."""
        return None

    # ── DEFENSE INSTRUCTIONS FOR PIPELINE ────

    def get_suppression_instructions(self) -> list[str]:
        """
        Returns node IDs that the self-concept defense system wants suppressed.
        These feed into the suppression system as additional suppression pressure.
        """
        return [t.threatening_node for t in self.state.active_threats if t.threat_intensity > 0.4]

    def get_active_reframe(self) -> Optional[str]:
        """
        Returns the most salient reframe narrative active this turn.
        This is the 'story Morgan is telling herself' — may bleed into language.
        """
        if not self.state.active_threats:
            return None
        # Return the reframe for the most intense threat
        most_intense = max(self.state.active_threats, key=lambda t: t.threat_intensity)
        return most_intense.reframe_narrative

    def get_defense_descriptor(self) -> Optional[str]:
        """
        Returns the primary defense mechanism active this turn.
        Used by CognitiveSummarizer to characterize response style.
        """
        if not self.state.active_defenses:
            return None
        return self.state.active_defenses[0]

    def get_defended_identity_claim(self) -> Optional[str]:
        """
        Returns the identity claim currently being defended.
        Feeds into LLM prompt as the 'belief Morgan is protecting.'
        """
        if not self.state.threatened_claims:
            return None
        return self.state.threatened_claims[0]

    # ── SERIALIZATION ────────────────────────

    def serialize(self) -> dict:
        return {
            "coherence": self.state.coherence,
            "core_identity_claims": self.state.core_identity_claims,
            "turn_count": self._turn_count,
            # Active threats are turn-scoped, don't persist
        }

    @classmethod
    def deserialize(cls, data: dict) -> "SelfConceptDefenseSystem":
        system = cls()
        system.state.coherence = data.get("coherence", 1.0)
        system.state.core_identity_claims = data.get("core_identity_claims", [])
        system._turn_count = data.get("turn_count", 0)
        return system
