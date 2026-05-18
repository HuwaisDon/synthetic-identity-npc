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
from typing import Optional

from schemas.cognitive_schemas import (
    SelfConceptState, SelfConceptThreat, EmotionalState, EmotionType, GoalState
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# MORGAN'S CORE IDENTITY CLAIMS
# These are what her self-concept REQUIRES to be true.
# Threats to these claims activate the defense system.
# ─────────────────────────────────────────────

MORGAN_IDENTITY_CLAIMS = [
    "I am a survivor who made the only choices available to me.",
    "I protected myself because no one else would.",
    "I am not responsible for what happened to Kara — she made her own choices.",
    "I am capable of love, even if I have done terrible things.",
    "I am still a good person at my core.",
    "What happened was necessary, not monstrous.",
]

# Map: memory node patterns that threaten specific identity claims
# Each entry: (node_id_pattern, threatened_claim_index, threat_type, defense)
THREAT_MAP = [
    ("helplessness_memory",  1, "helplessness",    "reframe"),
    ("begging_scene",        0, "helplessness",    "minimize"),
    ("crying_alone",         0, "helplessness",    "externalize_blame"),
    ("karas_body",           2, "moral_failure",   "mythologize"),
    ("river_location",       5, "moral_failure",   "deny"),
    ("the_knife",            5, "moral_failure",   "intellectualize"),
    ("the_night_it_happened",4, "moral_failure",   "reframe"),
    ("witness_old_maret",    2, "shame",           "minimize"),
    ("last_good_night",      3, "grief_for_self",  "reframe"),
    ("childhood_with_kara",  3, "loss",            "mythologize"),
    ("self_as_protector",    4, "contradiction",   "externalize_blame"),
]

# Reframe narratives: what Morgan 'tells herself' instead
REFRAME_LIBRARY = {
    "helplessness": [
        "I was trapped. Anyone would have done the same.",
        "There was no path that didn't end in loss.",
        "I did everything I could with what I had.",
    ],
    "moral_failure": [
        "It was her or me. That's not cruelty — that's survival.",
        "People who weren't there can't judge what I had to do.",
        "What looks monstrous from outside was the only door left open.",
    ],
    "shame": [
        "Old Maret never understood what was at stake.",
        "They didn't see what I was up against.",
        "Judgment is easy when you weren't the one drowning.",
    ],
    "grief_for_self": [
        "Grief is a luxury. I can't afford it.",
        "She's gone. The only thing that matters now is staying alive.",
        "Missing her doesn't undo anything.",
    ],
    "loss": [
        "What we had died before that night. I grieve the person she was, not what she became.",
        "She chose what she became. I chose to survive.",
    ],
    "contradiction": [
        "Protecting yourself isn't a contradiction with caring. They coexist.",
        "The same night you hold someone and let them go can be real.",
    ],
}


class SelfConceptDefenseSystem:
    """
    Evaluates activated memory nodes for identity threats.
    Produces defense instructions that feed into suppression and narrative drift.
    """

    def __init__(
        self,
        identity_claims: Optional[list[str]] = None,
        threat_map: Optional[list[tuple]] = None,
    ):
        self.identity_claims = identity_claims or MORGAN_IDENTITY_CLAIMS
        self.threat_map = threat_map or THREAT_MAP
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

        for node_id, activation in activated_nodes.items():
            if activation < 0.2:
                continue  # sub-threshold activations don't threaten self-concept

            threats = self._identify_threats(node_id, activation, emotional_state)
            for threat in threats:
                active_threats.append(threat)
                if threat.defense_mechanism not in active_defenses:
                    active_defenses.append(threat.defense_mechanism)
                claim = self.identity_claims[threat.threatening_node.count("_") % len(self.identity_claims)]
                if claim not in threatened_claims:
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
    ) -> list[SelfConceptThreat]:
        """
        Check if this node pattern matches any identity threat.
        """
        threats = []
        for (pattern, claim_idx, threat_type, defense) in self.threat_map:
            if pattern in node_id or node_id in pattern:
                # Threat intensity = activation * emotional amplifier
                amplifier = 1.0
                dominant = emotional_state.dominant
                if dominant:
                    if dominant.emotion in (EmotionType.SHAME, EmotionType.GUILT):
                        amplifier = 1.0 + dominant.intensity * 0.5
                    elif dominant.emotion == EmotionType.FEAR:
                        amplifier = 1.0 + dominant.intensity * 0.3

                threat_intensity = min(1.0, activation * amplifier)

                # Select reframe narrative
                reframe = self._select_reframe(threat_type)

                threat = SelfConceptThreat(
                    threatening_node=node_id,
                    threat_type=threat_type,
                    threat_intensity=threat_intensity,
                    defense_mechanism=defense,
                    reframe_narrative=reframe,
                )
                threats.append(threat)

        return threats

    def _select_reframe(self, threat_type: str) -> Optional[str]:
        """Select the most appropriate reframe narrative for this threat type."""
        options = REFRAME_LIBRARY.get(threat_type, [])
        if not options:
            return None
        # Simple selection: use turn count to vary across options
        return options[self._turn_count % len(options)]

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
        system.state.core_identity_claims = data.get("core_identity_claims", MORGAN_IDENTITY_CLAIMS)
        system._turn_count = data.get("turn_count", 0)
        return system
