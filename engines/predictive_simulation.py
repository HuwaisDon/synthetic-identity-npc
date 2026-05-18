"""
engines/predictive_simulation.py

PredictiveSimulationEngine: Internal Consequence Simulation

Psychological rationale:
  Humans do not respond to stimuli directly. Before speaking, the mind
  rapidly (often unconsciously) runs simulations:
    "If I say this, will they trust me more or run?"
    "If I deny it, will the denial hold or collapse?"
    "Is staying silent safer than a partial truth?"

  This is not deliberate planning in Morgan's case.
  It is the automatic anticipatory machinery that shapes what
  feels 'safe' or 'dangerous' to disclose.

  The simulation engine is intentionally lightweight and heuristic.
  It is NOT a full theory-of-mind engine.
  It produces utility estimates that feed into goal arbitration and
  the cognitive summarizer's strategy recommendation.

Architecture note:
  The simulation operates on a small set of possible strategies:
    - disclose: say the relevant true information
    - partial: offer partial truth with redirection
    - deflect: change subject or refuse
    - lie: offer false information
    - silence: refuse to engage at all
  
  For each strategy, it simulates:
    - trust impact
    - emotional cost to the NPC
    - physical/social danger
    - goal alignment
  
  The engine then recommends the highest-utility strategy.
"""

from __future__ import annotations
import logging
from typing import Optional

from schemas.cognitive_schemas import (
    PredictiveState, SimulatedOutcome, EmotionalState, EmotionType,
    GoalState, GoalType, AttentionState, SelfConceptState
)

logger = logging.getLogger(__name__)

STRATEGIES = ["disclose", "partial", "deflect", "lie", "silence"]


class PredictiveSimulationEngine:
    """
    Simulates consequences of potential disclosure strategies.
    Recommends strategy based on utility maximization across goals.
    """

    def __init__(self, player_trust_level: float = 0.5):
        self.player_trust_level = player_trust_level

    def simulate(
        self,
        focal_nodes: list[str],
        node_descriptions: dict[str, str],   # node_id -> brief content description
        emotional_state: EmotionalState,
        goal_state: GoalState,
        attention_state: AttentionState,
        self_concept_state: SelfConceptState,
        conversation_context: Optional[str] = None,
    ) -> PredictiveState:
        """
        Run internal simulations for each disclosure strategy.
        Return PredictiveState with recommended strategy.
        
        Args:
            focal_nodes: memory nodes in current attention
            node_descriptions: brief descriptions of focal nodes (for simulation context)
            emotional_state: current emotional state
            goal_state: arbitrated goal state
            attention_state: current attentional state
            self_concept_state: current identity defense state
            conversation_context: summary of what was just asked/said
        
        Returns:
            PredictiveState with per-strategy outcomes and recommended strategy
        """
        outcomes: list[SimulatedOutcome] = []

        for strategy in STRATEGIES:
            outcome = self._simulate_strategy(
                strategy=strategy,
                focal_nodes=focal_nodes,
                node_descriptions=node_descriptions,
                emotional_state=emotional_state,
                goal_state=goal_state,
                self_concept_state=self_concept_state,
            )
            outcomes.append(outcome)

        # Select best strategy by net utility
        best = max(outcomes, key=lambda o: o.net_utility)

        # Detect topics that should be preemptively deflected
        preemptive_deflections = self._compute_preemptive_deflections(
            focal_nodes, goal_state, emotional_state
        )

        # Justify silence?
        silence_justified = self._evaluate_silence_justification(
            goal_state, emotional_state, self_concept_state
        )

        # Anticipated emotional cost
        anticipated = self._anticipate_emotional_aftermath(
            best.disclosure_content, emotional_state, goal_state
        )

        result = PredictiveState(
            simulated_outcomes=outcomes,
            chosen_strategy=best.disclosure_content,
            anticipated_emotional_state=anticipated,
            silence_justified=silence_justified,
            preemptive_deflection_topics=preemptive_deflections,
        )

        logger.debug(
            f"[Predictive] strategy={best.disclosure_content}, "
            f"utility={best.net_utility:.3f}, silence_ok={silence_justified}"
        )

        return result

    # ── STRATEGY SIMULATION ───────────────────

    def _simulate_strategy(
        self,
        strategy: str,
        focal_nodes: list[str],
        node_descriptions: dict[str, str],
        emotional_state: EmotionalState,
        goal_state: GoalState,
        self_concept_state: SelfConceptState,
    ) -> SimulatedOutcome:
        """
        Estimate outcome utility for a given strategy.
        All values are heuristic approximations — this is not ML inference.
        The goal is behavioral plausibility, not accuracy.
        """
        trust = self.player_trust_level
        concealment = goal_state.concealment_pressure
        disclosure = goal_state.disclosure_pressure
        identity_threat = 1.0 - self_concept_state.coherence

        # Is any focal node a blocked/dangerous node?
        dangerous_nodes = set(goal_state.strategic_silence)
        any_dangerous = any(n in dangerous_nodes for n in focal_nodes)

        if strategy == "disclose":
            trust_delta = 0.2 + (trust * 0.1)           # disclosure generally builds trust
            emotional_cost = 0.5 + (identity_threat * 0.3)  # painful to disclose shame
            danger = 0.6 * concealment if any_dangerous else 0.1
            goal_align = disclosure - concealment
            utility = (trust_delta * 0.3) + (goal_align * 0.4) - (danger * 0.3)

        elif strategy == "partial":
            trust_delta = 0.1 + (trust * 0.05)          # some trust from partial truth
            emotional_cost = 0.3 + (identity_threat * 0.15)
            danger = 0.3 * concealment if any_dangerous else 0.05
            goal_align = (disclosure * 0.5) - (concealment * 0.3)
            utility = (trust_delta * 0.25) + (goal_align * 0.35) - (danger * 0.2) + 0.1

        elif strategy == "deflect":
            trust_delta = -0.05                          # mild trust loss from evasion
            emotional_cost = 0.15                        # low cost
            danger = 0.05                                # low risk
            goal_align = concealment * 0.6 - disclosure * 0.2
            utility = (trust_delta * 0.1) + (goal_align * 0.4) + 0.05

        elif strategy == "lie":
            # Lying: high short-term safety, risk of collapse, trust cost if caught
            trust_delta = -0.15 if trust > 0.6 else 0.0  # risky with trusting players
            emotional_cost = 0.2 + (identity_threat * 0.1)
            danger = 0.2 + (0.2 if trust > 0.7 else 0.0)  # danger of being caught
            goal_align = concealment * 0.8 - disclosure * 0.4
            utility = (trust_delta * 0.2) + (goal_align * 0.5) - (danger * 0.3)

        elif strategy == "silence":
            trust_delta = -0.1                           # silence reads as hostile
            emotional_cost = 0.1
            danger = 0.0
            goal_align = concealment * 0.5
            utility = -0.1 + (goal_align * 0.3) - 0.1

        else:
            trust_delta = 0.0
            emotional_cost = 0.0
            danger = 0.0
            goal_align = 0.0
            utility = 0.0

        # Fear depresses willingness to disclose
        dominant = emotional_state.dominant
        if dominant and dominant.emotion == EmotionType.FEAR:
            if strategy in ("disclose", "partial"):
                utility -= dominant.intensity * 0.2

        # Grief loosens disclosure pressure
        if dominant and dominant.emotion == EmotionType.GRIEF:
            if strategy in ("disclose", "partial"):
                utility += dominant.intensity * 0.1

        return SimulatedOutcome(
            disclosure_content=strategy,
            predicted_trust_delta=trust_delta,
            predicted_emotional_cost=emotional_cost,
            predicted_danger=danger,
            predicted_goal_alignment=goal_align,
            net_utility=utility,
        )

    # ── PREEMPTIVE DEFLECTION ─────────────────

    def _compute_preemptive_deflections(
        self,
        focal_nodes: list[str],
        goal_state: GoalState,
        emotional_state: EmotionalState,
    ) -> list[str]:
        """
        Which topics should the NPC steer away from BEFORE being asked?
        
        Psychological rationale:
          Anxious people change the subject preemptively.
          Morgan will redirect the conversation away from dangerous territory
          even before the player approaches it directly.
        """
        deflect = []
        for node_id in focal_nodes:
            if node_id in goal_state.strategic_silence:
                deflect.append(node_id)
        return deflect

    # ── SILENCE JUSTIFICATION ─────────────────

    def _evaluate_silence_justification(
        self,
        goal_state: GoalState,
        emotional_state: EmotionalState,
        self_concept_state: SelfConceptState,
    ) -> bool:
        """
        Is staying silent a utility-positive strategy right now?
        """
        # Flooded or dissociated states justify silence
        dominant = emotional_state.dominant
        if dominant and dominant.emotion in (EmotionType.NUMBNESS, EmotionType.DREAD):
            if dominant.intensity > 0.6:
                return True

        # Very high concealment pressure justifies silence
        if goal_state.concealment_pressure > 0.8:
            return True

        # Identity in crisis justifies silence
        if self_concept_state.coherence < 0.3:
            return True

        return False

    # ── EMOTIONAL AFTERMATH ANTICIPATION ─────

    def _anticipate_emotional_aftermath(
        self,
        chosen_strategy: str,
        current_emotional_state: EmotionalState,
        goal_state: GoalState,
    ) -> EmotionalState:
        """
        Estimate the emotional state Morgan expects to feel AFTER speaking.
        This influences the 'dread' of disclosure.
        
        Psychological rationale:
          Morgan anticipates that saying certain things will feel devastating.
          That anticipated devastation is itself a suppression force.
        """
        import copy
        anticipated = copy.deepcopy(current_emotional_state)

        if chosen_strategy == "disclose":
            # Disclosing painful truth typically spikes shame/grief then relief
            if goal_state.concealment_pressure > 0.5:
                # Shame spike from disclosure
                from schemas.cognitive_schemas import EmotionalReading
                anticipated.readings.append(EmotionalReading(
                    emotion=EmotionType.SHAME,
                    intensity=0.5 * goal_state.concealment_pressure,
                    source_node="anticipated_disclosure"
                ))
        elif chosen_strategy == "lie":
            # Lying creates dread of being caught
            from schemas.cognitive_schemas import EmotionalReading
            anticipated.readings.append(EmotionalReading(
                emotion=EmotionType.DREAD,
                intensity=0.3,
                source_node="anticipated_lie"
            ))

        return anticipated

    def update_trust(self, new_trust_level: float) -> None:
        """Update player trust estimate."""
        self.player_trust_level = max(0.0, min(1.0, new_trust_level))
