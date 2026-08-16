"""
engines/goal_engine.py

GoalEngine: Motivational Arbitration and Strategic Behavior

Psychological rationale:
  Humans are not driven by single objectives. At any moment, multiple
  competing desires, fears, and social calculations are in tension.
  Morgan doesn't have a 'rule' about the murder — she has a survival
  goal that makes disclosure dangerous, an identity goal that makes
  admission feel self-annihilating, and a possible trust goal that
  creates pressure in the other direction.

  The engine does not decide WHAT Morgan says.
  It produces the motivational field that shapes her behavioral choices.

Architecture note:
  This system is GOAL ARBITRATION, not planning.
  We are not building a planner that produces action sequences.
  We are modeling the felt pressure of competing motivations.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Optional

from schemas.cognitive_schemas import (
    Goal, GoalState, GoalType, EmotionalState, EmotionType
)
from schemas.character_schema import CharacterSchema, GoalTemplate

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# DEFAULT MORGAN GOALS
# These are Morgan's pre-loaded motivational profile.
# Goals have baseline utilities; runtime updates shift them.
# ─────────────────────────────────────────────

def build_goal_profile(character: CharacterSchema | None = None) -> list[Goal]:
    """Construct an initial motivational architecture from a character definition."""
    if character and character.goals:
        goals = []
        for template in character.goals:
            try:
                goal_type = GoalType(template.goal_type)
            except ValueError:
                goal_type = GoalType.IDENTITY
            goals.append(
                Goal(
                    goal_id=template.goal_id,
                    goal_type=goal_type,
                    description=template.description,
                    utility=template.utility,
                    urgency=template.urgency,
                    active=template.active,
                    blocking_topics=list(template.blocking_topics),
                    linked_memory_nodes=list(template.linked_memory_nodes),
                )
            )
        return goals

    return [
        Goal(
            goal_id="g_survival",
            goal_type=GoalType.SURVIVAL,
            description="Avoid physical harm, imprisonment, or death.",
            utility=0.95,
            urgency=0.7,
            blocking_topics=["the_night_it_happened", "karas_body", "river_location"]
        ),
        Goal(
            goal_id="g_identity",
            goal_type=GoalType.IDENTITY,
            description="Preserve self-image as a survivor who made hard choices, not a villain.",
            utility=0.85,
            urgency=0.4,
            blocking_topics=["helplessness_memory", "begging_scene", "crying_alone"],
            linked_memory_nodes=["self_as_survivor", "self_as_protector"]
        ),
        Goal(
            goal_id="g_trust_gain",
            goal_type=GoalType.TRUST_GAIN,
            description="Build trust with this person to extract information or gain safety.",
            utility=0.6,
            urgency=0.5,
        ),
        Goal(
            goal_id="g_trust_protect",
            goal_type=GoalType.TRUST_PROTECT,
            description="Do not say anything that will cause this person to distrust me.",
            utility=0.7,
            urgency=0.6,
        ),
        Goal(
            goal_id="g_freedom",
            goal_type=GoalType.FREEDOM,
            description="Maintain the ability to leave. Avoid entrapment, obligation, or exposure.",
            utility=0.75,
            urgency=0.5,
        ),
        Goal(
            goal_id="g_reputation",
            goal_type=GoalType.REPUTATION,
            description="Maintain reputation in the community as someone reliable and blameless.",
            utility=0.55,
            urgency=0.3,
        ),
        Goal(
            goal_id="g_concealment_primary",
            goal_type=GoalType.CONCEALMENT,
            description="The event must never be fully described to anyone.",
            utility=0.9,
            urgency=0.8,
            blocking_topics=["the_night_it_happened", "karas_body", "river_location",
                             "witness_old_maret", "the_knife"]
        ),
        Goal(
            goal_id="g_connection",
            goal_type=GoalType.CONNECTION,
            description="Morgan craves being truly known by someone. This wars with concealment.",
            utility=0.45,
            urgency=0.2,
            linked_memory_nodes=["childhood_with_kara", "last_good_night"]
        ),
    ]


# ─────────────────────────────────────────────
# GOAL ENGINE
# ─────────────────────────────────────────────

class GoalEngine:
    """
    Arbitrates between competing motivations and produces a GoalState
    describing the net behavioral pressure for a given turn.
    
    Does NOT decide what the NPC says.
    Produces the motivational field the NPC operates within.
    """

    def __init__(self, initial_goals: Optional[list[Goal]] = None):
        self.goals: list[Goal] = initial_goals or []
        self._turn_count: int = 0

    # ── GOAL MANAGEMENT ──────────────────────

    def add_goal(self, goal: Goal) -> None:
        self.goals.append(goal)

    def deactivate_goal(self, goal_id: str) -> None:
        for g in self.goals:
            if g.goal_id == goal_id:
                g.active = False

    def adjust_utility(self, goal_id: str, delta: float) -> None:
        """Runtime utility adjustment — e.g. trust just increased, so trust_gain less urgent."""
        for g in self.goals:
            if g.goal_id == goal_id:
                g.utility = max(-1.0, min(1.0, g.utility + delta))

    def adjust_urgency(self, goal_id: str, delta: float) -> None:
        for g in self.goals:
            if g.goal_id == goal_id:
                g.urgency = max(0.0, min(1.0, g.urgency + delta))

    # ── ARBITRATION CORE ─────────────────────

    def arbitrate(
        self,
        emotional_state: EmotionalState,
        activated_node_ids: list[str],
        player_trust_level: float = 0.5,
        threat_level: float = 0.0,
    ) -> GoalState:
        """
        Compute the current GoalState given emotional state and activated memories.

        Psychological rationale:
          Goals don't compete abstractly — they compete in context.
          A high fear state boosts survival and concealment.
          A high trust conversation shifts trust_protect upward.
          Morgan's connection goal rises when Kara-linked nodes are activated,
          creating vulnerability she will likely suppress.

        Args:
            emotional_state: current emotional profile
            activated_node_ids: memory nodes currently active in spreading activation
            player_trust_level: 0.0–1.0 accumulated trust with this player
            threat_level: 0.0–1.0 perceived danger from current exchange

        Returns:
            GoalState with dominant goal, pressures, and strategic signals
        """
        self._turn_count += 1
        active_goals = [g for g in self.goals if g.active]

        # Apply emotional modulation to goal utilities
        modulated = self._apply_emotional_modulation(active_goals, emotional_state, threat_level)

        # Apply trust context
        modulated = self._apply_trust_context(modulated, player_trust_level)

        # Apply memory activation context
        modulated = self._apply_activation_context(modulated, activated_node_ids)

        # Find dominant goal
        dominant = max(modulated, key=lambda g: g.utility * g.urgency, default=None)

        # Compute behavioral pressures
        concealment_pressure = self._compute_concealment_pressure(modulated, activated_node_ids)
        disclosure_pressure = self._compute_disclosure_pressure(modulated, activated_node_ids, player_trust_level)
        manipulation_intent = self._compute_manipulation_intent(modulated, threat_level)
        conflict_score = self._compute_conflict_score(modulated)

        # Strategic silence: nodes that multiple high-utility goals want suppressed
        strategic_silence = self._compute_strategic_silence(modulated, activated_node_ids)

        goal_state = GoalState(
            active_goals=modulated,
            dominant_goal=dominant,
            strategic_silence=strategic_silence,
            disclosure_pressure=disclosure_pressure,
            concealment_pressure=concealment_pressure,
            manipulation_intent=manipulation_intent,
            goal_conflict_score=conflict_score,
        )

        logger.debug(
            f"[GoalEngine] dominant={dominant.goal_id if dominant else None}, "
            f"concealment={concealment_pressure:.2f}, disclosure={disclosure_pressure:.2f}, "
            f"conflict={conflict_score:.2f}"
        )

        return goal_state

    # ── MODULATION HELPERS ───────────────────

    def _apply_emotional_modulation(
        self,
        goals: list[Goal],
        emotional_state: EmotionalState,
        threat_level: float
    ) -> list[Goal]:
        """
        Emotional state shifts goal utilities.
        Fear boosts survival and concealment.
        Grief boosts connection (and potentially disclosure).
        Shame boosts identity defense.
        """
        import copy
        modulated = copy.deepcopy(goals)
        dominant = emotional_state.dominant

        if not dominant:
            return modulated

        em = dominant.emotion
        intensity = dominant.intensity

        for g in modulated:
            if em == EmotionType.FEAR:
                if g.goal_type in (GoalType.SURVIVAL, GoalType.CONCEALMENT):
                    g.utility = min(1.0, g.utility + 0.2 * intensity)
                    g.urgency = min(1.0, g.urgency + 0.15 * intensity)
                if g.goal_type == GoalType.TRUST_GAIN:
                    g.utility -= 0.1 * intensity  # fear makes trust harder to seek
                if g.goal_type == GoalType.CONNECTION:
                    g.utility -= 0.15 * intensity  # fear closes connection

            elif em == EmotionType.GRIEF:
                if g.goal_type == GoalType.CONNECTION:
                    g.utility = min(1.0, g.utility + 0.25 * intensity)  # grief craves connection
                if g.goal_type == GoalType.CONCEALMENT:
                    g.utility -= 0.1 * intensity  # grief loosens concealment grip slightly

            elif em == EmotionType.SHAME:
                if g.goal_type == GoalType.IDENTITY:
                    g.utility = min(1.0, g.utility + 0.3 * intensity)
                    g.urgency = min(1.0, g.urgency + 0.2 * intensity)
                if g.goal_type == GoalType.REPUTATION:
                    g.utility = min(1.0, g.utility + 0.2 * intensity)
                if g.goal_type == GoalType.CONNECTION:
                    g.utility -= 0.2 * intensity  # shame pulls away from closeness

            elif em == EmotionType.ANGER:
                if g.goal_type == GoalType.REVENGE:
                    g.utility = min(1.0, g.utility + 0.35 * intensity)
                if g.goal_type == GoalType.CONCEALMENT:
                    g.utility -= 0.05 * intensity  # anger makes people slip

            elif em == EmotionType.GUILT:
                if g.goal_type == GoalType.CONCEALMENT:
                    g.utility -= 0.1 * intensity  # guilt corrodes concealment
                if g.goal_type == GoalType.TRUST_GAIN:
                    g.urgency = min(1.0, g.urgency + 0.1 * intensity)  # guilt seeks absolution

        # Threat level globally boosts survival and concealment
        if threat_level > 0.3:
            for g in modulated:
                if g.goal_type in (GoalType.SURVIVAL, GoalType.CONCEALMENT):
                    g.urgency = min(1.0, g.urgency + 0.1 * threat_level)

        return modulated

    def _apply_trust_context(self, goals: list[Goal], trust_level: float) -> list[Goal]:
        """
        High trust reduces trust_gain urgency and can lower concealment slightly.
        Low trust boosts trust_gain urgency.
        """
        for g in goals:
            if g.goal_type == GoalType.TRUST_GAIN:
                if trust_level > 0.7:
                    g.urgency -= 0.2   # already trusted; less urgent to pursue
                elif trust_level < 0.3:
                    g.urgency = min(1.0, g.urgency + 0.2)
            if g.goal_type == GoalType.CONCEALMENT and trust_level > 0.8:
                # Very high trust barely softens concealment — but not much
                g.utility -= 0.05
        return goals

    def _apply_activation_context(self, goals: list[Goal], activated_nodes: list[str]) -> list[Goal]:
        """
        When memory nodes linked to a goal are activated, that goal becomes more
        salient and urgent — even if the NPC isn't consciously 'deciding.'
        
        Psychological rationale:
          Morgan doesn't 'choose' to feel protective of her identity
          when the drowning memory activates. The memory itself triggers
          the identity goal's urgency.
        """
        activated_set = set(activated_nodes)
        for g in goals:
            overlap = activated_set.intersection(set(g.linked_memory_nodes))
            if overlap:
                g.urgency = min(1.0, g.urgency + 0.1 * len(overlap))
            # If blocking topics are activated, urgency of concealment spikes
            blocking_overlap = activated_set.intersection(set(g.blocking_topics))
            if blocking_overlap and g.goal_type == GoalType.CONCEALMENT:
                g.urgency = min(1.0, g.urgency + 0.15 * len(blocking_overlap))
        return goals

    # ── PRESSURE COMPUTATIONS ────────────────

    def _compute_concealment_pressure(
        self, goals: list[Goal], activated_nodes: list[str]
    ) -> float:
        """
        How strongly is the NPC's cognition pulling toward hiding things?
        Driven by high-utility concealment + survival goals with active blocking nodes.
        """
        activated_set = set(activated_nodes)
        pressure = 0.0
        for g in goals:
            if g.goal_type in (GoalType.CONCEALMENT, GoalType.SURVIVAL, GoalType.IDENTITY):
                blocking_active = activated_set.intersection(set(g.blocking_topics))
                if blocking_active:
                    pressure += g.utility * g.urgency * (0.5 + 0.5 * len(blocking_active) / max(1, len(g.blocking_topics)))
        return min(1.0, pressure)

    def _compute_disclosure_pressure(
        self, goals: list[Goal], activated_nodes: list[str], trust_level: float
    ) -> float:
        """
        How much is the NPC pulled toward revealing something?
        Trust gain + connection goals + guilt drive this.
        """
        pressure = 0.0
        activated_set = set(activated_nodes)
        for g in goals:
            if g.goal_type in (GoalType.TRUST_GAIN, GoalType.CONNECTION):
                linked_active = activated_set.intersection(set(g.linked_memory_nodes))
                if linked_active or trust_level > 0.6:
                    pressure += g.utility * g.urgency * trust_level
        return min(1.0, pressure)

    def _compute_manipulation_intent(self, goals: list[Goal], threat_level: float) -> float:
        """
        How likely is the NPC to shade truth, deflect, or manipulate?
        Driven by concealment + survival under threat.
        """
        intent = 0.0
        for g in goals:
            if g.goal_type in (GoalType.CONCEALMENT, GoalType.SURVIVAL, GoalType.REPUTATION):
                intent += g.utility * g.urgency * 0.3
        intent *= (1.0 + threat_level * 0.5)
        return min(1.0, intent)

    def _compute_conflict_score(self, goals: list[Goal]) -> float:
        """
        Measure how much goals are fighting each other.
        High conflict → behavioral instability, inconsistency, fragmentation.
        """
        if not goals:
            return 0.0
        utilities = [g.utility for g in goals if g.active]
        if len(utilities) < 2:
            return 0.0
        max_u = max(utilities)
        min_u = min(utilities)
        # Spread in utility values signals conflict
        spread = max_u - min_u
        # Also count goals with active blocking overlaps
        return min(1.0, spread * 0.7)

    def _compute_strategic_silence(
        self, goals: list[Goal], activated_nodes: list[str]
    ) -> list[str]:
        """
        Which activated nodes should be strategically avoided in language?
        These are nodes that high-utility goals are blocking.
        """
        activated_set = set(activated_nodes)
        silence = set()
        for g in goals:
            if g.utility > 0.6 and g.goal_type in (
                GoalType.CONCEALMENT, GoalType.SURVIVAL, GoalType.IDENTITY
            ):
                blocked = activated_set.intersection(set(g.blocking_topics))
                silence.update(blocked)
        return list(silence)

    # ── RUNTIME UPDATES ──────────────────────

    def on_trust_gained(self, amount: float) -> None:
        """Called when player trust increases."""
        self.adjust_utility("g_trust_gain", -amount * 0.3)   # less urgent now
        self.adjust_utility("g_trust_protect", amount * 0.2)  # more to protect

    def on_threat_detected(self, severity: float) -> None:
        """Called when dialogue content suggests danger to Morgan."""
        self.adjust_urgency("g_survival", severity * 0.3)
        self.adjust_urgency("g_concealment_primary", severity * 0.25)
        self.adjust_utility("g_connection", -severity * 0.2)

    def on_kara_mentioned(self) -> None:
        """Special trigger: Kara's name activates connection AND concealment."""
        self.adjust_urgency("g_connection", 0.15)
        self.adjust_urgency("g_concealment_primary", 0.1)
        self.adjust_urgency("g_identity", 0.1)

    def serialize(self) -> dict:
        """Persist goal state."""
        return {
            "goals": [
                {
                    "goal_id": g.goal_id,
                    "goal_type": g.goal_type.value,
                    "description": g.description,
                    "utility": g.utility,
                    "urgency": g.urgency,
                    "active": g.active,
                    "blocking_topics": g.blocking_topics,
                    "linked_memory_nodes": g.linked_memory_nodes,
                }
                for g in self.goals
            ],
            "turn_count": self._turn_count,
        }

    @classmethod
    def deserialize(cls, data: dict) -> GoalEngine:
        goals = []
        for gd in data["goals"]:
            goals.append(Goal(
                goal_id=gd["goal_id"],
                goal_type=GoalType(gd["goal_type"]),
                description=gd["description"],
                utility=gd["utility"],
                urgency=gd["urgency"],
                active=gd["active"],
                blocking_topics=gd["blocking_topics"],
                linked_memory_nodes=gd["linked_memory_nodes"],
            ))
        engine = cls(initial_goals=goals)
        engine._turn_count = data.get("turn_count", 0)
        return engine
