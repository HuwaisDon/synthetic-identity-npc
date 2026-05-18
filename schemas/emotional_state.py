
"""
Emotional state schema.

Emotional state is not just "current mood." It's a multi-dimensional
vector that biases every other cognitive operation — memory activation,
suppression force, spreading activation weights, distortion probability.

It is the single most important context the MemoryEngine receives.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmotionalState:
    """
    The NPC's current internal state at the moment of processing.

    Key design principle: emotional state is an INPUT to memory retrieval,
    not an output. Current grief makes grief-associated memories more
    accessible. Current fear makes fear-linked nodes activate faster.
    """
    npc_id: str
    turn: int = 0

    # --- Primary dimensions ---
    arousal: float = 0.4        # 0.0 (flat/dissociated) to 1.0 (acute distress/excitement)
    valence: float = 0.0        # -1.0 (negative) to +1.0 (positive)

    # --- Named emotional states (can coexist) ---
    grief: float = 0.0
    fear: float = 0.0
    anger: float = 0.0
    shame: float = 0.0
    guilt: float = 0.0
    loneliness: float = 0.0
    suspicion: float = 0.3      # Morgan's baseline
    warmth: float = 0.0
    pride: float = 0.0
    nostalgia: float = 0.0
    anxiety: float = 0.0

    # --- Trait-level suppressors ---
    emotional_suppression: float = 0.80   # Morgan's core trait — high
    # This dampens disclosure probability even when activation is high

    # --- Relational state ---
    trust_toward_player: float = 0.1
    hostility_toward_player: float = 0.0

    # --- Situational modifiers ---
    stress_level: float = 0.4
    fatigue: float = 0.0
    intoxication: float = 0.0
    physical_pain: float = 0.0

    # --- Active fear stack (ordered by intensity) ---
    active_fears: list[str] = field(default_factory=list)
    # e.g. ["drowning", "imprisonment", "betrayal"]

    # --- Active desires ---
    active_desires: list[str] = field(default_factory=list)
    # e.g. ["coin", "respect", "passage_south"]

    # --- Environmental context (drives ambient activation) ---
    environment_tags: list[str] = field(default_factory=list)
    # e.g. ["rain", "ship_deck", "night", "tavern"]
    # "rain" can activate drowning memory without player mentioning water

    def dominant_emotion(self) -> str:
        """Return the highest-magnitude named emotion."""
        emotions = {
            "grief": self.grief, "fear": self.fear, "anger": self.anger,
            "shame": self.shame, "guilt": self.guilt, "suspicion": self.suspicion,
            "warmth": self.warmth, "pride": self.pride, "nostalgia": self.nostalgia,
            "anxiety": self.anxiety, "loneliness": self.loneliness
        }
        return max(emotions, key=emotions.get)

    def suppression_force(self) -> float:
        """
        The active suppression pressure at this moment.
        Higher suppression = lower disclosure even when memories activate strongly.
        Fatigue and intoxication reduce suppression (defenses weaken).
        """
        base = self.emotional_suppression
        fatigue_reduction = self.fatigue * 0.3
        intox_reduction = self.intoxication * 0.4
        stress_increase = self.stress_level * 0.1
        return max(0.0, min(1.0, base - fatigue_reduction - intox_reduction + stress_increase))

    def as_activation_bias(self) -> dict[str, float]:
        """
        Convert emotional state into concept-level activation biases.
        These are added to graph node activation scores during retrieval.
        This is how environment and mood prime memory without explicit input.
        """
        biases: dict[str, float] = {}

        if self.grief > 0.3:
            for concept in ["loss", "father", "death", "drowning", "absence"]:
                biases[concept] = biases.get(concept, 0) + self.grief * 0.6

        if self.fear > 0.3:
            for concept in self.active_fears:
                biases[concept] = biases.get(concept, 0) + self.fear * 0.7

        if self.suspicion > 0.4:
            for concept in ["betrayal", "trap", "deception"]:
                biases[concept] = biases.get(concept, 0) + self.suspicion * 0.5

        if self.nostalgia > 0.3:
            for concept in ["childhood", "mentor", "old_veth", "port_carrath"]:
                biases[concept] = biases.get(concept, 0) + self.nostalgia * 0.5

        # Environmental priming — this is where rain activates drowning
        for tag in self.environment_tags:
            biases[tag] = biases.get(tag, 0) + 0.4

        return biases
