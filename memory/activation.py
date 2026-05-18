"""
Activation system.

Converts semantic retrieval into psychologically weighted activation.

This is where:
- current emotional state
- fears
- suppression sensitivity
- emotional congruence

begin influencing memory activation.

Semantic retrieval answers:
    "What memories are similar?"

Activation answers:
    "Which memories psychologically dominate right now?"
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ActivatedMemory:
    memory_id: str
    semantic_score: float
    activation_score: float
    emotional_resonance: float
    suppression_pressure: float
    metadata: dict


class ActivationEngine:

    def __init__(self):
        pass

    def compute_activation(
        self,
        retrieved_memories: List[tuple],
        emotional_state: dict,
    ) -> List[ActivatedMemory]:

        activated = []

        for memory_id, semantic_score, metadata in retrieved_memories:

            emotional_weight = metadata.get("emotional_weight", 0.5)
            suppression_level = metadata.get("suppression_level", 0.0)
            valence = metadata.get("valence", 0.0)

            # Current emotional state
            sadness = emotional_state.get("sadness", 0.0)
            fear = emotional_state.get("fear", 0.0)
            anger = emotional_state.get("anger", 0.0)

            # Emotional resonance
            emotional_resonance = (
                abs(valence) * 0.4
                + sadness * 0.3
                + fear * 0.2
                + anger * 0.1
            )

            # Base activation
            activation_score = (
                semantic_score * 0.5
                + emotional_weight * 0.3
                + emotional_resonance * 0.2
            )

            # Suppression pressure
            suppression_pressure = (
                activation_score * suppression_level
            )

            activated.append(
                ActivatedMemory(
                    memory_id=memory_id,
                    semantic_score=semantic_score,
                    activation_score=activation_score,
                    emotional_resonance=emotional_resonance,
                    suppression_pressure=suppression_pressure,
                    metadata=metadata,
                )
            )

        # Highest activation first
        activated.sort(
            key=lambda x: x.activation_score,
            reverse=True
        )

        return activated