"""
Suppression / interception engine.

Suppression is not forgetting and not deletion. It is a behavioral gate:
the memory remains emotionally active, but direct disclosure becomes harder.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from memory.activation import ActivatedMemory
from schemas.cognitive_schemas import EmotionalReading, EmotionType


@dataclass
class SuppressionResult:
    node_id: str
    suppression_strength: float
    disclosure_inhibition: float
    behavioral_leakage: float
    deflection_probability: float
    emotional_tension: float


@dataclass
class SuppressionState:
    results: list[SuppressionResult] = field(default_factory=list)
    total_disclosure_inhibition: float = 0.0
    total_behavioral_leakage: float = 0.0
    total_deflection_pressure: float = 0.0
    total_emotional_tension: float = 0.0
    intercepted_nodes: list[str] = field(default_factory=list)

    def by_node(self) -> dict[str, SuppressionResult]:
        return {result.node_id: result for result in self.results}


class SuppressionEngine:
    """
    Converts per-memory suppression pressure into behavioral interception.

    High suppression means:
    - lower direct disclosure salience
    - higher deflection pressure
    - higher emotional tension
    - more indirect behavioral leakage
    """

    def evaluate(
        self,
        activated_memories: list[ActivatedMemory],
        activated_nodes: dict[str, float],
        min_suppression_pressure: float = 0.12,
    ) -> SuppressionState:
        results = []

        for memory in activated_memories:
            activation = activated_nodes.get(memory.memory_id, memory.activation_score)
            suppression_pressure = memory.suppression_pressure

            if suppression_pressure < min_suppression_pressure:
                continue

            suppression_level = memory.metadata.get("suppression_level", 0.0)

            suppression_strength = self._clamp(
                suppression_pressure * 1.4 + suppression_level * 0.25
            )
            disclosure_inhibition = self._clamp(
                suppression_strength * 0.65 + activation * 0.25
            )
            behavioral_leakage = self._clamp(
                suppression_strength * activation * 1.15
            )
            emotional_tension = self._clamp(
                suppression_strength * 0.55 + activation * 0.45
            )
            deflection_probability = self._clamp(
                disclosure_inhibition * 0.6 + emotional_tension * 0.4
            )

            results.append(
                SuppressionResult(
                    node_id=memory.memory_id,
                    suppression_strength=suppression_strength,
                    disclosure_inhibition=disclosure_inhibition,
                    behavioral_leakage=behavioral_leakage,
                    deflection_probability=deflection_probability,
                    emotional_tension=emotional_tension,
                )
            )

        results.sort(key=lambda result: result.suppression_strength, reverse=True)
        return SuppressionState(
            results=results,
            total_disclosure_inhibition=self._max_metric(results, "disclosure_inhibition"),
            total_behavioral_leakage=self._max_metric(results, "behavioral_leakage"),
            total_deflection_pressure=self._max_metric(results, "deflection_probability"),
            total_emotional_tension=self._max_metric(results, "emotional_tension"),
            intercepted_nodes=[
                result.node_id
                for result in results
                if result.disclosure_inhibition >= 0.35
            ],
        )

    def apply_attention_interception(
        self,
        activated_nodes: dict[str, float],
        suppression_state: SuppressionState,
    ) -> dict[str, float]:
        """
        Reduce direct conversational salience without changing raw activation.
        """
        suppression_by_node = suppression_state.by_node()
        intercepted = {}

        for node_id, activation in activated_nodes.items():
            result = suppression_by_node.get(node_id)
            if not result:
                intercepted[node_id] = activation
                continue

            salience_multiplier = 1.0 - result.disclosure_inhibition * 0.55
            intercepted[node_id] = max(0.05, activation * salience_multiplier)

        return intercepted

    def build_leakage_readings(
        self,
        suppression_state: SuppressionState,
        node_emotion_tags: dict[str, list[str]],
    ) -> list[EmotionalReading]:
        readings = []

        for result in suppression_state.results:
            if result.behavioral_leakage < 0.15:
                continue

            tags = node_emotion_tags.get(result.node_id, [])
            emotion = self._emotion_from_tags(tags)
            intensity = self._clamp(
                result.emotional_tension * 0.55 + result.behavioral_leakage * 0.25
            )

            if intensity > 0.1:
                readings.append(
                    EmotionalReading(
                        emotion=emotion,
                        intensity=intensity,
                        source_node=result.node_id,
                        is_suppressed=True,
                    )
                )

        return readings

    @staticmethod
    def _emotion_from_tags(tags: list[str]) -> EmotionType:
        for tag in tags:
            try:
                return EmotionType(tag)
            except ValueError:
                continue
        return EmotionType.DREAD

    @staticmethod
    def _max_metric(results: list[SuppressionResult], metric: str) -> float:
        if not results:
            return 0.0
        return max(getattr(result, metric) for result in results)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, value))
