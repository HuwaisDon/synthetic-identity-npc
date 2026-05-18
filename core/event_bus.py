"""
core/event_bus.py

Event-Driven Emotional Updates: NPC Cognitive Event Bus

Architecture rationale:
  The cognitive pipeline processes direct player conversation.
  But NPC cognition is also driven by world events:
    - Witnessing something
    - Overhearing a conversation
    - Being threatened by a third party
    - Receiving news
    - Environmental triggers (returning to a location)

  The event bus allows external world systems to inject
  emotional/cognitive events into the NPC's state between turns.

  Events are processed asynchronously and update:
    - Goal utilities and urgencies
    - Emotional persistence (inject residue)
    - Self-concept state (if event is identity-threatening)
    - Attention priming (next turn starts with this node salient)

Psychological rationale:
  A person who just witnessed something doesn't wait for someone to ask
  about it before it affects their cognition. The event itself immediately
  reshapes attentional salience, emotional baseline, and goal priorities.
"""

from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

from schemas.cognitive_schemas import EmotionType, EmotionalReading

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# EVENT TYPES
# ─────────────────────────────────────────────

class CognitiveEventType(Enum):
    WITNESSED_VIOLENCE      = "witnessed_violence"
    RECEIVED_THREAT         = "received_threat"
    OVERHEARD_RELEVANT      = "overheard_relevant"
    LOCATION_TRIGGER        = "location_trigger"       # returned to significant place
    NAMED_IN_ACCUSATION     = "named_in_accusation"
    TRUSTED_PERSON_ARRIVED  = "trusted_person_arrived"
    DANGER_PASSED           = "danger_passed"
    UNEXPECTED_KINDNESS     = "unexpected_kindness"
    MEMORY_OBJECT_SEEN      = "memory_object_seen"     # saw an object linked to the past
    ANNIVERSARY             = "anniversary"             # date-linked emotional trigger


@dataclass
class CognitiveEvent:
    """
    An external world event that should immediately update NPC cognition.
    """
    event_type: CognitiveEventType
    npc_id: str
    description: str
    linked_nodes: list[str] = field(default_factory=list)  # memory nodes activated
    emotional_payload: list[tuple[EmotionType, float]] = field(default_factory=list)
    threat_delta: float = 0.0    # change in perceived threat level
    trust_delta: float = 0.0     # change in player trust
    goal_impacts: dict[str, float] = field(default_factory=dict)  # goal_id -> utility_delta
    timestamp: float = field(default_factory=time.time)
    processed: bool = False


# ─────────────────────────────────────────────
# EVENT BUS
# ─────────────────────────────────────────────

class CognitiveEventBus:
    """
    Routes world events to NPC cognitive engines.
    Maintains a queue of unprocessed events per NPC.
    """

    def __init__(self):
        self._queues: dict[str, list[CognitiveEvent]] = {}
        self._handlers: list[Callable] = []

    def emit(self, event: CognitiveEvent) -> None:
        """Emit an event. It will be processed on the NPC's next turn."""
        npc_id = event.npc_id
        if npc_id not in self._queues:
            self._queues[npc_id] = []
        self._queues[npc_id].append(event)
        logger.info(
            f"[EventBus] {event.event_type.value} → {npc_id}: {event.description[:60]}"
        )

    def drain(self, npc_id: str) -> list[CognitiveEvent]:
        """Get and clear all pending events for this NPC."""
        events = self._queues.get(npc_id, [])
        self._queues[npc_id] = []
        return events

    def apply_to_pipeline(self, npc_id: str, pipeline) -> None:
        """
        Apply all pending events to a pipeline's cognitive engines.
        Call this at the START of each turn before processing player input.
        """
        events = self.drain(npc_id)
        if not events:
            return

        for event in events:
            self._apply_event(event, pipeline)
            event.processed = True

    def _apply_event(self, event: CognitiveEvent, pipeline) -> None:
        """Apply a single event to the pipeline's engines."""
        logger.debug(f"[EventBus] Applying: {event.event_type.value}")

        # Inject emotional residue
        for emotion, intensity in event.emotional_payload:
            reading = EmotionalReading(
                emotion=emotion,
                intensity=intensity,
                source_node=event.linked_nodes[0] if event.linked_nodes else None,
            )
            pipeline.persistence_engine._inject_residue(reading, event.linked_nodes)

        # Update goal utilities
        for goal_id, delta in event.goal_impacts.items():
            pipeline.goal_engine.adjust_utility(goal_id, delta)

        # Threat events
        if event.threat_delta > 0:
            pipeline.goal_engine.on_threat_detected(event.threat_delta)

        # Trust events
        if event.trust_delta > 0:
            pipeline.goal_engine.on_trust_gained(event.trust_delta)

        # Event-specific logic
        if event.event_type == CognitiveEventType.LOCATION_TRIGGER:
            self._handle_location_trigger(event, pipeline)

        elif event.event_type == CognitiveEventType.NAMED_IN_ACCUSATION:
            self._handle_accusation(event, pipeline)

        elif event.event_type == CognitiveEventType.MEMORY_OBJECT_SEEN:
            self._handle_memory_object(event, pipeline)

        elif event.event_type == CognitiveEventType.ANNIVERSARY:
            self._handle_anniversary(event, pipeline)

    def _handle_location_trigger(self, event: CognitiveEvent, pipeline) -> None:
        """
        Returning to a significant location activates associated memories
        and injects location-linked emotional residue.
        
        Example: Morgan walks past the river → memory activation spike,
        fear/dread residue injected even without player interaction.
        """
        for node_id in event.linked_nodes:
            # Prime the attention engine: these nodes will be highly salient next turn
            pipeline.attention_engine.focal_capacity = max(
                1, pipeline.attention_engine.focal_capacity - 1
            )  # Temporarily narrow attention (overwhelmed by location)
        logger.info(f"[EventBus] Location trigger: attention narrowed for {event.npc_id}")

    def _handle_accusation(self, event: CognitiveEvent, pipeline) -> None:
        """
        Being named in an accusation spikes survival goal, concealment urgency,
        and activates self-concept defense.
        """
        pipeline.goal_engine.adjust_urgency("g_survival", 0.3)
        pipeline.goal_engine.adjust_urgency("g_concealment_primary", 0.25)
        pipeline.goal_engine.adjust_utility("g_identity", 0.2)
        logger.info(f"[EventBus] Accusation event: survival/concealment elevated for {event.npc_id}")

    def _handle_memory_object(self, event: CognitiveEvent, pipeline) -> None:
        """
        Seeing an object linked to traumatic memory (e.g., a knife, a letter)
        activates associated nodes and injects emotional residue.
        """
        # Additional shame/fear residue
        for node_id in event.linked_nodes:
            for emotion, intensity in [(EmotionType.FEAR, 0.4), (EmotionType.SHAME, 0.35)]:
                reading = EmotionalReading(
                    emotion=emotion,
                    intensity=intensity,
                    source_node=node_id,
                    is_suppressed=True,  # Likely suppressed immediately
                )
                pipeline.persistence_engine._inject_residue(reading, [node_id])

    def _handle_anniversary(self, event: CognitiveEvent, pipeline) -> None:
        """
        Date-linked triggers (anniversaries of traumatic events).
        Annual activation of full memory cluster with grief/dread.
        """
        for emotion, intensity in [
            (EmotionType.GRIEF, 0.6),
            (EmotionType.DREAD, 0.4),
            (EmotionType.NUMBNESS, 0.3),
        ]:
            reading = EmotionalReading(
                emotion=emotion,
                intensity=intensity,
                source_node="anniversary_trigger",
                is_suppressed=False,  # Anniversary grief often breaks through
            )
            pipeline.persistence_engine._inject_residue(reading, event.linked_nodes)
        # Temporarily lower self-concept coherence (anniversaries destabilize)
        pipeline.self_concept.state.coherence = max(
            0.3, pipeline.self_concept.state.coherence - 0.2
        )
        logger.info(f"[EventBus] Anniversary: grief/dread injected for {event.npc_id}")


# ─────────────────────────────────────────────
# EXAMPLE EVENTS FOR MORGAN
# ─────────────────────────────────────────────

def make_river_walk_event(npc_id: str = "morgan_veth") -> CognitiveEvent:
    """Morgan walks past the river — location trigger."""
    return CognitiveEvent(
        event_type=CognitiveEventType.LOCATION_TRIGGER,
        npc_id=npc_id,
        description="Morgan passes within sight of the river bend where it happened.",
        linked_nodes=["river_location", "karas_body", "the_night_it_happened"],
        emotional_payload=[
            (EmotionType.FEAR, 0.6),
            (EmotionType.DREAD, 0.5),
            (EmotionType.GRIEF, 0.4),
        ],
        threat_delta=0.1,
        goal_impacts={"g_concealment_primary": 0.1},
    )


def make_maret_sighting_event(npc_id: str = "morgan_veth") -> CognitiveEvent:
    """Morgan sees Old Maret looking at her in the market."""
    return CognitiveEvent(
        event_type=CognitiveEventType.RECEIVED_THREAT,
        npc_id=npc_id,
        description="Old Maret watched Morgan across the market square. Said nothing.",
        linked_nodes=["witness_old_maret", "the_night_it_happened"],
        emotional_payload=[
            (EmotionType.FEAR, 0.5),
            (EmotionType.DREAD, 0.45),
        ],
        threat_delta=0.25,
        goal_impacts={
            "g_survival": 0.15,
            "g_concealment_primary": 0.2,
        },
    )


def make_kara_anniversary_event(npc_id: str = "morgan_veth") -> CognitiveEvent:
    """Three-year anniversary of Kara's death."""
    return CognitiveEvent(
        event_type=CognitiveEventType.ANNIVERSARY,
        npc_id=npc_id,
        description="Three years since the night at the river. Morgan knows the date.",
        linked_nodes=[
            "childhood_with_kara", "last_good_night", "karas_body",
            "the_night_it_happened", "grief_for_kara",
        ],
        emotional_payload=[
            (EmotionType.GRIEF, 0.8),
            (EmotionType.DREAD, 0.5),
            (EmotionType.GUILT, 0.6),
        ],
        goal_impacts={"g_connection": 0.2},
    )
