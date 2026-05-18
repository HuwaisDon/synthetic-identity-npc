"""
engines/emotional_persistence.py

EmotionalPersistenceEngine: Temporal Emotional Carryover and Rumination

Psychological rationale:
  Emotions are not instantaneous or turn-scoped. A painful disclosure in
  conversation 1 doesn't reset to neutral by conversation 2.
  
  Humans experience:
  - Emotional residue: lingering activation after an emotional event
  - Rumination: recursive re-activation of a single distressing node
  - Mood drift: baseline emotional tone shifting over repeated exposure
  - Unresolved emotional residue: activated-but-unspeakable emotion 
    that corrupts subsequent cognition
  
  This engine manages the TEMPORAL DIMENSION of emotional cognition.

Architecture note:
  This engine runs between turns — it decays residue, accumulates mood drift,
  and adjusts the baseline emotional state that the activation system starts from.
  
  It also detects rumination: when the same node is repeatedly reactivated
  across turns, suggesting it is psychologically 'stuck.'
"""

from __future__ import annotations
import logging
from typing import Optional

from schemas.cognitive_schemas import (
    EmotionalPersistenceState, EmotionalResidueEntry,
    EmotionalState, EmotionalReading, EmotionType, AttentionState
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# RUMINATION THRESHOLD
# Node must be focal for this many consecutive turns to trigger rumination
RUMINATION_THRESHOLD = 3

# Decay rate per turn for residue
DEFAULT_DECAY_RATE = 0.15

# How much a single emotional event shifts mood baseline
MOOD_DRIFT_RATE = 0.05


class EmotionalPersistenceEngine:
    """
    Manages emotional carryover, residue decay, and rumination across turns.
    
    Called:
    1. At END of turn: inject new residue from this turn's emotional activation
    2. At START of next turn: apply residue to modify emotional baseline
    3. Between turns: decay residue entries
    """

    def __init__(self):
        self.state = EmotionalPersistenceState()
        self._focal_history: list[list[str]] = []   # per-turn list of focal nodes
        self._turn_count: int = 0

    # ── TURN START: Apply residue to baseline ───

    def get_modified_baseline(
        self, base_emotional_state: EmotionalState
    ) -> EmotionalState:
        """
        At the start of each turn, apply lingering emotional residue
        to the incoming emotional state.
        
        This means: if Morgan experienced intense shame two turns ago
        and it hasn't fully resolved, that shame is STILL influencing
        her cognition — even if nothing shameful is being discussed.
        
        Psychological rationale:
          Mood is a filter, not a reaction. Prior emotional events
          lower or raise activation thresholds for related content.
        """
        if not self.state.residue:
            return base_emotional_state

        import copy
        modified = copy.deepcopy(base_emotional_state)

        for entry in self.state.residue:
            if entry.intensity < 0.05:
                continue
            # Inject residue emotion into current readings
            residue_reading = EmotionalReading(
                emotion=entry.emotion,
                intensity=entry.intensity * 0.7,   # residue is dimmer than fresh activation
                source_node=entry.source_node,
                is_suppressed=False,
            )
            modified.readings.append(residue_reading)

        # Apply mood baseline shift
        if self.state.mood_baseline_shift != 0.0:
            modified.valence = max(-1.0, min(1.0,
                modified.valence + self.state.mood_baseline_shift * 0.3
            ))

        # Add rumination: if a node is being ruminated, boost its emotional weight
        if self.state.rumination_node and self.state.rumination_intensity > 0.1:
            rumination_reading = EmotionalReading(
                emotion=EmotionType.DREAD,   # rumination creates low-level dread
                intensity=self.state.rumination_intensity * 0.5,
                source_node=self.state.rumination_node,
            )
            modified.readings.append(rumination_reading)

        # Recompute dominant from modified readings
        if modified.readings:
            modified.dominant = max(modified.readings, key=lambda r: r.intensity)

        return modified

    # ── TURN END: Inject new residue ─────────

    def record_turn(
        self,
        emotional_readings: list[EmotionalReading],
        focal_nodes: list[str],
        resolved_nodes: Optional[list[str]] = None,  # nodes that were expressed/processed
    ) -> None:
        """
        Called at end of turn to:
        1. Decay existing residue
        2. Inject new residue from this turn's significant emotions
        3. Update rumination tracking
        4. Update mood drift
        
        Args:
            emotional_readings: emotions activated this turn
            focal_nodes: memory nodes that were in attention this turn
            resolved_nodes: nodes that were expressed (reduce residue for these)
        """
        self._turn_count += 1
        self._focal_history.append(focal_nodes)

        # Step 1: Decay existing residue
        self._decay_residue()

        # Step 2: Remove resolved residue (expression processes emotion)
        if resolved_nodes:
            self._process_expressed_nodes(resolved_nodes)

        # Step 3: Inject new residue from high-intensity emotions
        for reading in emotional_readings:
            if reading.intensity > 0.35:  # only significant emotions create residue
                self._inject_residue(reading, focal_nodes)

        # Step 4: Update rumination
        self._update_rumination()

        # Step 5: Update mood drift
        self._update_mood_drift(emotional_readings)

        logger.debug(
            f"[Persistence] turn={self._turn_count}, "
            f"residue_count={len(self.state.residue)}, "
            f"rumination={self.state.rumination_node}, "
            f"mood_drift={self.state.mood_baseline_shift:.3f}"
        )

    # ── RESIDUE MANAGEMENT ───────────────────

    def _inject_residue(
        self, reading: EmotionalReading, focal_nodes: list[str]
    ) -> None:
        """
        Create a residue entry from a significant emotional activation.
        
        Suppressed emotions leave MORE residue because they aren't processed.
        Psychological basis: unexpressed emotion doesn't disappear — it
        accumulates as pressure.
        """
        # How many turns will this linger?
        # High intensity + suppression = longer residue
        base_turns = max(1, int(reading.intensity * 6))
        if reading.is_suppressed:
            base_turns = int(base_turns * 1.6)  # suppression prolongs residue

        # Decay rate: suppressed emotions decay more slowly
        decay_rate = DEFAULT_DECAY_RATE
        if reading.is_suppressed:
            decay_rate *= 0.6

        # Check if residue for this node already exists (compound it)
        existing = self._find_residue(reading.source_node, reading.emotion)
        if existing:
            existing.intensity = min(1.0, existing.intensity + reading.intensity * 0.4)
            existing.turns_remaining = max(existing.turns_remaining, base_turns)
            return

        entry = EmotionalResidueEntry(
            emotion=reading.emotion,
            intensity=reading.intensity,
            source_description=f"turn_{self._turn_count}",
            source_node=reading.source_node,
            turns_remaining=base_turns,
            decay_rate=decay_rate,
        )
        self.state.residue.append(entry)

    def _find_residue(
        self, source_node: Optional[str], emotion: EmotionType
    ) -> Optional[EmotionalResidueEntry]:
        for entry in self.state.residue:
            if entry.source_node == source_node and entry.emotion == emotion:
                return entry
        return None

    def _decay_residue(self) -> None:
        """
        Per-turn residue decay. Entries below threshold are removed.
        """
        surviving = []
        for entry in self.state.residue:
            entry.intensity -= entry.decay_rate
            entry.turns_remaining -= 1
            if entry.intensity > 0.05 and entry.turns_remaining > 0:
                surviving.append(entry)
        self.state.residue = surviving

    def _process_expressed_nodes(self, resolved_nodes: list[str]) -> None:
        """
        When a memory is expressed (spoken about), its residue decays faster.
        Psychological basis: expression has a processing/cathartic effect,
        even partial disclosure reduces emotional pressure.
        """
        resolved_set = set(resolved_nodes)
        for entry in self.state.residue:
            if entry.source_node in resolved_set:
                entry.intensity *= 0.6   # expression reduces residue
                entry.turns_remaining = max(0, entry.turns_remaining - 1)

    # ── RUMINATION ───────────────────────────

    def _update_rumination(self) -> None:
        """
        Detect if a node is being repeatedly activated (rumination).
        
        Psychological rationale:
          Rumination occurs when a distressing memory is activated repeatedly
          but cannot be resolved — too painful to process, too salient to drop.
          Under suppression, the blocked memory rebounds and re-activates.
          
          Ruminating on a node elevates baseline dread and narrows attention.
        """
        if len(self._focal_history) < RUMINATION_THRESHOLD:
            return

        # Look at the last N turns' focal nodes
        recent_focal = self._focal_history[-RUMINATION_THRESHOLD:]

        # Count node frequency across recent turns
        frequency: dict[str, int] = {}
        for focal_set in recent_focal:
            for node_id in focal_set:
                frequency[node_id] = frequency.get(node_id, 0) + 1

        # A node appearing in ALL recent turns is potentially rumination
        for node_id, count in frequency.items():
            if count >= RUMINATION_THRESHOLD:
                if self.state.rumination_node != node_id:
                    logger.info(f"[Persistence] Rumination detected on: {node_id}")
                self.state.rumination_node = node_id
                self.state.rumination_intensity = min(
                    1.0, count / RUMINATION_THRESHOLD * 0.5
                )
                return

        # No consistent node = no rumination
        if self.state.rumination_node:
            self.state.rumination_intensity = max(
                0.0, self.state.rumination_intensity - 0.1
            )
            if self.state.rumination_intensity < 0.05:
                self.state.rumination_node = None

    # ── MOOD DRIFT ───────────────────────────

    def _update_mood_drift(self, readings: list[EmotionalReading]) -> None:
        """
        Repeated emotional activations shift the mood baseline over time.
        
        Psychological rationale:
          Someone who experiences shame repeatedly in a short period
          develops a shame-tinged baseline. The world starts to feel
          shameful before anything specific happens.
          
          This is analogous to emotional sensitization / kindling.
        """
        if not readings:
            return

        # Average valence of this turn's emotions
        turn_valence = sum(
            (-1.0 if r.emotion in (
                EmotionType.FEAR, EmotionType.GRIEF, EmotionType.SHAME,
                EmotionType.GUILT, EmotionType.ANGER, EmotionType.DREAD,
                EmotionType.NUMBNESS
            ) else 0.5) * r.intensity
            for r in readings
        ) / len(readings)

        # Slowly pull mood baseline toward current experience
        self.state.mood_baseline_shift += MOOD_DRIFT_RATE * (
            turn_valence - self.state.mood_baseline_shift
        )
        # Clamp
        self.state.mood_baseline_shift = max(
            -0.5, min(0.2, self.state.mood_baseline_shift)
        )

    # ── SERIALIZATION ────────────────────────

    def serialize(self) -> dict:
        return {
            "residue": [
                {
                    "emotion": e.emotion.value,
                    "intensity": e.intensity,
                    "source_description": e.source_description,
                    "source_node": e.source_node,
                    "turns_remaining": e.turns_remaining,
                    "decay_rate": e.decay_rate,
                }
                for e in self.state.residue
            ],
            "mood_baseline_shift": self.state.mood_baseline_shift,
            "rumination_node": self.state.rumination_node,
            "rumination_intensity": self.state.rumination_intensity,
            "turn_count": self._turn_count,
            "focal_history": self._focal_history[-10:],   # keep last 10 turns
        }

    @classmethod
    def deserialize(cls, data: dict) -> "EmotionalPersistenceEngine":
        engine = cls()
        engine._turn_count = data.get("turn_count", 0)
        engine._focal_history = data.get("focal_history", [])

        residue = []
        for rd in data.get("residue", []):
            residue.append(EmotionalResidueEntry(
                emotion=EmotionType(rd["emotion"]),
                intensity=rd["intensity"],
                source_description=rd["source_description"],
                source_node=rd["source_node"],
                turns_remaining=rd["turns_remaining"],
                decay_rate=rd["decay_rate"],
            ))

        engine.state = EmotionalPersistenceState(
            residue=residue,
            mood_baseline_shift=data.get("mood_baseline_shift", 0.0),
            rumination_node=data.get("rumination_node"),
            rumination_intensity=data.get("rumination_intensity", 0.0),
        )
        return engine
