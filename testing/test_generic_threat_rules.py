import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engines.self_concept_defense import SelfConceptDefenseSystem
from schemas.character_schema import ThreatRule
from schemas.cognitive_schemas import EmotionalState, EmotionType, EmotionalReading, GoalState


def test_threat_rules_match_memory_metadata_instead_of_node_ids():
    system = SelfConceptDefenseSystem(
        identity_claims=["I am loyal", "I protect others"],
        threat_rules=[
            ThreatRule(
                trigger_concepts=["betrayal", "panic"],
                threatened_claim_idx=1,
                threat_type="betrayal",
                defense_mechanism="reframe",
            )
        ],
    )

    emotional_state = EmotionalState(
        dominant=EmotionalReading(emotion=EmotionType.SHAME, intensity=0.7),
    )
    goal_state = GoalState()

    threats = system._identify_threats(
        node_id="mem_007",
        activation=0.8,
        emotional_state=emotional_state,
    )

    assert threats == []

    activated_nodes = {
        "mem_007": {
            "activation": 0.8,
            "source": "direct",
            "parent": None,
            "associated_concepts": ["betrayal", "panic"],
            "sensory_tags": ["cold"],
        }
    }

    system = SelfConceptDefenseSystem(
        identity_claims=["I am loyal", "I protect others"],
        threat_rules=[
            ThreatRule(
                trigger_concepts=["betrayal", "panic"],
                threatened_claim_idx=1,
                threat_type="betrayal",
                defense_mechanism="reframe",
            )
        ],
    )

    class FakeNode:
        def __init__(self, node_id, metadata):
            self.node_id = node_id
            self.metadata = metadata

    # The generic rule matcher should use metadata, not node IDs.
    fake_node = FakeNode("mem_007", {"associated_concepts": ["betrayal"], "sensory_tags": ["cold"]})
    match = system._matches_rule(fake_node, system.threat_rules[0])
    assert match is True
