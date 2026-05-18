
"""
Morgan's seeded autobiographical memory graph.

This is not lore. It is a psychological structure.
The graph topology — which memories connect to which, via which edge types —
determines behavioral patterns more than any individual memory's content.

The chain:  storm → father_drowning → helplessness → protectiveness
is not programmed as a rule. It emerges from edge traversal.
"""

from schemas.memory_schema import MemoryNode, MemoryEdge, EdgeType, EventType, PsychologicalEffect


# ═══════════════════════════════════════════════════════════════
# MEMORY NODES
# ═══════════════════════════════════════════════════════════════

MORGAN_MEMORIES: list[MemoryNode] = [

    MemoryNode(
        memory_id="mem_001",
        npc_id="morgan",
        event_type=EventType.TRAUMA,
        age_at_event=9,
        in_world_year="Year of the Crow, 1231",
        season="winter",
        objective_description=(
            "Father Aldric drowned when The Grey Hen capsized in the Strait of Maren. "
            "Morgan was on the dock, jumped in, water was black and freezing. "
            "Could not reach him. Pulled out by a dockhand. Screamed until voiceless."
        ),
        self_narrative_description=(
            "I was too small. Too slow. The water took him before I could get there. "
            "I tried. That is what I tell myself. I tried."
        ),
        # Note: self-narrative already contains defensive reframing ("I tried")
        emotional_weight=9.4,
        valence=-0.95,
        suppression_level=0.75,  # Deeply suppressed — rarely disclosed voluntarily
        sensory_tags=["cold_water", "salt", "screaming", "darkness", "rope", "ice", "storm"],
        people_involved=["Father - Aldric", "Dockhand who pulled him out"],
        location="Strait of Maren docks, Port Carrath",
        confidence_score=0.97,
        distortion_score=0.15,  # slight distortion — "I tried" may be self-protective
        decay_rate=0.005,       # trauma memories decay very slowly
        psychological_effects=[
            PsychologicalEffect("fear_of_deep_water", 0.85),
            PsychologicalEffect("distrust_of_calm_seas", 0.60),
            PsychologicalEffect("suppressed_grief", 0.90),
            PsychologicalEffect("protective_toward_youth", 0.70),
            PsychologicalEffect("helplessness_wound", 0.80),
        ],
        associated_concepts=[
            "father", "drowning", "winter", "loss", "helplessness",
            "dock", "sea", "cold", "darkness", "failure", "grief"
        ],
    ),

    MemoryNode(
        memory_id="mem_002",
        npc_id="morgan",
        event_type=EventType.SHAME,
        age_at_event=11,
        season="summer",
        objective_description=(
            "Caught stealing bread from Merchant Vashe's cart for the third time. "
            "Instead of beating Morgan, Vashe made him work three days unloading barrels. "
            "Hard labor in front of other dock children."
        ),
        self_narrative_description=(
            "Vashe was fair. Fairer than I deserved. The shame of working in front of "
            "the others was worse than any beating. I learned: punishment that respects "
            "you is harder to bear than punishment that doesn't."
        ),
        emotional_weight=5.2,
        valence=-0.4,
        suppression_level=0.2,
        sensory_tags=["sweat", "wood_dust", "crowd", "sun", "heavy_barrels"],
        people_involved=["Merchant Vashe", "dock children"],
        location="Port Carrath docks",
        confidence_score=0.90,
        distortion_score=0.05,
        decay_rate=0.02,
        psychological_effects=[
            PsychologicalEffect("pragmatic_about_theft", 0.60),
            PsychologicalEffect("respect_for_fairness", 0.70),
            PsychologicalEffect("shame_sensitivity", 0.55),
        ],
        associated_concepts=[
            "theft", "shame", "fairness", "labor", "punishment",
            "merchant", "childhood", "dock", "hunger"
        ],
    ),

    MemoryNode(
        memory_id="mem_003",
        npc_id="morgan",
        event_type=EventType.FORMATIVE,
        age_at_event=14,
        season="autumn",
        objective_description=(
            "First mate Old Veth took Morgan to the prow on a clear night and taught "
            "him to navigate by stars. Said: 'Every sailor who drowned lights one of "
            "those. They're watching the ones still floating.' Morgan cried quietly. Veth said nothing."
        ),
        self_narrative_description=(
            "Old Veth gave me the stars. Gave me a way to look at the sea at night "
            "without only thinking of my father. Maybe that was the point."
        ),
        emotional_weight=7.1,
        valence=0.55,
        suppression_level=0.30,
        sensory_tags=["stars", "cold_air", "ship_deck", "night", "salt_wind", "quiet"],
        people_involved=["Old Veth"],
        location="Deck of a merchant vessel, open sea",
        confidence_score=0.92,
        distortion_score=0.10,
        decay_rate=0.01,
        psychological_effects=[
            PsychologicalEffect("stars_as_sacred", 0.80),
            PsychologicalEffect("loyalty_to_veth_memory", 0.75),
            PsychologicalEffect("celestial_navigation_as_ritual", 0.70),
            PsychologicalEffect("grief_partially_reframed", 0.50),
        ],
        associated_concepts=[
            "stars", "navigation", "father", "grief", "veth", "sea",
            "night", "sailors", "death", "belonging", "mentor"
        ],
    ),

    MemoryNode(
        memory_id="mem_004",
        npc_id="morgan",
        event_type=EventType.BETRAYAL,
        age_at_event=24,
        season="spring",
        objective_description=(
            "Morgan's entire crew sold him to the harbormaster's men during the Maren job. "
            "Pre-planned. Collected a bounty. Morgan spent 3 months in Greywood Prison. "
            "One crewman — Declan — had eaten at Morgan's table for two years."
        ),
        self_narrative_description=(
            "Declan. I still see his face sometimes. Not angry. Just... decided. "
            "That was the worst part. He wasn't desperate. He just chose coin over me. "
            "I understand it now. Doesn't mean I've forgiven it."
        ),
        emotional_weight=8.3,
        valence=-0.88,
        suppression_level=0.45,
        sensory_tags=["chains", "cell_smell", "damp_stone", "silence", "Declan_face"],
        people_involved=["Declan", "full Maren crew", "Harbormaster Rensch"],
        location="Greywood Prison, Port Maren",
        confidence_score=0.95,
        distortion_score=0.20,  # Morgan's self-narrative ("I understand it") may be a coping distortion
        decay_rate=0.008,
        psychological_effects=[
            PsychologicalEffect("extreme_trust_vigilance", 0.85),
            PsychologicalEffect("slow_to_commit_loyalty", 0.80),
            PsychologicalEffect("fast_to_end_loyalty_when_broken", 0.90),
            PsychologicalEffect("claustrophobia_mild", 0.45),
        ],
        associated_concepts=[
            "betrayal", "loyalty", "prison", "coin", "trust",
            "crew", "Declan", "captivity", "chains", "isolation"
        ],
    ),

    MemoryNode(
        memory_id="mem_005",
        npc_id="morgan",
        event_type=EventType.ACHIEVEMENT,
        age_at_event=31,
        season="summer",
        objective_description=(
            "Pulled a child from a burning warehouse at Port Serrath docks. "
            "Lost the ring finger on his left hand to a falling beam. "
            "Did not think about it. Just went in."
        ),
        self_narrative_description=(
            "I don't talk about it. People make it into something grand. "
            "It wasn't. She was trapped and I was the one nearest the door."
        ),
        # Self-narrative actively deflects pride — characteristic of Morgan
        emotional_weight=7.8,
        valence=0.60,
        suppression_level=0.65,   # High suppression — Morgan deflects this story
        sensory_tags=["smoke", "heat", "child_crying", "burning_wood", "blood", "pain"],
        people_involved=["unnamed child", "Port Serrath bystanders"],
        location="Port Serrath warehouse docks",
        confidence_score=0.88,
        distortion_score=0.08,
        decay_rate=0.015,
        psychological_effects=[
            PsychologicalEffect("protective_instinct_toward_children", 0.90),
            PsychologicalEffect("acts_without_calculation_when_child_threatened", 0.85),
            PsychologicalEffect("dismisses_own_heroism", 0.70),
        ],
        associated_concepts=[
            "fire", "child", "rescue", "courage", "loss",
            "pain", "instinct", "dock", "smoke", "heroism"
        ],
    ),

    MemoryNode(
        memory_id="mem_006",
        npc_id="morgan",
        event_type=EventType.RELATIONAL,
        age_at_event=19,
        season="winter",
        objective_description=(
            "Old Veth died of fever at sea, three days from port. "
            "Morgan sat with him through the last night. Veth gave Morgan his compass. "
            "Said: 'You know the stars now. You don't need me anymore.'"
        ),
        self_narrative_description=(
            "He was wrong. I still needed him. I just never said so while he could hear it. "
            "I keep the compass but I don't use it. Feels wrong to use it."
        ),
        emotional_weight=8.0,
        valence=-0.70,
        suppression_level=0.60,
        sensory_tags=["fever_smell", "compass", "ship_lantern", "night", "cold_hands"],
        people_involved=["Old Veth"],
        location="At sea, unnamed vessel",
        confidence_score=0.93,
        distortion_score=0.12,
        decay_rate=0.008,
        psychological_effects=[
            PsychologicalEffect("complicated_grief_about_veth", 0.80),
            PsychologicalEffect("compass_as_talisman", 0.75),
            PsychologicalEffect("regret_about_unexpressed_affection", 0.70),
            PsychologicalEffect("stars_now_carry_double_grief", 0.65),
        ],
        associated_concepts=[
            "veth", "death", "mentor", "compass", "grief", "regret",
            "stars", "night", "sea", "fever", "loss", "father"
        ],
    ),

]


# ═══════════════════════════════════════════════════════════════
# MEMORY GRAPH EDGES
# Topology determines emergent behavior — not rules
# ═══════════════════════════════════════════════════════════════

MORGAN_EDGES: list[MemoryEdge] = [

    # Father drowning → Veth teaching stars
    # Veth consciously reframed the drowning grief via stars
    MemoryEdge(
        source_id="mem_001", target_id="mem_003",
        edge_type=EdgeType.REFRAMES,
        weight=0.70,
        notes="Veth's star lesson was a direct response to Morgan's drowning grief"
    ),

    # Veth death → father drowning (double loss reinforcement)
    # Losing Veth re-activates the original abandonment wound
    MemoryEdge(
        source_id="mem_006", target_id="mem_001",
        edge_type=EdgeType.REMINDS_OF,
        weight=0.80,
        notes="Both are primary attachment losses; Veth's death reopens father wound"
    ),

    # Father drowning → warehouse rescue
    # The helplessness of failing to save father drives the instinct to save the child
    MemoryEdge(
        source_id="mem_001", target_id="mem_005",
        edge_type=EdgeType.LEADS_TO,
        weight=0.65,
        notes="Psychological shadow of mem_001 — compulsive action to prevent another drowning"
    ),

    # Betrayal → father drowning (shared abandonment structure)
    MemoryEdge(
        source_id="mem_004", target_id="mem_001",
        edge_type=EdgeType.REMINDS_OF,
        weight=0.55,
        notes="Both involve being left. Different context, same emotional topology."
    ),

    # Stars → both losses (father + Veth)
    # Sensory: seeing stars triggers both griefs simultaneously
    MemoryEdge(
        source_id="mem_003", target_id="mem_001",
        edge_type=EdgeType.ASSOCIATED_SENSORY,
        weight=0.75,
        bidirectional=True,
        notes="Stars are the sensory anchor binding both griefs"
    ),
    MemoryEdge(
        source_id="mem_003", target_id="mem_006",
        edge_type=EdgeType.ASSOCIATED_SENSORY,
        weight=0.80,
        bidirectional=True,
        notes="Stars now carry Veth's death as well as father's"
    ),

    # Betrayal → shame (shame of being fooled)
    MemoryEdge(
        source_id="mem_004", target_id="mem_002",
        edge_type=EdgeType.REMINDS_OF,
        weight=0.40,
        notes="Being caught by Vashe and being sold by Declan share the shame of being outmaneuvered"
    ),

    # Drowning fear → rescue (driven to act when water/helplessness involved)
    MemoryEdge(
        source_id="mem_001", target_id="mem_005",
        edge_type=EdgeType.TRIGGERS_FEAR,
        weight=0.60,
        notes="Fear of helplessness drives compulsive protective action"
    ),

    # Veth loss → stars (the memory of stars carries his death now)
    MemoryEdge(
        source_id="mem_006", target_id="mem_003",
        edge_type=EdgeType.UNRESOLVED_CONFLICT,
        weight=0.70,
        notes="Morgan taught by Veth to love stars; Veth's death made that love painful"
    ),

]


def get_morgan_memories() -> tuple[list[MemoryNode], list[MemoryEdge]]:
    return MORGAN_MEMORIES, MORGAN_EDGES
