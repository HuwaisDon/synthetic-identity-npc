"""
Spreading activation system.

Activated memories spread activation to connected memories
through associative emotional links.

This simulates:
- intrusive associations
- emotional chaining
- autobiographical recall cascades
- thematic cognition

Example:

storm
→ father drowning
→ helplessness
→ protectiveness
→ distrust
"""

from dataclasses import dataclass
from typing import List

import networkx as nx

from memory.activation import ActivatedMemory


@dataclass
class SpreadResult:
    source_memory: str
    target_memory: str
    edge_type: str
    spread_strength: float


class SpreadingActivationEngine:

    def __init__(self):

        self.graph = nx.DiGraph()

        self._build_morgan_graph()

    def _build_morgan_graph(self):

        """
        Initial autobiographical topology.
        Later this becomes dynamically loaded.
        """

        # Nodes
        memories = [
            "mem_001",
            "mem_002",
            "mem_003",
            "mem_004",
            "mem_005",
            "mem_006",
        ]

        for memory in memories:
            self.graph.add_node(memory)

        # Emotional / autobiographical links

        self.graph.add_edge(
            "mem_001",
            "mem_003",
            edge_type="grief_association",
            weight=0.8
        )

        self.graph.add_edge(
            "mem_001",
            "mem_006",
            edge_type="father_association",
            weight=0.9
        )

        self.graph.add_edge(
            "mem_006",
            "mem_003",
            edge_type="mentor_loss",
            weight=0.7
        )

        self.graph.add_edge(
            "mem_001",
            "mem_005",
            edge_type="protective_response",
            weight=0.6
        )

    def spread(
        self,
        activated_memories: List[ActivatedMemory],
        decay_factor: float = 0.7,
        min_spread_threshold: float = 0.25
    ) -> List[SpreadResult]:

        spread_results = []

        for activated in activated_memories:

            source_memory = activated.memory_id
            source_activation = activated.activation_score

            # Get connected memories
            neighbors = self.graph.neighbors(source_memory)

            for target_memory in neighbors:

                edge_data = self.graph.get_edge_data(
                    source_memory,
                    target_memory
                )

                edge_weight = edge_data["weight"]
                edge_type = edge_data["edge_type"]

                # Spread weakens over traversal
                spread_strength = (
                    source_activation
                    * edge_weight
                    * decay_factor
                )

                if spread_strength >= min_spread_threshold:

                    spread_results.append(
                        SpreadResult(
                            source_memory=source_memory,
                            target_memory=target_memory,
                            edge_type=edge_type,
                            spread_strength=spread_strength
                        )
                    )

        # Strongest spreads first
        spread_results.sort(
            key=lambda x: x.spread_strength,
            reverse=True
        )

        return spread_results