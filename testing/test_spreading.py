from memory.memory_store import MemoryStore
from memory.morgan_memories import MORGAN_MEMORIES
from memory.activation import ActivationEngine
from memory.spreading import SpreadingActivationEngine


store = MemoryStore("morgan")

store.load_all(MORGAN_MEMORIES)

retrieved = store.semantic_search(
    "storm at sea"
)

emotional_state = {
    "sadness": 0.8,
    "fear": 0.6,
    "anger": 0.2
}

activation_engine = ActivationEngine()

activated = activation_engine.compute_activation(
    retrieved,
    emotional_state
)

spreading_engine = SpreadingActivationEngine()

spread_results = spreading_engine.spread(
    activated
)

for result in spread_results:

    print("\n====================")

    print("Source:", result.source_memory)

    print("Target:", result.target_memory)

    print("Edge Type:", result.edge_type)

    print(
        "Spread Strength:",
        round(result.spread_strength, 3)
    )