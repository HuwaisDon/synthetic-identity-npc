from memory.memory_store import MemoryStore
from memory.morgan_memories import MORGAN_MEMORIES
from memory.activation import ActivationEngine


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

for memory in activated:

    print("\n===================")

    print("Memory:", memory.memory_id)

    print("Semantic Score:", round(memory.semantic_score, 3))

    print("Activation Score:", round(memory.activation_score, 3))

    print(
        "Suppression Pressure:",
        round(memory.suppression_pressure, 3)
    )

    print(
        "Concepts:",
        memory.metadata.get("concepts")
    )