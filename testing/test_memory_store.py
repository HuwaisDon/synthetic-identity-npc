from memory.memory_store import MemoryStore
from memory.morgan_memories import MORGAN_MEMORIES

store = MemoryStore("morgan")

store.load_all(MORGAN_MEMORIES)

results = store.semantic_search(
    "storm at sea"
)

print(results)