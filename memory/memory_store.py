"""
ChromaDB memory store — the semantic proximity layer.

This handles: "what memories are topically similar to this input?"
The memory graph handles: "given these activated memories, what else gets pulled in?"

Both are needed. Neither alone is sufficient.
"""

import os

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

from loguru import logger
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from schemas.memory_schema import MemoryNode


EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_CHROMA_PERSIST_DIR = "./data/chromadb"

# Lazy-load embedding model — don't load until first use (saves RAM)
_embed_model: SentenceTransformer | None = None


def disable_chroma_telemetry() -> None:
    """
    Keep Chroma's optional telemetry from breaking local runtime tests.

    Some Chroma/PostHog version combinations call posthog.capture with an
    outdated signature. Telemetry should never affect cognition runtime.
    """
    try:
        import posthog

        posthog.disabled = True
        posthog.capture = lambda *args, **kwargs: None
    except Exception:
        pass


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def embed_text(text: str) -> list[float]:
    """Embed a single text. CPU-friendly — no batching needed for game interactions."""
    model = get_embed_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


class MemoryStore:
    """
    ChromaDB-backed semantic memory store.

    Collection per NPC — memories are private and separate.
    Persists to disk between sessions.
    """

    def __init__(self, npc_id: str, persist_dir: str | None = None):

        self.npc_id = npc_id
        self.collection_name = f"memories_{npc_id}"
        self.persist_dir = persist_dir or os.getenv(
            "CHROMA_PERSIST_DIR",
            DEFAULT_CHROMA_PERSIST_DIR,
        )

        # Ensure persistence path is a DIRECTORY
        if os.path.exists(self.persist_dir):

            # If somehow a file exists with same name, remove it
            if os.path.isfile(self.persist_dir):
                os.remove(self.persist_dir)

        else:
            os.makedirs(self.persist_dir, exist_ok=True)

        disable_chroma_telemetry()

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            f"MemoryStore initialized for {npc_id} "
            f"({self.collection.count()} memories loaded)"
        )

    def upsert_memory(self, memory: MemoryNode) -> None:
        """Add or update a memory in the store."""
        # Build the text representation for embedding
        # Combines description + concepts for richer semantic matching
        embed_text_content = (
            f"{memory.objective_description} "
            f"{' '.join(memory.associated_concepts)} "
            f"{' '.join(memory.sensory_tags)}"
        )

        embedding = embed_text(embed_text_content)

        self.collection.upsert(
            ids=[memory.memory_id],
            embeddings=[embedding],
            documents=[memory.objective_description],
            metadatas=[{
                "npc_id": memory.npc_id,
                "event_type": memory.event_type.value,
                "age_at_event": memory.age_at_event,
                "emotional_weight": memory.emotional_weight,
                "valence": memory.valence,
                "suppression_level": memory.suppression_level,
                "current_strength": memory.current_strength,
                "concepts": ",".join(memory.associated_concepts),
                "sensory_tags": ",".join(memory.sensory_tags),
            }],
        )

    def load_all(self, memories: list[MemoryNode]) -> None:
        """Bulk load memories — only upserts new ones."""
        existing_ids = set()
        if self.collection.count() > 0:
            existing = self.collection.get()
            existing_ids = set(existing["ids"])

        for memory in memories:
            if memory.memory_id not in existing_ids:
                self.upsert_memory(memory)
                logger.debug(f"Loaded memory: {memory.memory_id}")

        logger.info(f"MemoryStore: {self.collection.count()} total memories for {self.npc_id}")

    def semantic_search(
        self,
        query_text: str,
        n_results: int = 4,
        min_similarity: float = 0.25,
    ) -> list[tuple[str, float, dict]]:
        """
        Semantic similarity search.
        Returns (memory_id, similarity_score, metadata) tuples.
        """
        if self.collection.count() == 0:
            return []

        query_embedding = embed_text(query_text)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self.collection.count()),
            include=["distances", "metadatas", "documents"],
        )

        output = []
        for i, memory_id in enumerate(results["ids"][0]):
            # ChromaDB returns cosine distance (0=identical, 2=opposite)
            # Convert to similarity: 1 - (distance/2)
            distance = results["distances"][0][i]
            similarity = 1.0 - (distance / 2.0)

            if similarity >= min_similarity:
                output.append((
                    memory_id,
                    similarity,
                    results["metadatas"][0][i],
                ))

        return output

    def count(self) -> int:
        return self.collection.count()
