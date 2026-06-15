from pathlib import Path
import os
import sys
from uuid import uuid4

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.cognition_pipeline import CognitionPipeline, TurnInput
from persistence.npc_state_store import NPCStateStore


NODE_EMOTION_TAGS = {
    "mem_001": ["fear", "grief", "dread"],
    "mem_002": ["shame"],
    "mem_003": ["grief", "longing"],
    "mem_004": ["fear", "anger", "shame"],
    "mem_005": ["fear", "relief"],
    "mem_006": ["grief", "longing"],
}

NODE_DESCRIPTIONS = {
    "mem_001": "father drowning in the storm",
    "mem_002": "being caught stealing bread",
    "mem_003": "Old Veth teaching star navigation",
    "mem_004": "crew betrayal and prison",
    "mem_005": "saving a child from fire",
    "mem_006": "Old Veth dying at sea",
}


def run_pipeline_runtime(state_dir: Path):
    os.environ["CHROMA_PERSIST_DIR"] = str(state_dir.parent / "chromadb")

    pipeline = CognitionPipeline(
        npc_id="morgan",
        npc_name="Morgan",
        state_store=NPCStateStore(state_dir),
    )

    output = pipeline.process_turn(
        TurnInput(
            npc_id="morgan",
            player_message="Do storms still bother you?",
            player_trust_level=0.55,
            threat_level=0.1,
            node_emotion_tags=NODE_EMOTION_TAGS,
            node_descriptions=NODE_DESCRIPTIONS,
        )
    )

    print("\nRetrieved memories:")
    for memory_id, score, metadata in output.retrieved_memories:
        print(f"  {memory_id}: semantic={score:.3f} concepts={metadata.get('concepts')}")

    print("\nActivation scores:")
    for memory in output.activated_memories:
        print(
            "  "
            f"{memory.memory_id}: activation={memory.activation_score:.3f} "
            f"semantic={memory.semantic_score:.3f} "
            f"resonance={memory.emotional_resonance:.3f} "
            f"suppression={memory.suppression_pressure:.3f}"
        )

    print("\nSpread chains:")
    for spread in output.spread_results:
        print(
            "  "
            f"{spread.source_memory} -> {spread.target_memory} "
            f"({spread.edge_type}) strength={spread.spread_strength:.3f}"
        )

    print("\nActivated node provenance:")
    for node_id, data in output.activated_nodes.items():
        print(
            "  "
            f"{node_id}: activation={data['activation']:.3f} "
            f"source={data['source']} parent={data['parent']}"
        )

    emotional_state = output.cognitive_state.emotional_state
    print("\nActivation emotional adapter:")
    print(f"  {output.activation_emotional_state}")

    print("\nEmotional state:")
    print(f"  dominant={emotional_state.dominant}")
    print(f"  valence={emotional_state.valence:.3f}")
    print(f"  arousal={emotional_state.arousal:.3f}")
    for reading in emotional_state.readings:
        print(
            "  "
            f"{reading.emotion.value}: intensity={reading.intensity:.3f} "
            f"source={reading.source_node}"
        )

    print("\nCognitive summary:")
    print(output.cognitive_summary_text)

    print("\nFinal NPC response:")
    print(output.npc_response)

    assert output.retrieved_memories
    assert output.activated_memories
    assert output.activated_nodes
    assert output.npc_response


def test_pipeline_runtime(tmp_path):
    runtime_dir = Path(os.getenv("NPC_TEST_RUNTIME_DIR", "C:/tmp/synthetic_npc_tests"))
    run_pipeline_runtime(runtime_dir / f"pipeline_{uuid4().hex}" / "npc_states")


if __name__ == "__main__":
    runtime_dir = Path(os.getenv("NPC_TEST_RUNTIME_DIR", "C:/tmp/synthetic_npc_tests"))
    run_pipeline_runtime(runtime_dir / f"pipeline_{uuid4().hex}" / "npc_states")
