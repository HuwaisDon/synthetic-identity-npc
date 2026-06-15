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


def run_suppression_runtime(state_dir: Path):
    os.environ["CHROMA_PERSIST_DIR"] = str(state_dir.parent / "chromadb")

    pipeline = CognitionPipeline(
        npc_id="morgan",
        npc_name="Morgan",
        state_store=NPCStateStore(state_dir),
    )

    output = pipeline.process_turn(
        TurnInput(
            npc_id="morgan",
            player_message="Tell me what happened to your father.",
            player_trust_level=0.45,
            threat_level=0.15,
            node_emotion_tags=NODE_EMOTION_TAGS,
            node_descriptions=NODE_DESCRIPTIONS,
        )
    )

    print("\nActivated memories:")
    for memory in output.activated_memories:
        print(
            "  "
            f"{memory.memory_id}: activation={memory.activation_score:.3f} "
            f"suppression_pressure={memory.suppression_pressure:.3f}"
        )

    print("\nSuppression outputs:")
    for result in output.suppression_results:
        print(
            "  "
            f"{result.node_id}: strength={result.suppression_strength:.3f} "
            f"inhibition={result.disclosure_inhibition:.3f} "
            f"leakage={result.behavioral_leakage:.3f} "
            f"deflect={result.deflection_probability:.3f} "
            f"tension={result.emotional_tension:.3f}"
        )

    print("\nModified attention priorities:")
    for node_id, score in sorted(
        output.attention_activation_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        raw = output.activated_nodes[node_id]["activation"]
        print(f"  {node_id}: raw={raw:.3f} attention={score:.3f}")

    emotional_state = output.cognitive_state.emotional_state
    print("\nEmotional state:")
    print(f"  dominant={emotional_state.dominant}")
    print(f"  valence={emotional_state.valence:.3f}")
    print(f"  arousal={emotional_state.arousal:.3f}")
    for reading in emotional_state.readings:
        print(
            "  "
            f"{reading.emotion.value}: intensity={reading.intensity:.3f} "
            f"source={reading.source_node} suppressed={reading.is_suppressed}"
        )

    print("\nAttention state:")
    attention = output.cognitive_state.attention_state
    print(f"  focal={attention.focal_nodes}")
    print(f"  suppressed_from_attention={attention.suppressed_from_attention}")
    print(f"  load={attention.cognitive_load:.3f}")

    print("\nCognitive summary:")
    print(output.cognitive_summary_text)

    print("\nFinal NPC response:")
    print(output.npc_response)

    assert output.activated_memories
    assert output.suppression_results
    assert output.attention_activation_scores
    assert output.npc_response


def test_suppression_runtime(tmp_path):
    runtime_dir = Path(os.getenv("NPC_TEST_RUNTIME_DIR", "C:/tmp/synthetic_npc_tests"))
    run_suppression_runtime(runtime_dir / f"suppression_{uuid4().hex}" / "npc_states")


if __name__ == "__main__":
    runtime_dir = Path(os.getenv("NPC_TEST_RUNTIME_DIR", "C:/tmp/synthetic_npc_tests"))
    run_suppression_runtime(runtime_dir / f"suppression_{uuid4().hex}" / "npc_states")
